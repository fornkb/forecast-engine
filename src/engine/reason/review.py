"""Monthly review orchestration, bounded research, and durable finalization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.assistant import (
    PROMPT_VERSION,
    AnthropicAssistant,
    AssistantError,
    LookupInput,
    ReadRecord,
    Transport,
    TriageOutput,
    TriageWork,
    json_text,
)
from engine.reason.collect import (
    _check_snapshot,
    _config_digest,
    collect_evidence,
    load_evidence_batch,
    lookup_scope,
    make_query_scope,
    recover_review_attempts,
)
from engine.reason.context import get_forecast_snapshot
from engine.reason.erp import PostgresFetch, fetch_postgres_context, require_reasoning
from engine.reason.facts import require_client
from engine.reason.integrity import make_seal
from engine.reason.providers import HttpTransport, http_get
from engine.reason.questions import record_question
from engine.reason.review_inputs import (
    compact_evidence,
    compact_fact,
    group_packet,
    latest_checks,
    load_inputs,
)
from engine.reason.scenarios import (
    METHOD_VERSION,
    calculate_assumption,
    calculate_evidence,
    direction,
    validate_proposal,
)
from engine.reason.schemas import (
    Assessment,
    AssessmentProposal,
    ContextError,
    GroupResult,
    MappingProposal,
    ModelCall,
    QueryRequestContext,
    ReviewManifest,
    ReviewQuestion,
    ScenarioScope,
    content_digest,
)
from engine.reason.triage import map_event, news_events, validate_triage
from engine.store.duckdb_store import EngineStore

_LEASE = timedelta(seconds=150)


def load_review(
    store: EngineStore, review_id: str, *, revision: int | None = None
) -> ReviewManifest:
    if store.path.exists():
        with store.connect() as con:
            manifest = records.read_review(con, store.client_id, review_id, revision)
            if manifest:
                return manifest
    raise ContextError(f"Review {review_id!r} was not found for {store.client_id!r}")


def _review_config(profile: ClientProfile) -> str:
    return content_digest(
        [
            _config_digest(profile),
            require_reasoning(profile).model_dump(mode="json", exclude={"evidence", "evaluation"}),
            PROMPT_VERSION,
            METHOD_VERSION,
        ]
    )


def _manifest(profile, snapshot, batch, followup_limit, review_id):
    config = require_reasoning(profile)
    now = datetime.now(UTC)
    return ReviewManifest(
        review_id=review_id,
        client_id=profile.client_id,
        forecast_run_id=snapshot.forecast_run_id,
        snapshot_id=snapshot.snapshot_id,
        base_batch_id=batch.batch_id,
        forecast_digest=snapshot.input_digest,
        collection_config_digest=batch.config_digest,
        budget_owner_id=review_id,
        revision=1,
        started_at=now,
        collection_cutoff=batch.knowledge_cutoff,
        knowledge_cutoff=batch.knowledge_cutoff,
        config_digest=_review_config(profile),
        prompt_version=PROMPT_VERSION,
        method_version=METHOD_VERSION,
        model_id=config.assistant.model_id,
        model_request_limit=config.assistant.max_requests,
        followup_limit=followup_limit,
        questions=(),
        gaps=batch.gaps,
        approved_lookups=_approved_tools(profile),
        source_access={source.id: source.kind for source in config.evidence.erp},
        provider_rates=config.evaluation.rates,
    )


def _question_from_fact(question) -> ReviewQuestion:
    return ReviewQuestion(
        question_id=question.question_id,
        purpose="company_fact",
        message=question.spec.message,
        subject=question.spec.subject,
        months=question.spec.affected_months,
        evidence_ids=question.spec.evidence_ids,
    )


class ReviewJob:
    """One leased writer. Every transaction verifies the lease before advancing the head."""

    def __init__(self, profile, store, manifest, token, *, transport=None, http=None, fetch=None):
        self.profile = profile
        self.store = store
        self.manifest = manifest
        self.token = token
        self.snapshot = get_forecast_snapshot(store, manifest.snapshot_id)
        self.batch = load_evidence_batch(store, manifest.base_batch_id)
        _check_snapshot(profile, self.snapshot, manifest.forecast_run_id)
        if (
            self.batch.snapshot_id != manifest.snapshot_id
            or self.batch.forecast_run_id != manifest.forecast_run_id
            or self.snapshot.input_digest != manifest.forecast_digest
            or self.batch.config_digest != manifest.collection_config_digest
        ):
            raise ContextError("Review, base evidence and forecast bindings disagree")
        self.assistant = AnthropicAssistant(require_reasoning(profile).assistant, transport)
        self.http = http or http_get
        self.fetch = fetch or fetch_postgres_context
        self.months = tuple(sorted({row.period for row in self.snapshot.rows}))

    @contextmanager
    def write(self) -> Iterator:
        expected = self.manifest
        try:
            with records.write_transaction(self.store) as con:
                changed = con.execute(
                    """UPDATE reason_review_heads SET lease_until = ?
                    WHERE client_id = ? AND review_id = ? AND lease_id = ? AND revision = ?
                    RETURNING revision""",
                    [
                        datetime.now(UTC) + _LEASE,
                        expected.client_id,
                        expected.review_id,
                        self.token,
                        expected.revision,
                    ],
                ).fetchone()
                if changed is None:
                    raise ContextError(
                        "Review changed while work was in progress. Resume its latest revision."
                    )
                yield con
        except BaseException:
            self.manifest = expected
            raise

    def advance(self, con, **updates) -> None:
        manifest = self.manifest
        if manifest.state != "draft":
            raise ContextError("A finalized review needs a successor")
        updates = {key: value for key, value in updates.items() if getattr(manifest, key) != value}
        if updates:
            manifest = manifest.model_copy(update={"revision": manifest.revision + 1, **updates})
            records.save_review(con, manifest)
            self.manifest = manifest

    def sync(self, *, resume: bool = False) -> None:
        with self.write() as con:
            current = self.manifest
            if resume:
                recover_review_attempts(
                    con, current.client_id, current.budget_owner_id, current.requested_scopes
                )
            checks = records.read_checks(
                con, (current.client_id, "review", current.budget_owner_id)
            )
            ids = tuple(
                dict.fromkeys(
                    key
                    for check in checks
                    for key in check.evidence_ids
                    if key not in self.batch.evidence_ids
                )
            )
            attempts = records.read_lookup_attempts(con, current.client_id, current.budget_owner_id)
            calls = records.read_model_calls(con, current.client_id, current.budget_owner_id)
            if resume:
                for call in calls:
                    if call.ended_at is None:
                        records.save_model_call(
                            con,
                            current.client_id,
                            current.budget_owner_id,
                            call.model_copy(
                                update={
                                    "ended_at": datetime.now(UTC),
                                    "outcome": "indeterminate_model_request",
                                }
                            ),
                        )
                calls = records.read_model_calls(con, current.client_id, current.budget_owner_id)
            cutoff = max((current.knowledge_cutoff, *(check.ended_at for check in checks)))
            self.advance(
                con,
                knowledge_cutoff=cutoff,
                appended_check_ids=tuple(check.check_id for check in checks),
                appended_evidence_ids=ids,
                lookup_attempts=attempts,
                model_calls=calls,
            )

    def invoke(
        self,
        stage,
        work_id,
        packet,
        response_type,
        *,
        validate,
        evidence=None,
        facts=None,
        allow_lookup=True,
    ):
        supplied = evidence if evidence is not None else {}
        pinned = facts if facts is not None else {}
        can_lookup = allow_lookup and self.manifest.followup_limit > 0

        def reserve(request_kind: str) -> str:
            with self.write() as con:
                current = self.manifest
                calls = records.read_model_calls(con, current.client_id, current.budget_owner_id)
                if len(calls) >= current.model_request_limit:
                    raise AssistantError("model_request_budget_exhausted")
                call = ModelCall(
                    call_id=str(uuid4()),
                    stage=stage,
                    work_id=work_id,
                    started_at=datetime.now(UTC),
                    request_kind=request_kind,
                )
                records.save_model_call(con, current.client_id, current.budget_owner_id, call)
                return call.call_id

        def record(call_id, message, outcome) -> None:
            with self.write() as con:
                current = self.manifest
                calls = records.read_model_calls(con, current.client_id, current.budget_owner_id)
                call = next(item for item in calls if item.call_id == call_id)
                updates = {"ended_at": datetime.now(UTC), "outcome": outcome[:160]}
                if message is not None:
                    updates.update(message.usage.model_dump())
                    updates["token_usage_reported"] = message.usage_reported
                    if message.tool and message.tool.name in {
                        "submit_result",
                        "lookup_scope",
                        "read_record",
                    }:
                        updates["response_tools"] = (message.tool.name,)
                records.save_model_call(
                    con, current.client_id, current.budget_owner_id, call.model_copy(update=updates)
                )

        def dispatch(name: str, args: dict):
            if name == "read_record":
                identifier = ReadRecord.model_validate(args).record_version_id
                if identifier in supplied:
                    return compact_evidence(supplied[identifier])
                if identifier in pinned:
                    return compact_fact(pinned[identifier])
                raise ContextError("record_id_outside_packet")
            if name != "lookup_scope" or not can_lookup:
                raise ContextError("tool_not_allowed")
            lookup = LookupInput.model_validate(args)
            if lookup.client_id not in (None, self.store.client_id):
                raise ContextError("lookup_client_scope_mismatch")
            if set(lookup.months) - set(self.months):
                raise ContextError("lookup_month_outside_forecast")
            scope = make_query_scope(
                self.profile,
                lookup.source_kind,
                lookup.source_id,
                lookup.months,
                page=lookup.page,
                location_role=lookup.location_role,
            )
            current = self.manifest
            with self.write() as con:
                if scope not in current.requested_scopes:
                    self.advance(con, requested_scopes=(*current.requested_scopes, scope))
            result = lookup_scope(
                self.profile,
                self.store,
                scope,
                QueryRequestContext(
                    client_id=current.client_id,
                    forecast_run_id=current.forecast_run_id,
                    snapshot_id=current.snapshot_id,
                    review_id=current.budget_owner_id,
                    months=self.months,
                    request_budget=current.followup_limit,
                ),
                http=self.http,
                fetch=self.fetch,
            )
            self.sync()
            state = self.inputs()
            supplied.update({key: state.evidence[key] for key in result.evidence_ids})
            return {
                "checks": [
                    check.model_dump(mode="json", exclude={"scope"}) for check in result.checks
                ],
                "record_ids": list(result.evidence_ids),
                "attempts_used": result.attempts_used,
                "attempts_allowed": current.followup_limit,
            }

        try:
            return self.assistant.invoke(
                stage,
                packet,
                response_type,
                reserve=reserve,
                record=record,
                validate=validate,
                dispatch=dispatch if stage != "clarification" else None,
                allow_lookup=can_lookup,
            )
        finally:
            self.sync()

    def inputs(self):
        return load_inputs(self.store, self.profile, self.snapshot, self.batch, self.manifest)

    def cached(self, work_id, digest):
        with self.store.connect() as con:
            return records.read_review_work(con, self.manifest, work_id, digest)

    def save_work(self, work_id, digest, value):
        with self.write() as con:
            records.save_review_work(con, self.manifest, work_id, digest, value.model_dump_json())


@contextmanager
def open_review_job(profile, store, review_id, **dependencies):
    require_client(profile, store)
    token = str(uuid4())
    with records.write_transaction(store) as con:
        manifest = records.read_review(con, store.client_id, review_id)
        if manifest is None or manifest.state != "draft":
            raise ContextError("Review must be an existing draft")
        if manifest.config_digest != _review_config(profile):
            raise ContextError(
                "Review settings have changed. Restore the pinned settings to resume."
            )
        claimed = con.execute(
            """UPDATE reason_review_heads SET lease_id = ?, lease_until = ?
            WHERE client_id = ? AND review_id = ? AND (lease_until IS NULL OR lease_until <= ?)
            RETURNING revision""",
            [token, datetime.now(UTC) + _LEASE, store.client_id, review_id, datetime.now(UTC)],
        ).fetchone()
        if claimed is None:
            raise ContextError("Review is already in progress. Resume after its lease expires.")
    try:
        job = ReviewJob(profile, store, manifest, token, **dependencies)
        job.sync(resume=True)
        yield job
    finally:
        with records.write_transaction(store) as con:
            con.execute(
                """UPDATE reason_review_heads SET lease_id = NULL, lease_until = NULL
                WHERE client_id = ? AND review_id = ? AND lease_id = ?""",
                [store.client_id, review_id, token],
            )


def _approved_tools(profile) -> dict:
    config = require_reasoning(profile).evidence
    return {
        "erp": [source.id for source in config.erp],
        "news": [search.id for search in config.news.searches] if config.news else [],
        "weather": [
            dict(source_id=item.location_id, location_role=item.role)
            for item in config.weather.locations
            if item.role in config.weather.applicable_roles
        ]
        if config.weather
        else [],
        "enso": ["noaa_cpc"] if config.enso else [],
    }


def _triage_packets(job, inputs, events):
    facts = {key: value for key, value in inputs.facts.items() if key in inputs.active_fact_ids}
    evidence = {
        key: item
        for key, item in inputs.evidence.items()
        if item.operational and item.operational.dataset in {"bom", "vendor_link"}
    }
    context = [
        *[("facts", key, compact_fact(fact)) for key, fact in sorted(facts.items())],
        *[
            ("operating_links", key, compact_evidence(item))
            for key, item in sorted(evidence.items())
        ],
    ]
    budget = job.assistant.config.max_input_bytes
    shared = {
        "months": [month.isoformat() for month in job.months],
        "configured_lookups": _approved_tools(job.profile) if job.manifest.followup_limit else {},
        "facts": [],
        "operating_links": [],
    }
    shown_facts, shown_evidence = {}, {}
    remaining = budget // 3
    for field, key, value in context:
        size = len(json_text(value).encode())
        if size > remaining:
            continue
        remaining -= size
        shared[field].append(value)
        if field == "facts":
            shown_facts[key] = facts[key]
        else:
            shown_evidence[key] = evidence[key]
    shared["context_truncated"] = len(shown_evidence) + len(shown_facts) < len(context)
    batch, payload, used = [], [], 0
    for event in events:
        value = {
            "event_id": event.event_id,
            "articles": [
                {
                    "article": compact_evidence(inputs.evidence[item.evidence_id]),
                    "possible_entities": [entity.model_dump() for entity in item.entities],
                    "corroborating_ids": item.corroborating_ids,
                    "duplicate_ids": item.duplicate_ids,
                }
                for item in event.candidates
            ],
        }
        size = len(json_text(value).encode())
        if batch and (used + size > budget // 3 or len(batch) >= 16):
            yield tuple(batch), {**shared, "events": payload}, shown_facts, shown_evidence
            batch, payload, used = [], [], 0
        batch.append(event)
        payload.append(value)
        used += size
    if batch:
        yield tuple(batch), {**shared, "events": payload}, shown_facts, shown_evidence


def _triage(job, inputs, *, offline):
    candidates = inputs.candidates
    events = news_events(candidates, inputs.evidence)
    proposals, failures, gaps = {}, {}, set()
    if offline:
        proposals.update({item.event_id: item for item in job.manifest.mappings})
        failures.update(
            {
                item.event_id: item.terminal_reason
                for item in job.manifest.mappings
                if item.terminal_reason
            }
        )
    else:
        for batch, packet, facts, links in _triage_packets(job, inputs, events):
            work_id = "triage:" + content_digest([event.event_id for event in batch])
            digest = content_digest(packet)
            cached = job.cached(work_id, digest)
            terminal_reason = None
            if cached:
                work = TriageWork.model_validate_json(cached)
                output, terminal_reason = work.output, work.terminal_reason
            else:
                evidence = {
                    **links,
                    **{key: inputs.evidence[key] for event in batch for key in event.article_ids},
                }

                def validate(output, batch=batch, evidence=evidence, facts=facts):
                    validate_triage(output.decisions, batch, inputs.snapshot.rows, evidence, facts)
                    return output

                try:
                    output = job.invoke(
                        "triage",
                        work_id,
                        packet,
                        TriageOutput,
                        validate=validate,
                        evidence=evidence,
                        facts=facts,
                    )
                except AssistantError as exc:
                    terminal_reason = str(exc)
                    output = TriageOutput(
                        decisions=tuple(
                            MappingProposal(
                                event_id=event.event_id,
                                article_ids=event.article_ids,
                                entities=(),
                                months=(),
                                status="uncertain",
                                basis=f"Triage incomplete ({exc}).",
                            )
                            for event in batch
                        )
                    )
                job.save_work(
                    work_id, digest, TriageWork(output=output, terminal_reason=terminal_reason)
                )
            for decision in output.decisions:
                if terminal_reason:
                    failures[decision.event_id] = terminal_reason
                proposals[decision.event_id] = decision

    groups, decisions = [], []
    with job.write() as con:
        questions = {question.question_id: question for question in job.manifest.questions}
        for event in events:
            proposal = proposals.get(event.event_id)
            if proposal is None:
                continue
            decision, selected, unresolved = map_event(
                proposal,
                event,
                inputs.snapshot.rows,
                inputs.evidence,
                inputs.facts,
                inputs.links,
                inputs.operating,
                tuple(questions.values()),
                inputs.active_fact_ids,
            )
            question_ids = []
            for spec in unresolved:
                question = record_question(con, job.store.client_id, spec, datetime.now(UTC))
                questions.setdefault(
                    question.question_id,
                    ReviewQuestion(
                        question_id=question.question_id,
                        purpose="mapping_confirmation",
                        message=spec.message,
                        subject=spec.subject,
                        months=spec.affected_months,
                        related_entity=spec.subject,
                        evidence_ids=event.article_ids,
                    ),
                )
                question_ids.append(question.question_id)
            reason = failures.get(event.event_id)
            decisions.append(
                decision.model_copy(
                    update={
                        "question_ids": tuple(question_ids),
                        "terminal_reason": reason,
                    }
                )
            )
            groups.extend(selected)
            if reason:
                gaps.add(f"triage:{reason}")
            if decision.status == "uncertain":
                gaps.add(f"tentative_mapping:{decision.event_id}")
        current_ids = {identifier for decision in decisions for identifier in decision.question_ids}
        resolved = []
        for question in questions.values():
            if (
                question.purpose == "mapping_confirmation"
                and question.status == "pending"
                and question.question_id not in current_ids
            ):
                question = question.model_copy(
                    update={"status": "cancelled", "resolved_at": datetime.now(UTC)}
                )
            resolved.append(question)
        job.advance(
            con, candidates=candidates, mappings=tuple(decisions), questions=tuple(resolved)
        )
    return tuple(groups), gaps


def _input_key(inputs):
    return (
        tuple(inputs.evidence),
        tuple(inputs.facts),
        tuple(check.check_id for check in latest_checks(inputs.checks)),
    )


def _addition_key(manifest):
    return (
        manifest.appended_fact_version_ids,
        manifest.appended_evidence_ids,
        manifest.appended_check_ids,
    )


def _assess_group(job, inputs, group, previous, *, offline):
    packet, evidence, facts, checks = group_packet(inputs, group)
    if job.manifest.followup_limit:
        packet["configured_lookups"] = _approved_tools(job.profile)
    digest = content_digest(packet)
    work_id = f"group:{group.group_id}"
    cached = job.cached(work_id, digest)
    if cached:
        return GroupResult.model_validate_json(cached)
    inherited = previous.get(group.group_id)
    if inherited is not None and inherited.input_digest == digest:
        return inherited
    reason, proposal = None, None
    if not group.members:
        reason = "verified_membership_missing"
    elif offline:
        reason = "inputs_changed_after_operator_confirmation"
    else:
        # A lookup may expand the available set. Rebuild before using results for changed packets.
        initial = dict(evidence)
        try:
            proposal = job.invoke(
                "assessment",
                work_id,
                packet,
                AssessmentProposal,
                evidence=evidence,
                facts=facts,
                validate=lambda value: validate_proposal(value, group, initial, facts, checks),
            )
        except AssistantError as exc:
            reason = str(exc)
    result = GroupResult(
        group=group, input_digest=digest, proposal=proposal, terminal_reason=reason
    )
    job.save_work(work_id, digest, result)
    return result


def _partial(check, period) -> bool:
    return check.coverage != "complete" or any(
        item.period == period and item.status != "complete" for item in check.months
    )


def _checks_for_range(proposal, checks, inputs, group) -> bool:
    if any(_partial(checks[key], group.period) for key in proposal.required_check_ids):
        return False
    operand_ids = {
        operand.record_version_id
        for condition in proposal.conditions
        for operand in condition.operands.values()
    }
    for key in operand_ids:
        supporting = [
            check
            for check in latest_checks(inputs.checks)
            if key in check.evidence_ids and group.period in check.scope.months
        ]
        if not supporting or not any(not _partial(check, group.period) for check in supporting):
            return False
    return True


def _scenario_question(manifest, group, rows, target, candidates, per_unit):
    by_unit = defaultdict(list)
    for row in rows:
        if row.series_id in candidates and row.unit:
            by_unit[row.unit].append(row)
    active_ids = set()
    for unit, members in by_unit.items():
        names = tuple(sorted(row.series_id for row in members))
        scope = ScenarioScope(
            group_id=group.group_id,
            event_ids=group.event_ids,
            target=target,
            period=group.period,
            unit=unit,
            series_ids=names,
        )
        question_id = "assumption_question_" + content_digest(
            [manifest.client_id, manifest.forecast_run_id, scope.model_dump(mode="json")]
        )
        active_ids.add(question_id)
        per_unit.setdefault(
            question_id,
            ReviewQuestion(
                question_id=question_id,
                purpose="scenario_assumption",
                scope=scope,
                evidence_ids=group.evidence_ids,
                months=(group.period,),
                baselines=tuple(sorted((row.series_id, row.baseline) for row in members)),
                message=(
                    "Confirm low and high signed increments or percentages, "
                    "the conditions, and why each effect is incremental to this forecast."
                ),
            ),
        )
    return active_ids


def materialize_assessments(manifest, inputs, group_results, manifest_revision):
    assessments = []
    questions = {question.question_id: question for question in manifest.questions}
    assumptions = {item.question_id: item for item in manifest.assumptions}
    active_questions = set()
    target = inputs.snapshot.target.id if inputs.snapshot.target else None
    for result in group_results:
        group, proposal = result.group, result.proposal
        _, evidence, _, checks = group_packet(inputs, group)
        members = {item.series_id for item in group.members}
        rows = tuple(
            row
            for row in inputs.snapshot.rows
            if row.period == group.period and row.series_id in members
        )
        asked = set()
        for row in rows:
            relevant = proposal is not None and (
                proposal.series_ids is None or row.series_id in proposal.series_ids
            )
            channel = (
                proposal.target_type
                if relevant
                else ("material_availability" if group.kind == "supply" else "demand")
            )
            exact_target = (target or "unknown") if channel == "demand" else channel
            gaps, conditions, method, assumption = list(group.flags), (), None, None
            finding, directed = "incomplete_review", "unknown"
            rationale = result.terminal_reason or "This row was outside the assessed subset."
            cited, facts, contrary, required = group.evidence_ids, group.fact_version_ids, (), ()
            if relevant:
                cited = tuple(
                    sorted(
                        set(group.evidence_ids + proposal.evidence_ids)
                        | {
                            ref.record_version_id
                            for condition in proposal.conditions
                            for ref in condition.operands.values()
                        }
                    )
                )
                facts = tuple(sorted(set(facts + proposal.fact_version_ids)))
                contrary, required = proposal.contrary_ids, proposal.required_check_ids
                rationale, directed = proposal.rationale, proposal.direction
                finding = proposal.finding
                if finding == "no_material_change_found" and not (
                    proposal.evidence_ids or required
                ):
                    finding = "incomplete_review"
                    gaps.append("no_material_change_requires_evidence_or_checks")
                if finding == "no_material_change_found" and (
                    "partial_regional_outlook" in group.flags
                    or "incomplete_supply_inputs" in group.flags
                ):
                    finding = "incomplete_review"
                if (
                    finding == "no_material_change_found"
                    and group.shared_shortfall is not None
                    and channel in {"material_availability", "production", "shipment_capacity"}
                ):
                    finding = "possible_impact"
                    gaps.append("projected_material_shortfall")
                if proposal.missing_facts:
                    gaps.extend(proposal.missing_facts)
                needed = [checks[key] for key in required if key in checks]
                if any(
                    _partial(check, group.period) or check.status == "unavailable"
                    for check in needed
                ):
                    finding = "incomplete_review"
                    gaps.append("required_source_coverage_incomplete")
                if finding == "what_if_range":
                    try:
                        if not target or not _checks_for_range(proposal, checks, inputs, group):
                            raise ContextError("numeric_operand_coverage_or_target_unknown")
                        conditions = calculate_evidence(
                            proposal, group, row, inputs.snapshot, evidence, target
                        )
                        method = proposal.method_id
                    except ContextError as exc:
                        finding = "possible_impact"
                        gaps.append(str(exc))
                if finding == "possible_impact" and channel == "demand" and target and row.unit:
                    asked.add(row.series_id)
            if exact_target == "unknown":
                finding = "incomplete_review"
                gaps.append("forecast_target_unknown")

            if channel == "demand" and target:
                assumption = next(
                    (
                        item
                        for item in reversed(tuple(assumptions.values()))
                        if item.scope.group_id == group.group_id
                        and item.scope.period == row.period
                        and item.scope.target == target
                        and row.series_id in item.scope.series_ids
                    ),
                    None,
                )
            if assumption:
                if (
                    assumption.scope.event_ids != group.event_ids
                    or not set(assumption.scope.series_ids) <= members
                    or assumption.scope.unit != row.unit
                ):
                    gaps.append("confirmed_assumption_scope_changed")
                    assumption = None
                else:
                    conditions = calculate_assumption(assumption, row)
                    method = assumption.values.method_id
                    finding, rationale = "what_if_range", assumption.values.rationale
            if conditions:
                directed = direction(conditions)
            scenario_id = "scenario_" + content_digest([group.group_id, exact_target])
            member = next(item for item in group.members if item.series_id == row.series_id)
            assessments.append(
                Assessment(
                    review_id=manifest.review_id,
                    manifest_revision=manifest_revision,
                    input_digest=content_digest(
                        [result.input_digest, assumption.assumption_id if assumption else None]
                    ),
                    scenario_id=scenario_id,
                    group_id=group.group_id,
                    event_ids=group.event_ids,
                    series_id=row.series_id,
                    item_id=row.item_id,
                    customer_id=row.customer_id,
                    period=row.period,
                    horizon=row.horizon,
                    model=row.model,
                    target=exact_target,
                    target_type=channel,
                    unit=row.unit
                    if channel == "demand"
                    else (group.unit if channel == "material_availability" else None),
                    baseline=row.baseline if channel == "demand" else None,
                    forecast_unit=row.unit,
                    forecast_baseline=row.baseline,
                    finding=finding,
                    direction=directed,
                    method_id=method,
                    method_version=METHOD_VERSION if method else None,
                    conditions=conditions,
                    assumption_id=assumption.assumption_id if assumption else None,
                    evidence_ids=cited,
                    fact_version_ids=facts,
                    contrary_ids=contrary,
                    required_check_ids=required,
                    gaps=tuple(dict.fromkeys(gaps)),
                    rationale=rationale,
                    supply_shortfall=member.shortfall
                    if member.shortfall is not None
                    else group.shared_shortfall,
                    supply_balance=-group.shared_shortfall
                    if group.shared_shortfall is not None
                    else None,
                    supply_balance_scope="material_at_site"
                    if group.shared_shortfall is not None
                    else None,
                    supply_unit=group.unit if group.kind == "supply" else None,
                    supply_at=group.event_at if group.kind == "supply" else None,
                )
            )
        if asked:
            active_questions.update(
                _scenario_question(manifest, group, rows, target, asked, questions)
            )
    return tuple(assessments), tuple(
        question.model_copy(update={"status": "cancelled", "resolved_at": datetime.now(UTC)})
        if question.purpose == "scenario_assumption"
        and question.status == "pending"
        and question.question_id not in active_questions
        else question
        for question in questions.values()
    )


def finish_review(job: ReviewJob, *, offline=False, retriage=True) -> ReviewManifest:
    previous = {item.group.group_id: item for item in job.manifest.groups}
    while True:
        inputs = job.inputs()
        original = _input_key(inputs)
        additions = _addition_key(job.manifest)
        news, gaps = _triage(job, inputs, offline=offline or not retriage)
        if _addition_key(job.manifest) != additions and _input_key(job.inputs()) != original:
            retriage = True
            continue
        group_results = []
        for group in sorted((*inputs.groups, *news), key=lambda item: item.group_id):
            result = _assess_group(job, inputs, group, previous, offline=offline)
            if _addition_key(job.manifest) != additions and _input_key(job.inputs()) != original:
                retriage = True
                break
            group_results.append(result)
        else:
            break
    current = job.manifest
    assessments, questions = materialize_assessments(
        current, inputs, group_results, current.revision + 1
    )
    assessed_keys = {(item.series_id, item.period) for item in assessments}
    if len(assessed_keys) < len(inputs.snapshot.rows):
        gaps.add(
            f"rows_without_verified_assessment:{len(inputs.snapshot.rows) - len(assessed_keys)}"
        )
    gaps.update(job.batch.gaps)
    gaps.update(
        f"{check.scope.source_kind}:{check.scope.source_id}:{check.reason or check.status}"
        for check in latest_checks(inputs.checks)
        if check.status != "not_applicable" and check.coverage != "complete"
    )
    gaps.update(
        f"group:{item.group.group_id}:{item.terminal_reason}"
        for item in group_results
        if item.terminal_reason
    )
    if any(item.reason == "request_budget_exhausted" for item in inputs.checks):
        gaps.add("request_budget_exhausted")
    with job.write() as con:
        now = datetime.now(UTC)
        changes = dict(
            state="finalized",
            finalized_at=now,
            knowledge_cutoff=now,
            groups=tuple(group_results),
            questions=questions,
            gaps=tuple(sorted(gaps)),
        )
        final = current.model_copy(update={**changes, "revision": current.revision + 1})
        seal = make_seal(final, job.snapshot, job.batch, inputs, assessments)
        job.advance(con, **changes, seal=seal)
        records.save_assessments(con, current.client_id, assessments)
    return job.manifest


def run_review(
    profile: ClientProfile,
    store: EngineStore,
    forecast_run_id: int,
    *,
    batch_id: str | None = None,
    followup_limit: int | None = None,
    resume_id: str | None = None,
    new_review: bool = False,
    transport: Transport | None = None,
    http: HttpTransport | None = None,
    fetch: PostgresFetch | None = None,
    on_start: Callable[[str], None] | None = None,
    on_collection_start: Callable[[str], None] | None = None,
) -> ReviewManifest:
    require_client(profile, store)
    config = require_reasoning(profile)
    if followup_limit is not None and (
        type(followup_limit) is not int or not 0 <= followup_limit <= 3
    ):
        raise ContextError("Follow-up limit must be between zero and three")
    limit = min(config.evidence.followup_limit, followup_limit if followup_limit is not None else 3)
    if resume_id:
        manifest = load_review(store, resume_id)
        if (
            manifest.forecast_run_id != forecast_run_id
            or (batch_id and manifest.base_batch_id != batch_id)
            or (followup_limit is not None and manifest.followup_limit != limit)
            or new_review
        ):
            raise ContextError("Resume arguments disagree with the saved review")
    else:
        batch = (
            load_evidence_batch(store, batch_id)
            if batch_id
            else collect_evidence(
                profile,
                store,
                forecast_run_id,
                http=http or http_get,
                fetch=fetch or fetch_postgres_context,
                on_start=on_collection_start,
            )
        )
        snapshot = get_forecast_snapshot(store, batch.snapshot_id)
        _check_snapshot(profile, snapshot, forecast_run_id)
        if (
            batch.forecast_run_id != forecast_run_id
            or batch.client_id != store.client_id
            or snapshot.captured_at != batch.snapshot_captured_at
        ):
            raise ContextError("Base batch does not belong to this client and forecast")
        nonce = str(uuid4()) if new_review else None
        review_id = "review_" + content_digest(
            [store.client_id, batch.batch_id, _review_config(profile), limit, nonce]
        )
        with records.write_transaction(store) as con:
            con.execute(records.REASON_DDL)
            con.execute(records.REVIEW_DDL)
            manifest = records.read_review(con, store.client_id, review_id)
            if manifest is None:
                manifest = _manifest(profile, snapshot, batch, limit, review_id)
                questions = tuple(
                    _question_from_fact(question)
                    for key in snapshot.question_ids
                    if (question := records.get_question(con, store.client_id, key))
                )
                manifest = manifest.model_copy(update={"questions": questions})
                records.insert_review(con, manifest)
    if on_start:
        on_start(manifest.review_id)
    if manifest.state == "finalized":
        return manifest
    with open_review_job(
        profile, store, manifest.review_id, transport=transport, http=http, fetch=fetch
    ) as job:
        return finish_review(
            job,
            offline=job.manifest.resume_mode == "operator_offline",
            retriage=job.manifest.resume_mode == "review",
        )
