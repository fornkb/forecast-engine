"""Review-scoped operator confirmations and offline assumption recalculation."""

from __future__ import annotations

from datetime import UTC, datetime

from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.assistant import Clarification, Transport
from engine.reason.context import get_forecast_snapshot
from engine.reason.facts import require_client, required_text
from engine.reason.questions import _parse_answer, confirm_in_transaction
from engine.reason.review import _review_config, finish_review, load_review, open_review_job
from engine.reason.scenarios import calculate_assumption
from engine.reason.schemas import (
    AssumptionInput,
    ContextError,
    PlannerAssumption,
    ReviewManifest,
    content_digest,
)
from engine.store.duckdb_store import EngineStore


def _find_question(manifest, question_id):
    question = next((item for item in manifest.questions if item.question_id == question_id), None)
    if question is None:
        raise ContextError("Question is outside this review")
    return question


def _successor_id(manifest: ReviewManifest) -> str:
    return "review_" + content_digest(
        [manifest.client_id, manifest.review_id, "operator_successor"]
    )


def _successor(manifest: ReviewManifest, now: datetime) -> ReviewManifest:
    return manifest.model_copy(
        update={
            "review_id": _successor_id(manifest),
            "successor_of": manifest.review_id,
            "revision": 1,
            "started_at": now,
            "state": "draft",
            "finalized_at": None,
            "seal": None,
            "operator_action_digest": None,
        }
    )


def _amend(con, profile, manifest, snapshot, question_id, action, actor, value, assumption):
    question = _find_question(manifest, question_id)
    now = datetime.now(UTC)
    facts, assumptions = manifest.appended_fact_version_ids, manifest.assumptions
    updates = {"actor": actor, "resolved_at": now}
    if action == "cancel":
        if question.status == "confirmed":
            raise ContextError("Edit the confirmed answer to revise its scope or value")
        updates["status"] = "cancelled"
    else:
        if action == "confirm" and question.status != "pending":
            raise ContextError("Use edit to replace an existing resolution")
        if action == "edit" and question.status == "pending":
            raise ContextError("Use confirm for a pending question")
        updates["status"] = "confirmed"
        if question.purpose == "scenario_assumption":
            if assumption is None or value is not None or question.scope is None:
                raise ContextError("An assumption requires structured low/high values and scope")
            scope = question.scope
            saved_group = next(
                (item.group for item in manifest.groups if item.group.group_id == scope.group_id),
                None,
            )
            if manifest.groups and (
                saved_group is None
                or saved_group.event_ids != scope.event_ids
                or not set(scope.series_ids) <= {member.series_id for member in saved_group.members}
            ):
                raise ContextError("The assumption question's saved membership has changed")
            if (
                assumption.unit != scope.unit
                or len(assumption.series_ids) != len(scope.series_ids)
                or set(assumption.series_ids) != set(scope.series_ids)
            ):
                raise ContextError("Confirm the question's exact unit and every included series")
            previous = next(
                (item for item in reversed(assumptions) if item.question_id == question_id), None
            )
            revision = previous.revision + 1 if previous else 1
            identifier = "assumption_" + content_digest(
                [
                    manifest.review_id,
                    question_id,
                    revision,
                    assumption.model_dump(mode="json"),
                    actor,
                ]
            )
            record = PlannerAssumption(
                assumption_id=identifier,
                question_id=question_id,
                revision=revision,
                supersedes_id=previous.assumption_id if previous else None,
                scope=scope,
                values=assumption,
                actor=actor,
                confirmed_at=now,
                condition_ids=(f"{identifier}:low", f"{identifier}:high"),
            )
            # The scenario is validated against every member before an answer is written.
            baselines = dict(question.baselines)
            matched = [
                row
                for row in snapshot.rows
                if row.period == scope.period and row.series_id in baselines
            ]
            if {row.series_id for row in matched} != set(scope.series_ids):
                raise ContextError("Assumption is outside the saved forecast")
            for row in matched:
                calculate_assumption(record, row.model_copy(update={"unit": scope.unit}))
            assumptions = (*assumptions, record)
        else:
            if value is None or assumption is not None:
                raise ContextError("Provide a single typed value for this question")
            phase_one = records.get_question(con, manifest.client_id, question_id)
            if phase_one is None:
                raise ContextError("The underlying company question is missing")
            if question.purpose == "mapping_confirmation" and not any(
                row.series_id == value.strip() and row.period in question.months
                for row in snapshot.rows
            ):
                raise ContextError("Mapping must name an existing forecast series for the month")
            previous = phase_one.answer
            answer = confirm_in_transaction(
                con,
                profile,
                question_id,
                value,
                actor,
                supersedes_answer_id=previous.answer_id if previous else None,
            )
            facts = tuple(dict.fromkeys((*facts, answer.fact_version_id)))
            updates["fact_version_id"] = answer.fact_version_id
            now = max(now, answer.confirmed_at)
            updates["resolved_at"] = now
    questions = tuple(
        item.model_copy(update=updates) if item.question_id == question_id else item
        for item in manifest.questions
    )
    return dict(
        questions=questions,
        assumptions=assumptions,
        appended_fact_version_ids=facts,
        knowledge_cutoff=now,
        resume_mode=(
            "operator_offline"
            if question.purpose == "scenario_assumption" or action == "cancel"
            else "operator_reassess"
        ),
    )


def resolve_review_question(
    profile: ClientProfile,
    store: EngineStore,
    review_id: str,
    question_id: str,
    action: str,
    actor: str,
    *,
    value: str | None = None,
    assumption: AssumptionInput | None = None,
    transport: Transport | None = None,
) -> ReviewManifest:
    require_client(profile, store)
    parent = load_review(store, review_id)
    try:
        actor = required_text(actor, "actor")
    except ValueError as exc:
        raise ContextError(str(exc)) from exc
    if action not in {"confirm", "edit", "cancel"}:
        raise ContextError("Unknown review question action")
    if action == "cancel" and (value is not None or assumption is not None):
        raise ContextError("Cancel takes no answer")
    question = _find_question(parent, question_id)
    action_digest = content_digest(
        [
            question_id,
            action,
            actor,
            value.strip() if value is not None else None,
            assumption.model_dump(mode="json") if assumption else None,
        ]
    )
    offline = question.purpose == "scenario_assumption" or action == "cancel"
    snapshot = get_forecast_snapshot(store, parent.snapshot_id)
    if parent.config_digest != _review_config(profile):
        raise ContextError("Review settings have changed. Restore the pinned settings to amend it.")
    if parent.state == "finalized":
        if parent.operator_action_digest == action_digest:
            return parent
        draft = _successor(parent, datetime.now(UTC))
        with records.write_transaction(store) as con:
            existing = records.read_review(con, store.client_id, draft.review_id)
            if existing is None:
                updates = _amend(
                    con, profile, draft, snapshot, question_id, action, actor, value, assumption
                )
                records.insert_review(
                    con,
                    draft.model_copy(
                        update={
                            **updates,
                            "operator_action_digest": action_digest,
                        }
                    ),
                )
            elif existing.operator_action_digest != action_digest:
                raise ContextError(
                    f"A successor already exists. Apply changes to {existing.review_id}."
                )
            elif existing.state == "finalized":
                return existing
        with open_review_job(profile, store, draft.review_id, transport=transport) as job:
            return finish_review(job, offline=offline, retriage=False)
    with open_review_job(profile, store, review_id, transport=transport) as job:
        if job.manifest.operator_action_digest != action_digest:
            with job.write() as con:
                updates = _amend(
                    con,
                    profile,
                    job.manifest,
                    snapshot,
                    question_id,
                    action,
                    actor,
                    value,
                    assumption,
                )
                job.advance(con, **updates, operator_action_digest=action_digest)
        return finish_review(job, offline=offline, retriage=False)


def propose_clarification(
    profile: ClientProfile,
    store: EngineStore,
    review_id: str,
    question_id: str,
    text: str,
    *,
    transport: Transport | None = None,
) -> tuple[str, str]:
    """Parse operator prose into a typed proposal. Confirmation is a distinct action."""
    require_client(profile, store)
    manifest = load_review(store, review_id)
    question = _find_question(manifest, question_id)
    if question.purpose == "scenario_assumption":
        raise ContextError("Use structured low/high entry for scenario assumptions")
    if manifest.config_digest != _review_config(profile):
        raise ContextError("Review settings have changed. Restore the pinned settings to amend it.")
    with records.write_transaction(store) as con:
        phase_one = records.get_question(con, store.client_id, question_id)
        if phase_one is None:
            raise ContextError("The underlying company question is missing")
        if manifest.state == "finalized":
            draft = _successor(manifest, datetime.now(UTC))
            existing = records.read_review(con, store.client_id, draft.review_id)
            if existing is None:
                records.insert_review(con, draft)
            elif existing.state != "draft":
                raise ContextError(f"Apply changes to successor {existing.review_id}.")
            review_id = draft.review_id

    def validate(proposal):
        try:
            _parse_answer(phase_one.spec, proposal.value)
        except ValueError as exc:
            raise ContextError("clarification_type_mismatch") from exc
        return proposal

    with open_review_job(profile, store, review_id, transport=transport) as job:
        value = job.invoke(
            "clarification",
            question_id,
            {
                "question": question.message,
                "type": phase_one.spec.answer_type,
                "object_type": phase_one.spec.object_type,
                "operator_text": text,
            },
            Clarification,
            validate=validate,
            allow_lookup=False,
        ).value
    return review_id, value
