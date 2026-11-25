"""Offline verification and deterministic replay of a finalized review."""

from __future__ import annotations

from math import isclose

from engine.config.base import StrictModel
from engine.reason import store as records
from engine.reason.integrity import manifest_digest, record_digests
from engine.reason.report import ReviewReport, load_report
from engine.reason.review import load_review, materialize_assessments
from engine.reason.review_inputs import group_packet, pinned_inputs
from engine.reason.scenarios import METHOD_VERSION, validate_proposal
from engine.reason.schemas import Assessment, ContextError, ReviewManifest, content_digest
from engine.store.duckdb_store import EngineStore


class ReplayResult(StrictModel):
    schema_version: str = "forecast-review-replay/1"
    manifest: ReviewManifest
    reproduced: bool
    seal_verified: bool = False
    issues: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    stored_assessments: tuple[Assessment, ...] = ()
    replayed_assessments: tuple[Assessment, ...] = ()


def _key(row: Assessment) -> tuple:
    return row.scenario_id, row.series_id, row.period, row.target


def _difference(a, b, path="") -> str | None:
    if isinstance(a, float) and isinstance(b, float):
        return None if isclose(a, b, rel_tol=1e-9, abs_tol=1e-8) else path
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return next(
            (diff for key in a if (diff := _difference(a[key], b[key], f"{path}.{key}"))), None
        )
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return next(
            (
                diff
                for index, (left, right) in enumerate(zip(a, b, strict=True))
                if (diff := _difference(left, right, f"{path}[{index}]"))
            ),
            None,
        )
    return None if a == b else path or "value"


def _seal_errors(report: ReviewReport) -> tuple[str, ...]:
    seal = report.manifest.seal
    if seal is None:
        if report.manifest.approved_lookups is not None:
            return (f"Missing integrity seal for {report.manifest.review_id}",)
        return ()
    errors = []
    if seal.manifest_digest != manifest_digest(report.manifest):
        errors.append(f"Manifest checksum mismatch for {report.manifest.review_id}")
    actual = record_digests(
        report.forecast,
        report.base_batch,
        report.evidence,
        report.facts,
        report.source_checks,
        report.assessments,
    )
    errors.extend(
        f"Checksum or membership mismatch for {key}"
        for key in sorted(seal.records.keys() | actual.keys())
        if seal.records.get(key) != actual.get(key)
    )
    return tuple(errors)


def _check_links(report: ReviewReport, inputs) -> None:
    m, snapshot, batch = report.manifest, report.forecast, report.base_batch
    cutoff = m.knowledge_cutoff
    if m.state != "finalized" or m.finalized_at != cutoff:
        raise ContextError(f"Review {m.review_id} has no finalized knowledge cutoff")
    if (
        snapshot.snapshot_id != m.snapshot_id
        or batch.batch_id != m.base_batch_id
        or batch.snapshot_id != m.snapshot_id
        or snapshot.input_digest != m.forecast_digest
        or batch.config_digest != m.collection_config_digest
        or batch.knowledge_cutoff != m.collection_cutoff
        or batch.snapshot_captured_at != snapshot.captured_at
        or snapshot.captured_at > m.collection_cutoff
        or m.collection_cutoff > cutoff
        or (
            snapshot.forecast_available_at and snapshot.forecast_available_at > snapshot.captured_at
        )
        or batch.forecast_run_id != m.forecast_run_id
        or snapshot.forecast_run_id != m.forecast_run_id
        or snapshot.granularity != "monthly"
    ):
        raise ContextError(f"Review {m.review_id} has inconsistent snapshot or batch bindings")
    if any(
        client != m.client_id
        for client in (
            snapshot.client_id,
            batch.client_id,
            *(row.client_id for row in snapshot.rows),
            *(item.client_id for item in report.evidence),
            *(item.client_id for item in report.facts),
            *(check.scope.client_id for check in report.source_checks),
        )
    ):
        raise ContextError(f"Review {m.review_id} contains a foreign-client revision")
    for row in snapshot.rows:
        if row.period.day != 1 or row.forecast_run_id != m.forecast_run_id:
            raise ContextError(f"Invalid forecast row {row.series_id}/{row.period}")
        for key in row.fact_version_ids:
            fact = inputs.facts[key]
            if fact.known_at > snapshot.captured_at or fact.captured_at > snapshot.captured_at:
                raise ContextError(f"Fact {key} was unavailable at snapshot {snapshot.snapshot_id}")
    for revision in report.evidence:
        if revision.evidence_id in batch.evidence_ids and (
            revision.first_known_at > batch.knowledge_cutoff
            or revision.captured_at > batch.knowledge_cutoff
        ):
            raise ContextError(
                f"Evidence {revision.evidence_id} was unavailable at base collection"
            )
    checks = {item.check_id: item for item in report.source_checks}
    for key in m.appended_check_ids:
        check = checks[key]
        if check.started_at > check.ended_at or check.ended_at > cutoff:
            raise ContextError(f"Source check {key} falls outside the review cutoff")
    for attempt in m.lookup_attempts:
        if attempt.started_at > cutoff or attempt.check_id not in checks:
            raise ContextError(f"Lookup attempt {attempt.attempt_id} has no eligible check")
    if len(m.lookup_attempts) > m.followup_limit:
        raise ContextError(f"Review {m.review_id} exceeded its follow-up attempt limit")
    if len(m.model_calls) > m.model_request_limit or any(
        call.started_at > cutoff or call.ended_at is None or call.ended_at > cutoff
        for call in m.model_calls
    ):
        raise ContextError(f"Review {m.review_id} has an invalid model request ledger")
    for assumption in m.assumptions:
        if assumption.confirmed_at > cutoff:
            raise ContextError(f"Assumption {assumption.assumption_id} was confirmed after cutoff")
    for decision in m.mappings:
        missing = (
            set(decision.article_ids + decision.evidence_ids) - inputs.evidence.keys()
            | set(decision.fact_version_ids) - inputs.facts.keys()
        )
        if missing:
            raise ContextError(
                f"Mapping {decision.decision_id} references {', '.join(sorted(missing))}"
            )
    rows = {(row.series_id, row.period): row for row in inputs.snapshot.rows}
    if len(rows) != len(snapshot.rows):
        raise ContextError(f"Snapshot {snapshot.snapshot_id} repeats a row")
    groups = set()
    for result in m.groups:
        group = result.group
        if group.group_id in groups:
            raise ContextError(f"Duplicate group {group.group_id}")
        groups.add(group.group_id)
        missing = (
            set(group.evidence_ids) - inputs.evidence.keys()
            | set(group.fact_version_ids) - inputs.facts.keys()
        )
        if missing:
            raise ContextError(f"Group {group.group_id} references {', '.join(sorted(missing))}")
        for member in (*group.members, *group.exceptions):
            row = rows.get((member.series_id, member.period))
            if (
                row is None
                or row.period != group.period
                or (row.item_id, row.customer_id) != (member.item_id, member.customer_id)
            ):
                raise ContextError(f"Group {group.group_id} has invalid member {member.series_id}")
        packet, evidence, facts, checks = group_packet(inputs, group)
        if m.followup_limit and m.approved_lookups is not None:
            packet["configured_lookups"] = m.approved_lookups
        if (not m.followup_limit or m.approved_lookups is not None) and (
            content_digest(packet) != result.input_digest
        ):
            raise ContextError(f"Group input digest mismatch for {group.group_id}")
        if result.proposal:
            try:
                validate_proposal(result.proposal, group, evidence, facts, checks)
            except ContextError as exc:
                raise ContextError(f"Group {group.group_id}: {exc}") from exc


def report_inputs(report: ReviewReport):
    return pinned_inputs(
        report.forecast,
        report.manifest,
        report.evidence,
        {item.fact_version_id: item for item in report.facts},
        report.source_checks,
    )


def verify_report(report: ReviewReport) -> ReplayResult:
    manifest = report.manifest
    common = dict(manifest=manifest, stored_assessments=report.assessments)
    limitations = []
    if manifest.seal is None:
        limitations.append("legacy_manifest_has_no_integrity_seal")
    if manifest.followup_limit and manifest.approved_lookups is None:
        limitations.append("legacy_group_packets_have_no_saved_tool_catalogue")
    if report.forecast.forecast_available_at is None:
        limitations.append("legacy_forecast_availability_bounded_by_snapshot_capture")
    limitations.extend(
        f"seasonal_issue_time_unknown:{item.evidence_id}"
        for item in report.evidence
        if item.weather and item.issued_at is None
    )
    limitations.extend(f"archive_or_source_gap:{gap}" for gap in manifest.gaps)
    issues = _seal_errors(report)
    verified = manifest.seal is not None and not issues
    if issues:
        return ReplayResult(
            **common, reproduced=False, issues=issues, limitations=tuple(limitations)
        )
    try:
        if manifest.method_version != METHOD_VERSION:
            raise ContextError(f"Unsupported review method version {manifest.method_version}")
        for row in report.assessments:
            if row.method_version is not None and row.method_version != METHOD_VERSION:
                raise ContextError(f"Unsupported method {row.method_version} for {row.scenario_id}")
        inputs = report_inputs(report)
        _check_links(report, inputs)
        replayed, _ = materialize_assessments(
            manifest,
            inputs,
            manifest.groups,
            manifest.revision,
        )
    except (ContextError, KeyError) as exc:
        return ReplayResult(
            **common,
            reproduced=False,
            seal_verified=verified,
            issues=(str(exc),),
            limitations=tuple(limitations),
        )
    saved = {_key(item): item for item in report.assessments}
    replay = {_key(item): item for item in replayed}
    differences = []
    for key in sorted(saved.keys() | replay.keys()):
        label = "/".join(map(str, key))
        if key not in saved or key not in replay:
            differences.append(f"Assessment membership mismatch for {label}")
        elif diff := _difference(
            saved[key].model_dump(mode="json"), replay[key].model_dump(mode="json")
        ):
            differences.append(f"Assessment mismatch {label}{diff}")
    return ReplayResult(
        **common,
        reproduced=not differences,
        seal_verified=verified,
        issues=tuple(differences),
        limitations=tuple(limitations),
        replayed_assessments=tuple(
            replay[key] for key in map(_key, report.assessments) if key in replay
        )
        + tuple(replay[key] for key in sorted(replay.keys() - saved.keys())),
    )


def replay_review(store: EngineStore, review_id: str) -> ReplayResult:
    """Reproduce an exact finalized decision using saved data and no current deployment profile."""
    manifest = load_review(store, review_id)
    if manifest.state != "finalized":
        return ReplayResult(
            manifest=manifest,
            reproduced=False,
            issues=(f"Review {review_id} is a draft. Resume it before replay.",),
        )
    try:
        report = load_report(store, review_id)
    except ContextError as exc:
        with store.connect() as con:
            original = records.read_assessments(con, store.client_id, review_id)
        return ReplayResult(
            manifest=manifest, reproduced=False, issues=(str(exc),), stored_assessments=original
        )
    return verify_report(report)
