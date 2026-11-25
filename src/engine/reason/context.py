"""Pin stored monthly forecasts and the context observed during a review."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import UTC, date, datetime
from math import isfinite
from zoneinfo import ZoneInfo

from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.config import ReasoningConfig, TargetConfig
from engine.reason.erp import require_reasoning
from engine.reason.facts import find_conflicts, local_midnight, overlaps, require_client
from engine.reason.questions import record_question
from engine.reason.schemas import (
    CompanyFact,
    ContextError,
    ContextGap,
    EntityRef,
    ForecastRow,
    ForecastSnapshot,
    QuestionSpec,
    content_digest,
)
from engine.store.duckdb_store import EngineStore


def _next_month(period: date) -> date:
    year, month = divmod(period.year * 12 + period.month, 12)
    return date(year, month + 1, 1)


def _month_offset(cutoff: date, period: date) -> int:
    return (period.year - cutoff.year) * 12 + period.month - cutoff.month


def _run_config(header: dict) -> dict:
    try:
        config = {} if header["config"] is None else json.loads(header["config"])
    except (TypeError, ValueError) as exc:
        raise ContextError("Stored forecast run configuration is invalid") from exc
    if not isinstance(config, dict):
        raise ContextError("Stored forecast run configuration must be an object")
    if "horizon" in config and (type(config["horizon"]) is not int or config["horizon"] < 1):
        raise ContextError("Stored forecast run declares an invalid horizon")
    return config


def _validate_saved_rows(raw: list[dict], declared_horizon: int | None) -> None:
    if not raw:
        raise ContextError(
            "The forecast run has no stored point rows. Select a completed forecast."
        )
    cutoff = raw[0]["cutoff_period"]
    if cutoff.day != 1:
        raise ContextError("Monthly reasoning requires a month-start history cutoff")
    seen = defaultdict(set)
    for row in raw:
        series = row["series_id"]
        horizon = row["horizon"]
        period = row["period"]
        if cutoff != row["cutoff_period"]:
            raise ContextError("The forecast run has inconsistent history cutoffs")
        if not series.strip() or not row["model"].strip():
            raise ContextError("Stored point rows require a series ID and selected model")
        if period.day != 1 or horizon < 1 or _month_offset(cutoff, period) != horizon:
            raise ContextError(
                f"Invalid monthly target period or horizon for {series!r} on {period}"
            )
        if period == date(9999, 12, 1):
            raise ContextError(
                f"The end of monthly target period {period} is outside supported dates"
            )
        if horizon in seen[series]:
            raise ContextError(f"Duplicate point forecast for series {series!r} on {period}")
        seen[series].add(horizon)
        if row["forecast"] is None or not isfinite(row["forecast"]):
            raise ContextError(f"The point baseline for {series!r} on {period} must be finite")
    for series, horizons in seen.items():
        end = declared_horizon if declared_horizon is not None else max(horizons)
        if len(horizons) != end or max(horizons) != end:
            raise ContextError(
                f"Incomplete forecast horizons for series {series!r}; expected 1–{end}"
            )


def _build_rows(
    profile: ClientProfile,
    raw: list[dict],
    dimensions: dict[str, dict],
    zone: ZoneInfo,
    target: TargetConfig | None,
) -> tuple[list[ForecastRow], list[ContextGap]]:
    rows = []
    gaps = []
    for entry in raw:
        series = entry["series_id"]
        period = entry["period"]
        dim = dimensions.get(series, {})
        unit = None
        if target:
            unit = target.units_by_series.get(series) or dim.get(target.unit_field)
        if not dim:
            gaps.append(
                ContextGap(
                    code="missing_dimension",
                    series_id=series,
                    period=period,
                    message="No current dim_series row matches this series.",
                )
            )
        if not unit or not unit.strip():
            unit = None
        rows.append(
            ForecastRow(
                client_id=profile.client_id,
                forecast_run_id=entry["run_id"],
                series_id=series,
                cutoff_period=entry["cutoff_period"],
                period=period,
                horizon=entry["horizon"],
                model=entry["model"],
                baseline=entry["forecast"],
                item_id=dim.get("item_code") or None,
                customer_id=dim.get("customer") or None,
                unit=unit,
                target_start=local_midnight(period, zone),
                target_end=local_midnight(_next_month(period), zone),
            )
        )
    return rows, gaps


def _row_context(
    row: ForecastRow,
    config: ReasoningConfig,
    by_subject: dict[EntityRef, list[CompanyFact]],
) -> tuple[ForecastRow, list[ContextGap], list[QuestionSpec]]:
    series = EntityRef(type="series", id=row.series_id)
    pending = deque([EntityRef(type="company", id=row.client_id), series])
    if row.item_id:
        pending.append(EntityRef(type="item", id=row.item_id))
    if row.customer_id:
        pending.append(EntityRef(type="customer", id=row.customer_id))
    subjects = set()
    relevant = []
    superseded = set()
    gaps = []
    questions = []
    unit = row.unit
    while pending:
        subject = pending.popleft()
        if subject in subjects:
            continue
        subjects.add(subject)
        current = [
            fact
            for fact in by_subject.get(subject, ())
            if overlaps(fact, row.target_start, row.target_end)
        ]
        relevant.extend(current)
        if subject == series and unit is None:
            unit = next(
                (
                    fact.value
                    for fact in current
                    if fact.relation == "unit"
                    and fact.status == "confirmed"
                    and isinstance(fact.value, str)
                ),
                None,
            )
        for source in config.sources:
            if (
                source.dataset == "relationship"
                and source.required_for_forecast
                and source.subject_type == subject.type
                and not any(
                    fact.relation == source.relation_type
                    and fact.location_role == source.location_role
                    and fact.object is not None
                    and fact.object.type == source.object_type
                    for fact in current
                )
            ):
                message = (
                    f"Missing {source.relation_type} relationship "
                    f"for {subject.type} {subject.id!r}."
                )
                gaps.append(
                    ContextGap(
                        code="missing_relationship",
                        message=message,
                        series_id=row.series_id,
                        period=row.period,
                    )
                )
                questions.append(
                    QuestionSpec(
                        subject=subject,
                        field=source.relation_type,
                        location_role=source.location_role,
                        purpose="missing_fact",
                        message=message,
                        answer_type="object_id",
                        object_type=source.object_type,
                        cardinality=source.cardinality,
                        affected_months=(row.period,),
                    )
                )
        overridden = {
            version
            for fact in current
            if fact.effective_from <= row.target_start
            and (fact.effective_to is None or fact.effective_to >= row.target_end)
            for version in fact.resolves_fact_ids
        }
        superseded.update(
            fact.fact_version_id for fact in current if fact.fact_version_id in overridden
        )
        pending.extend(
            fact.object
            for fact in current
            if fact.object and fact.fact_version_id not in overridden
        )

    if unit is None:
        message = "No unit was found in the series mapping, metadata, or confirmed facts."
        gaps.append(
            ContextGap(
                code="missing_unit", series_id=row.series_id, period=row.period, message=message
            )
        )
        questions.append(
            QuestionSpec(
                subject=series,
                field="unit",
                purpose="missing_fact",
                message=message,
                answer_type="text",
                affected_months=(row.period,),
            )
        )
    by_version = {fact.fact_version_id: fact for fact in relevant}
    for conflict in find_conflicts(
        tuple(relevant), valid_from=row.target_start, valid_to=row.target_end
    ):
        example = by_version[conflict.fact_version_ids[0]]
        message = (
            f"Conflicting {conflict.relation} values "
            f"for {conflict.subject.type} {conflict.subject.id!r}."
        )
        gaps.append(
            ContextGap(
                code="conflicting_fact",
                message=message,
                series_id=row.series_id,
                period=row.period,
                fact_version_ids=conflict.fact_version_ids,
            )
        )
        answer_type = "text"
        if example.object:
            answer_type = "object_id"
        elif isinstance(example.value, bool):
            answer_type = "boolean"
        elif isinstance(example.value, float):
            answer_type = "number"
        questions.append(
            QuestionSpec(
                subject=conflict.subject,
                field=conflict.relation,
                location_role=conflict.location_role,
                purpose="fact_conflict",
                message=message,
                answer_type=answer_type,
                object_type=example.object.type if example.object else None,
                affected_months=(row.period,),
                evidence_ids=conflict.fact_version_ids,
                candidate_fact_ids=conflict.fact_version_ids,
            )
        )
    return (
        row.model_copy(
            update={
                "unit": unit,
                "fact_version_ids": tuple(sorted(by_version)),
                "superseded_fact_version_ids": tuple(sorted(superseded)),
            }
        ),
        gaps,
        questions,
    )


def load_forecast_snapshot(
    profile: ClientProfile, store: EngineStore, forecast_run_id: int
) -> ForecastSnapshot:
    """Freeze a particular saved run and its current metadata, or reuse the same inputs."""
    require_client(profile, store)
    config = require_reasoning(profile)
    if profile.calendar.granularity != "monthly":
        raise ContextError(
            "Reasoning requires a monthly client forecast. Provide a monthly profile/run."
        )
    if type(forecast_run_id) is not int or forecast_run_id < 1:
        raise ContextError("Supply a positive saved forecast run ID")
    with records.write_transaction(store) as con:
        header, raw, dimensions = records.read_saved_run(con, profile.client_id, forecast_run_id)
        run_config = _run_config(header)
        if run_config.get("granularity", "monthly") != "monthly":
            raise ContextError("Reasoning requires a saved monthly forecast run")
        _validate_saved_rows(raw, run_config.get("horizon"))
        captured_at = datetime.now(UTC)
        zone = ZoneInfo(config.timezone)
        periods = sorted({row["period"] for row in raw})
        start = local_midnight(periods[0], zone)
        end = local_midnight(_next_month(periods[-1]), zone)
        con.execute(records.REASON_DDL)
        facts = records.eligible_facts(con, profile.client_id, captured_at, start, end)
        by_subject = defaultdict(list)
        for fact in facts:
            by_subject[fact.subject].append(fact)
        company = EntityRef(type="company", id=profile.client_id)
        target = config.target or next(
            (
                fact.value
                for fact in by_subject[company]
                if fact.relation == "target"
                and fact.status == "confirmed"
                and isinstance(fact.value, TargetConfig)
                and fact.effective_from <= start
                and (fact.effective_to is None or fact.effective_to >= end)
            ),
            None,
        )
        rows, gaps = _build_rows(profile, raw, dimensions, zone, target)
        questions = []
        for index, row in enumerate(rows):
            rows[index], row_gaps, row_questions = _row_context(row, config, by_subject)
            gaps.extend(row_gaps)
            questions.extend(row_questions)
        gaps.append(
            ContextGap(
                code="catalog_unverified",
                message="The run does not prove which series were expected to be forecast.",
            )
        )
        if not header.get("forecast_available_at"):
            gaps.append(
                ContextGap(
                    code="legacy_provenance",
                    message=(
                        "The forecast creation timestamp has an unknown timezone. Series metadata "
                        "was observed at snapshot creation and may differ from the original."
                    ),
                )
            )
        if "horizon" not in run_config:
            gaps.append(
                ContextGap(
                    code="horizon_unverified",
                    message="The run did not declare an expected horizon.",
                )
            )
        if target is None:
            message = "Configure or confirm the business target and outcome reference."
            gaps.append(ContextGap(code="missing_target", message=message))
            questions.append(
                QuestionSpec(
                    subject=company,
                    field="target",
                    purpose="missing_fact",
                    message=message,
                    answer_type="target",
                    affected_months=tuple(periods),
                )
            )
        config_digest = content_digest(
            {
                "reasoning": config.model_dump(
                    mode="json", exclude={"evidence", "assistant", "evaluation"}
                ),
                "calendar": profile.calendar.model_dump(mode="json"),
            }
        )
        metadata_digest = content_digest(dimensions)
        input_digest = content_digest(
            {
                "client": profile.client_id,
                "run": header,
                "metadata": metadata_digest,
                "config": config_digest,
                "rows": [row.model_dump(mode="json") for row in rows],
                "target": target.model_dump(mode="json") if target else None,
            }
        )
        snapshot_id = f"snapshot_{input_digest}"
        existing = records.read_snapshot(con, profile.client_id, snapshot_id)
        if existing:
            return existing
        question_ids = {
            record_question(con, profile.client_id, spec, captured_at).question_id
            for spec in questions
        }
        snapshot = ForecastSnapshot(
            snapshot_id=snapshot_id,
            client_id=profile.client_id,
            forecast_run_id=forecast_run_id,
            input_digest=input_digest,
            config_digest=config_digest,
            metadata_digest=metadata_digest,
            captured_at=captured_at,
            metadata_observed_at=captured_at,
            forecast_created_at_local=header["created_at"],
            forecast_available_at=header.get("forecast_available_at"),
            timezone=config.timezone,
            target=target,
            rows=tuple(rows),
            gaps=tuple(gaps),
            question_ids=tuple(sorted(question_ids)),
        )
        records.save_snapshot(con, snapshot)
        return snapshot


def get_forecast_snapshot(store: EngineStore, snapshot_id: str) -> ForecastSnapshot:
    """Resolve an immutable snapshot in the given client store."""
    if store.path.exists():
        with store.connect() as con:
            if records.table_exists(con, "reason_forecast_snapshots"):
                snapshot = records.read_snapshot(con, store.client_id, snapshot_id)
                if snapshot:
                    return snapshot
    raise ContextError(f"Snapshot {snapshot_id!r} was not found for {store.client_id!r}")
