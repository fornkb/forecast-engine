"""Evaluate frozen advisory decisions on shared, date-eligible cohorts."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import polars as pl
from pydantic import AwareDatetime

from engine.config.base import StrictModel
from engine.config.schema import ClientProfile
from engine.evaluate.metrics import aggregate
from engine.reason.facts import require_client
from engine.reason.outcomes import GroupLabel, Outcome, OutcomeIndex, TriageLabel, read_csv
from engine.reason.replay import report_inputs, verify_report
from engine.reason.report import load_report
from engine.reason.review_inputs import latest_checks, order_groups, pinned_inputs
from engine.reason.scenarios import order_operand
from engine.reason.schemas import CalculationOperand, ContextError, ForecastRow, content_digest
from engine.reason.usage import ReviewUsage, UsageTotal, provider_usage, review_usage
from engine.store.duckdb_store import EngineStore

RULE_VERSION = "confirmed-order-increment/1"
Phase = Literal["pre_period", "in_period", "post_period", "after_actual"]
_COMPARISONS = {
    "baseline_vs_final_rule": ("baseline", "rule_final"),
    "baseline_vs_base_rule": ("baseline", "rule_base"),
    "planner_point_paired": ("baseline", "rule_final", "planner"),
    "realized_condition_paired": ("baseline", "rule_final", "realized_condition"),
}


class RulePoint(StrictModel):
    point: float
    version: str = RULE_VERSION
    reason: str
    operand: CalculationOperand | None = None


class RangeOutcome(StrictModel):
    scenario_id: str
    group_id: str
    provenance: str
    condition_ids: tuple[str, ...]
    realized_condition_id: str | None = None
    conditional_point: float | None = None
    conditional_abs_error: float | None = None
    width: float
    inside_range: bool | None = None
    retrospective_label: bool | None = None
    reason: str | None = None


class EvaluatedRow(StrictModel):
    review_id: str
    client_id: str
    forecast_run_id: int
    decision_at: AwareDatetime
    forecast_available_at: AwareDatetime | None = None
    forecast_available_by: AwareDatetime
    phase: Phase
    series_id: str
    period: date
    horizon: int
    target: str
    unit: str | None
    baseline: float
    rule_base: RulePoint
    rule_final: RulePoint
    actual: float | None = None
    actual_revision: int | None = None
    actual_source_ref: str | None = None
    actual_observed_at: AwareDatetime | None = None
    first_actual_at: AwareDatetime | None = None
    planner_point: float | None = None
    planner_selected_at: AwareDatetime | None = None
    ranges: tuple[RangeOutcome, ...] = ()
    followup_evidence_ids: tuple[str, ...] = ()
    added_fact_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    exclusion: str | None = None
    planner_exclusion: str | None = None

    @property
    def key(self):
        return self.client_id, self.series_id, self.period, self.target, self.unit, self.phase


class PointError(StrictModel):
    comparator: str
    n: int
    mae: float | None = None
    wape: float | None = None


class PointComparison(StrictModel):
    comparison: str
    provenance: str | None = None
    client_id: str
    phase: str
    unit: str | None
    target: str
    review_id: str | None = None
    review_date: date | None = None
    horizon: int | None = None
    candidate_rows: int
    n: int
    actual_sum: float
    exclusions: dict[str, int]
    errors: tuple[PointError, ...]


class RangeSummary(StrictModel):
    client_id: str
    phase: str
    unit: str | None
    target: str
    provenance: str
    ranges: int
    resolved: int
    unresolved: int
    condition_counts: dict[str, int]
    condition_counts_by_scenario: dict[str, dict[str, int]]
    width_mean: float
    conditional_mae: float | None = None
    observed_actuals: int
    inside_range: int


class LabelSummary(StrictModel):
    independent_group_labels: int
    group_results: dict[str, int]
    independent_articles: int
    article_results: dict[str, int]
    expected_associations: int
    matched_associations: int
    missing_associations: int
    incorrect_associations: int
    incorrect_product_links: int
    missing_product_links: int


class EvaluationResult(StrictModel):
    schema_version: str = "forecast-review-evaluation/1"
    client_id: str
    as_of: AwareDatetime
    review_ids: tuple[str, ...]
    outcome_input_digest: str
    group_label_digest: str | None = None
    triage_label_digest: str | None = None
    rows: tuple[EvaluatedRow, ...]
    aggregate_unique_rows: int
    repeated_rows_removed: int
    point_comparisons: tuple[PointComparison, ...]
    conditional_ranges: tuple[RangeSummary, ...]
    labels: LabelSummary
    reviews: tuple[ReviewUsage, ...]
    aggregate_usage: tuple[UsageTotal, ...]
    aggregate_elapsed_seconds: float
    unmatched_outcomes: int
    limitations: tuple[str, ...] = ()


class _Rule:
    def __init__(self, inputs):
        self.inputs = inputs
        self.records = defaultdict(dict)
        self.checks = latest_checks(inputs.checks)
        for group in order_groups(inputs.snapshot, inputs.evidence):
            for member in group.members:
                key = member.series_id, member.period
                for identifier in group.evidence_ids:
                    item = inputs.evidence[identifier]
                    if (
                        item.operational.dataset == "order_amendment"
                        and item.operational.status == "confirmed"
                    ):
                        self.records[key][identifier] = group.flags

    def apply(self, row: ForecastRow, target: str) -> RulePoint:
        records = self.records[row.series_id, row.period]
        if not records:
            return RulePoint(point=row.baseline, reason="no_direct_confirmed_increment")
        if len(records) != 1:
            return RulePoint(
                point=row.baseline, reason="multiple_or_conflicting_confirmed_increments"
            )
        identifier, flags = next(iter(records.items()))
        if flags:
            return RulePoint(point=row.baseline, reason="uncertain_order_scope")
        if not any(
            identifier in check.evidence_ids
            and row.period in check.scope.months
            and check.coverage == "complete"
            and all(
                month.status == "complete" for month in check.months if month.period == row.period
            )
            for check in self.checks
        ):
            return RulePoint(point=row.baseline, reason="increment_source_coverage_incomplete")
        try:
            operand = order_operand(
                "confirmed_increment", self.inputs.evidence[identifier], row, target
            )
        except ContextError as exc:
            return RulePoint(point=row.baseline, reason=str(exc))
        point = float(Decimal(str(row.baseline)) + Decimal(str(operand.value)))
        if not isfinite(point) or point < 0:
            return RulePoint(point=row.baseline, reason="invalid_rule_endpoint")
        return RulePoint(point=point, reason="direct_confirmed_increment", operand=operand)


def _phase(row, cutoff, observed) -> Phase:
    if observed is not None and cutoff >= observed:
        return "after_actual"
    if cutoff < row.target_start:
        return "pre_period"
    if cutoff < row.target_end:
        return "in_period"
    return "post_period"


def _ranges(assessments, label, actual, scored, cutoff, as_of):
    results = []
    for item in assessments:
        if not item.conditions:
            continue
        bounds = item.conditions
        matched = None
        reason = "condition_unresolved"
        if label and label.scenario_id == item.scenario_id:
            if label.condition_observed_at is None or label.condition_observed_at <= as_of:
                matched = next(
                    (case for case in bounds if case.condition_id == label.condition_id), None
                )
                reason = None if matched else "unknown_condition_id"
            else:
                reason = "condition_label_not_yet_available"
        point = matched.endpoint if matched else None
        results.append(
            RangeOutcome(
                scenario_id=item.scenario_id,
                group_id=item.group_id,
                provenance=(
                    "planner_assumption"
                    if item.assumption_id
                    else "validated_driver"
                    if item.method_id == "historical_driver_response"
                    else "evidence_bound"
                ),
                condition_ids=tuple(case.condition_id for case in bounds),
                realized_condition_id=matched.condition_id if matched else None,
                conditional_point=point,
                conditional_abs_error=abs(actual - point) if scored and point is not None else None,
                width=bounds[-1].endpoint - bounds[0].endpoint,
                inside_range=(bounds[0].endpoint <= actual <= bounds[-1].endpoint)
                if scored
                else None,
                retrospective_label=(
                    label.condition_observed_at is None or label.condition_observed_at > cutoff
                )
                if matched
                else None,
                reason=reason,
            )
        )
    return tuple(results)


def _review_rows(report, outcomes, as_of):
    manifest = report.manifest
    final = report_inputs(report)
    base = report.base_batch
    base_manifest = manifest.model_copy(
        update={
            "knowledge_cutoff": base.knowledge_cutoff,
            "appended_fact_version_ids": (),
        }
    )
    base_set, checks = set(base.evidence_ids), set(base.check_ids)
    original = pinned_inputs(
        report.forecast,
        base_manifest,
        tuple(item for item in report.evidence if item.evidence_id in base_set),
        final.facts,
        tuple(check for check in final.checks if check.check_id in checks),
    )
    baseline_rule, final_rule = _Rule(original), _Rule(final)
    original_rows = {(row.series_id, row.period): row for row in original.snapshot.rows}
    assessments = defaultdict(list)
    for finding in report.assessments:
        if finding.target_type == "demand":
            assessments[finding.series_id, finding.period, finding.target].append(finding)
    results = []
    append_ids, added_facts = (
        set(manifest.appended_evidence_ids),
        set(manifest.appended_fact_version_ids),
    )
    for row in final.snapshot.rows:
        target = final.snapshot.target.id if final.snapshot.target else "unknown"
        key = row.series_id, row.period, target
        original_row = original_rows[row.series_id, row.period]
        first_actual = outcomes.first_seen.get(key)
        observed = first_actual if first_actual is not None and first_actual <= as_of else None
        actual = outcomes.actuals.get(key)
        cutoff = manifest.knowledge_cutoff
        phase = _phase(row, cutoff, observed)
        label = outcomes.decision(key, manifest.review_id)
        available_by = report.forecast.forecast_available_at or report.forecast.captured_at
        exclusion = None
        if phase in {"after_actual", "post_period"}:
            exclusion = (
                "issued_after_outcome" if phase == "after_actual" else "issued_after_target_month"
            )
        elif available_by > cutoff:
            exclusion = "baseline_unavailable_at_decision"
        elif target == "unknown" or row.unit is None:
            exclusion = "target_or_unit_unknown"
        elif actual is None:
            exclusion = (
                "outcome_not_yet_available"
                if key in outcomes.pending
                else "missing_matching_outcome"
            )
        elif actual.unit != row.unit:
            exclusion = "unit_mismatch"
        if label and label.planner_point is not None:
            if label.selected_at < cutoff:
                planner_exclusion = "selection_predates_final_review"
            elif label.selected_at > as_of:
                planner_exclusion = "selection_not_yet_available"
            elif _phase(row, label.selected_at, observed) != phase:
                planner_exclusion = "selection_in_different_decision_window"
            else:
                planner_exclusion = None
        else:
            planner_exclusion = "no_recorded_planner_point"
        predicted = final_rule.apply(row, target)
        old_target = original.snapshot.target.id if original.snapshot.target else None
        base_prediction = (
            baseline_rule.apply(original_row, target)
            if old_target == target and row.unit == original_row.unit
            else RulePoint(
                point=row.baseline,
                reason="target_or_unit_unverified_in_base",
            )
        )
        available_ranges = assessments[key]
        cited = {identifier for item in available_ranges for identifier in item.evidence_ids}
        if predicted.operand:
            cited.add(predicted.operand.record_version_id)
        results.append(
            EvaluatedRow(
                review_id=manifest.review_id,
                client_id=manifest.client_id,
                forecast_run_id=manifest.forecast_run_id,
                decision_at=cutoff,
                forecast_available_at=report.forecast.forecast_available_at,
                forecast_available_by=available_by,
                phase=phase,
                series_id=row.series_id,
                period=row.period,
                horizon=row.horizon,
                unit=row.unit,
                target=target,
                baseline=row.baseline,
                rule_base=base_prediction,
                rule_final=predicted,
                actual=actual.actual if actual else None,
                actual_revision=actual.revision if actual else None,
                actual_source_ref=actual.source_ref if actual else None,
                actual_observed_at=actual.observed_at if actual else None,
                first_actual_at=observed,
                planner_point=label.planner_point if label and planner_exclusion is None else None,
                planner_selected_at=label.selected_at if label else None,
                ranges=_ranges(
                    available_ranges,
                    label,
                    actual.actual if actual else None,
                    not exclusion,
                    cutoff,
                    as_of,
                ),
                followup_evidence_ids=tuple(sorted(cited & append_ids)),
                added_fact_ids=tuple(
                    sorted(
                        {key for item in available_ranges for key in item.fact_version_ids}
                        & added_facts
                    )
                ),
                assumption_ids=tuple(
                    sorted({item.assumption_id for item in available_ranges if item.assumption_id})
                ),
                exclusion=exclusion,
                planner_exclusion=planner_exclusion,
            )
        )
    return results


def _points(row, comparison, provenance):
    if row.exclusion:
        return (), row.exclusion
    if comparison == "baseline_vs_final_rule":
        return (row.baseline, row.rule_final.point), None
    if comparison == "baseline_vs_base_rule":
        if row.rule_base.reason == "target_or_unit_unverified_in_base":
            return (), row.rule_base.reason
        return (row.baseline, row.rule_base.point), None
    if comparison == "planner_point_paired":
        if row.planner_point is None:
            return (), row.planner_exclusion
        return (row.baseline, row.rule_final.point, row.planner_point), None
    realized = [item for item in row.ranges if item.conditional_point is not None]
    if len(realized) != 1:
        return (), "condition_unresolved_or_ambiguous"
    if provenance and realized[0].provenance != provenance:
        return (), "condition_from_different_basis"
    return (row.baseline, row.rule_final.point, realized[0].conditional_point), None


def _comparisons(rows, *, review_id=None, review_date=None, horizon=None):
    by_scope = defaultdict(list)
    for row in rows:
        by_scope[row.client_id, row.phase, row.unit or "", row.target].append(row)
    totals = []
    for (client, phase, unit, target), group in sorted(by_scope.items()):
        conditional = "realized_condition_paired"
        range_bases = sorted({value.provenance for row in group for value in row.ranges})
        variants = [
            (name, methods, None) for name, methods in _COMPARISONS.items() if name != conditional
        ]
        variants.extend(
            (conditional, _COMPARISONS[conditional], basis) for basis in range_bases or [None]
        )
        for comparison, names, provenance in variants:
            exclusions = Counter()
            predictions = []
            selected = []
            for row in group:
                values, reason = _points(row, comparison, provenance)
                if reason:
                    exclusions[reason] += 1
                    continue
                selected.append(row)
                predictions.extend(
                    dict(comparator=name, forecast=value, actual=row.actual)
                    for name, value in zip(names, values, strict=True)
                )
            scores = {}
            if predictions:
                scores = {
                    item["comparator"]: item
                    for item in aggregate(pl.DataFrame(predictions), ["comparator"]).to_dicts()
                }
            totals.append(
                PointComparison(
                    comparison=comparison,
                    provenance=provenance,
                    client_id=client,
                    phase=phase,
                    unit=unit or None,
                    target=target,
                    review_id=review_id,
                    review_date=review_date,
                    horizon=horizon,
                    candidate_rows=len(group),
                    n=len(selected),
                    actual_sum=sum(row.actual for row in selected),
                    exclusions=dict(sorted(exclusions.items())),
                    errors=tuple(
                        PointError(
                            comparator=name,
                            n=len(selected),
                            mae=scores.get(name, {}).get("mae"),
                            wape=scores.get(name, {}).get("wape"),
                        )
                        for name in names
                    ),
                )
            )
    return totals


def _range_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        for scenario in row.ranges:
            groups[
                row.client_id, row.phase, row.unit or "", row.target, scenario.provenance
            ].append(scenario)
    result = []
    for (client, phase, unit, target, provenance), values in sorted(groups.items()):
        counts = Counter(
            item.realized_condition_id for item in values if item.realized_condition_id
        )
        by_scenario = defaultdict(Counter)
        for item in values:
            if item.realized_condition_id:
                by_scenario[item.scenario_id][item.realized_condition_id] += 1
        errors = [
            item.conditional_abs_error for item in values if item.conditional_abs_error is not None
        ]
        result.append(
            RangeSummary(
                client_id=client,
                phase=phase,
                unit=unit or None,
                target=target,
                provenance=provenance,
                ranges=len(values),
                resolved=sum(counts.values()),
                unresolved=len(values) - sum(counts.values()),
                condition_counts=dict(sorted(counts.items())),
                condition_counts_by_scenario={
                    key: dict(sorted(value.items())) for key, value in sorted(by_scenario.items())
                },
                width_mean=sum(item.width for item in values) / len(values),
                conditional_mae=sum(errors) / len(errors) if errors else None,
                observed_actuals=sum(item.inside_range is not None for item in values),
                inside_range=sum(item.inside_range is True for item in values),
            )
        )
    return tuple(result)


def _labels(reports, group_labels, triage_labels):
    by_review = {report.manifest.review_id: report for report in reports}
    group_results, articles = Counter(), Counter()
    groups_seen, by_article = set(), defaultdict(list)
    for label in group_labels:
        key = label.review_id, label.group_id
        if key in groups_seen or label.review_id not in by_review:
            raise ContextError(f"Duplicate or unselected group label {key}")
        groups_seen.add(key)
        report = by_review[label.review_id]
        found = any(
            row.group_id == label.group_id and row.finding in {"possible_impact", "what_if_range"}
            for row in report.assessments
        )
        if label.label == "unresolved":
            group_results["unresolved"] += 1
        elif label.label == "affected":
            group_results["affected_found" if found else "affected_missed"] += 1
        else:
            group_results["unnecessary_alert" if found else "unaffected_quiet"] += 1
    for label in triage_labels:
        if label.review_id not in by_review:
            raise ContextError(f"Label refers to unselected review {label.review_id}")
        by_article[label.review_id, label.article_id].append(label)
    matched = expected = incorrect = missing_products = wrong_products = 0
    for (review_id, article), values in by_article.items():
        if len(set(values)) != len(values) or len({item.label for item in values}) != 1:
            raise ContextError(f"Duplicate or conflicting triage labels for {article}")
        manifest = by_review[review_id].manifest
        related = {article}
        for item in manifest.candidates:
            if article == item.evidence_id or article in item.duplicate_ids:
                related.add(item.evidence_id)
        mapping = next(
            (item for item in manifest.mappings if related & set(item.article_ids)), None
        )
        truth = values[0].label
        decision = mapping.status if mapping else "missing"
        articles[f"{truth}:{decision}"] += 1
        associations = {
            (item.entity_type, item.entity_id, item.period) for item in values if item.entity_id
        }
        if not associations:
            continue
        relevant = [
            item.group
            for item in manifest.groups
            if mapping and mapping.event_id in item.group.event_ids
        ]
        actual = {(item.dependency.type, item.dependency.id, item.period) for item in relevant}
        expected += len(associations)
        matched += len(actual & associations)
        incorrect += len(actual - associations)
        for association in associations:
            series = {
                item.series_id
                for item in values
                if (item.entity_type, item.entity_id, item.period) == association and item.series_id
            }
            if not series:
                continue
            observed = {
                member.series_id
                for group in relevant
                for member in group.members
                if (group.dependency.type, group.dependency.id, group.period) == association
            }
            missing_products += len(series - observed)
            wrong_products += len(observed - series)
    return LabelSummary(
        independent_group_labels=len(group_labels),
        group_results=dict(group_results),
        independent_articles=len(by_article),
        article_results=dict(articles),
        expected_associations=expected,
        matched_associations=matched,
        missing_associations=expected - matched,
        incorrect_associations=incorrect,
        incorrect_product_links=wrong_products,
        missing_product_links=missing_products,
    )


def evaluate_reviews(
    profile: ClientProfile,
    store: EngineStore,
    review_ids: tuple[str, ...],
    outcomes_path: Path,
    *,
    as_of: datetime | None = None,
    group_labels_path: Path | None = None,
    triage_labels_path: Path | None = None,
) -> EvaluationResult:
    require_client(profile, store)
    if not review_ids or len(set(review_ids)) != len(review_ids):
        raise ContextError("Supply one or more distinct finalized review IDs")
    now = datetime.now(UTC)
    if as_of is not None and (as_of.utcoffset() is None or as_of > now):
        raise ContextError(
            "Evaluation as-of must have a UTC offset and be at or before the current time"
        )
    as_of = (as_of or now).astimezone(UTC)
    reports = tuple(load_report(store, key) for key in review_ids)
    verified = tuple(verify_report(report) for report in reports)
    for item in verified:
        if not item.reproduced:
            raise ContextError(
                f"Review {item.manifest.review_id} cannot be evaluated: " + "; ".join(item.issues)
            )
        if item.manifest.knowledge_cutoff > as_of:
            raise ContextError(
                f"Review {item.manifest.review_id} was not available at evaluation as-of"
            )
    zones = {report.forecast.timezone for report in reports}
    if len(zones) != 1:
        raise ContextError("Select reviews on one pinned client calendar")
    zone = ZoneInfo(next(iter(zones)))
    columns = profile.reasoning.evaluation.outcome_columns if profile.reasoning else {}
    outcomes = read_csv(outcomes_path, Outcome, store.client_id, columns=columns, zone=zone)
    index = OutcomeIndex(outcomes, set(review_ids), as_of, zone)
    group_labels = (
        read_csv(group_labels_path, GroupLabel, store.client_id) if group_labels_path else ()
    )
    triage_labels = (
        read_csv(triage_labels_path, TriageLabel, store.client_id) if triage_labels_path else ()
    )
    rows, latest = [], {}
    totals = []
    for report in reports:
        review_rows = _review_rows(report, index, as_of)
        rows.extend(review_rows)
        date_key = report.manifest.knowledge_cutoff.astimezone(zone).date()
        by_horizon = defaultdict(list)
        for row in review_rows:
            by_horizon[row.horizon].append(row)
            previous = latest.get(row.key)
            if previous is None or (row.decision_at, row.review_id) > (
                previous.decision_at,
                previous.review_id,
            ):
                latest[row.key] = row
        totals.extend(
            _comparisons(review_rows, review_id=report.manifest.review_id, review_date=date_key)
        )
        for horizon, group in sorted(by_horizon.items()):
            totals.extend(
                _comparisons(
                    group,
                    review_id=report.manifest.review_id,
                    review_date=date_key,
                    horizon=horizon,
                )
            )
    unique = tuple(latest.values())
    totals.extend(_comparisons(unique))
    checked = tuple(review_usage(report, reproduced=True) for report in reports)
    return EvaluationResult(
        client_id=profile.client_id,
        as_of=as_of,
        review_ids=review_ids,
        outcome_input_digest=content_digest([item.model_dump(mode="json") for item in outcomes]),
        group_label_digest=content_digest([item.model_dump(mode="json") for item in group_labels])
        if group_labels_path
        else None,
        triage_label_digest=content_digest([item.model_dump(mode="json") for item in triage_labels])
        if triage_labels_path
        else None,
        rows=tuple(rows),
        aggregate_unique_rows=len(unique),
        repeated_rows_removed=len(rows) - len(unique),
        point_comparisons=tuple(totals),
        conditional_ranges=_range_summary(unique),
        labels=_labels(reports, group_labels, triage_labels),
        reviews=checked,
        aggregate_usage=provider_usage(reports),
        aggregate_elapsed_seconds=sum(item.elapsed_seconds for item in checked),
        unmatched_outcomes=len(
            index.actuals.keys() - {(row.series_id, row.period, row.target) for row in rows}
        ),
        limitations=tuple(sorted({value for item in verified for value in item.limitations})),
    )
