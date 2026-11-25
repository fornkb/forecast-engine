"""Deterministic, evidence-bound scenario arithmetic for monthly forecast rows."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from math import isfinite
from zoneinfo import ZoneInfo

from engine.reason.facts import local_midnight
from engine.reason.schemas import (
    AssessmentProposal,
    CalculatedCondition,
    CalculationOperand,
    CompanyFact,
    ContextError,
    EvidenceRevision,
    ExposureGroup,
    ForecastRow,
    ForecastSnapshot,
    PlannerAssumption,
    SourceCheck,
)

METHOD_VERSION = "forecast-scenarios/1"


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _period_add(period: date, offset: int) -> date:
    year, month = divmod(period.year * 12 + period.month - 1 + offset, 12)
    return date(year, month + 1, 1)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContextError(reason)


def validate_proposal(
    proposal: AssessmentProposal,
    group: ExposureGroup,
    evidence: dict[str, EvidenceRevision],
    facts: dict[str, CompanyFact],
    checks: dict[str, SourceCheck],
) -> AssessmentProposal:
    _require(
        proposal.group_id == group.group_id and proposal.periods == (group.period,),
        "assessment_scope_mismatch",
    )
    _require(
        len(proposal.event_ids) == len(group.event_ids)
        and set(proposal.event_ids) == set(group.event_ids),
        "assessment_event_mismatch",
    )
    if proposal.series_ids is not None:
        _require(
            bool(proposal.series_ids)
            and len(set(proposal.series_ids)) == len(proposal.series_ids)
            and set(proposal.series_ids) <= {member.series_id for member in group.members},
            "assessment_member_outside_verified_group",
        )
    cited = set(proposal.evidence_ids) | set(proposal.contrary_ids)
    cited.update(
        ref.record_version_id
        for condition in proposal.conditions
        for ref in condition.operands.values()
    )
    _require(
        cited <= evidence.keys() and set(proposal.fact_version_ids) <= facts.keys(),
        "assessment_unknown_reference",
    )
    _require(set(proposal.required_check_ids) <= checks.keys(), "assessment_unknown_source_check")
    if proposal.finding == "what_if_range":
        _require(
            proposal.method_id is not None
            and len(proposal.conditions) == 2
            and proposal.conditions[0].condition_id != proposal.conditions[1].condition_id,
            "range_requires_two_distinct_conditions",
        )
    else:
        _require(
            not proposal.conditions and proposal.method_id is None,
            "qualitative_result_cannot_supply_numeric_operands",
        )
    return proposal


def _operand(name: str, value: EvidenceRevision, basis: str) -> CalculationOperand:
    _require(value.quantity is not None and bool(value.unit), "operand_needs_quantity_and_unit")
    return CalculationOperand(
        name=name,
        record_version_id=value.evidence_id,
        field="quantity",
        value=value.quantity,
        unit=value.unit,
        source_ref=value.source_ref,
        basis=basis,
    )


def _row_scope(record, row: ForecastRow, target: str, *, order: bool = False) -> bool:
    if record.target_id != target:
        return False
    if record.series_id is None or order:
        return (
            record.series_id in (None, row.series_id)
            and bool(row.customer_id and row.item_id)
            and (record.customer_id, record.item_id) == (row.customer_id, row.item_id)
        )
    return (
        record.series_id == row.series_id
        and record.customer_id in (None, row.customer_id)
        and record.item_id in (None, row.item_id)
    )


def _orders(
    values: dict[str, EvidenceRevision],
    row: ForecastRow,
    snapshot: ForecastSnapshot,
    target: str,
    candidates: tuple[EvidenceRevision, ...],
) -> tuple[Decimal, tuple[CalculationOperand, ...]]:
    del snapshot
    permitted = {item.evidence_id for item in candidates}
    required = {item.evidence_id for item in candidates if item.operational.status == "confirmed"}
    selected = {item.evidence_id for item in values.values()}
    _require(
        bool(candidates) and required <= selected <= permitted and len(values) == len(selected),
        "incremental_order_basis_missing_or_ambiguous",
    )
    operands = tuple(order_operand(name, item, row, target) for name, item in values.items())
    return sum((_decimal(op.value) for op in operands), Decimal(0)), operands


def order_operand(name: str, item: EvidenceRevision, row: ForecastRow, target: str):
    record = item.operational
    _require(
        _row_scope(record, row, target, order=True)
        and item.unit == row.unit
        and row.target_start <= item.valid_from < row.target_end
        and record.baseline_includes is False
        and bool(item.source_version)
        and bool(record.prior_version)
        and item.source_version != record.prior_version,
        "order_increment_requires_matching_scope_unit_and_baseline_exclusion",
    )
    basis = record.basis or f"Signed change from {record.prior_version} to {item.source_version}"
    return _operand(name, item, basis)


def _parameter(
    item: EvidenceRevision,
    role: str,
    row: ForecastRow,
    target: str,
) -> None:
    record = item.operational
    _require(
        record is not None
        and record.dataset == "method_parameter"
        and record.parameter_role == role
        and bool(item.source_version)
        and _row_scope(record, row, target)
        and bool(record.basis),
        "unvalidated_or_out_of_scope_method_parameter",
    )


def _in_month(item: EvidenceRevision, row: ForecastRow) -> bool:
    return item.valid_from is not None and row.target_start <= item.valid_from < row.target_end


def _percent(
    values: dict[str, EvidenceRevision],
    row: ForecastRow,
    snapshot: ForecastSnapshot,
    target: str,
    candidates: tuple[EvidenceRevision, ...],
) -> tuple[Decimal, tuple[CalculationOperand, ...]]:
    del candidates, snapshot
    _require(set(values) == {"percent"} and row.baseline > 0, "percent_requires_nonzero_baseline")
    item = values["percent"]
    _parameter(item, "percent_change", row, target)
    record = item.operational
    _require(
        item.unit == "%"
        and _in_month(item, row)
        and record.baseline_includes is False
        and bool(record.validation_ref),
        "percent_requires_dated_target_effect_and_basis",
    )
    op = _operand("percent", item, f"{record.basis} ({record.validation_ref})")
    return _decimal(row.baseline) * _decimal(op.value) / 100, (op,)


def _driver(
    values: dict[str, EvidenceRevision],
    row: ForecastRow,
    snapshot: ForecastSnapshot,
    target: str,
    candidates: tuple[EvidenceRevision, ...],
) -> tuple[Decimal, tuple[CalculationOperand, ...]]:
    del candidates
    roles = ("sensitivity", "scenario_driver", "expected_driver")
    _require(set(values) == set(roles), "driver_requires_sensitivity_scenario_and_expected")
    for role, item in values.items():
        _parameter(item, role, row, target)
    relation, scenario, expected = (values[name] for name in roles)
    record = relation.operational
    zone = ZoneInfo(snapshot.timezone)
    trained_before = local_midnight(_period_add(row.cutoff_period, 1), zone)
    if snapshot.forecast_available_at is not None:
        trained_before = min(trained_before, snapshot.forecast_available_at)
    _require(
        record.training_cutoff is not None
        and record.training_cutoff <= trained_before
        and record.validation_ref
        and record.lag_months is not None
        and record.min_horizon is not None
        and record.max_horizon is not None
        and record.min_horizon <= row.horizon <= record.max_horizon
        and relation.valid_from <= row.target_start
        and (relation.valid_to is None or relation.valid_to >= row.target_end)
        and record.minimum_driver is not None
        and record.maximum_driver is not None,
        "driver_relationship_missing_validated_training_scope",
    )
    driver_month = _period_add(row.period, -record.lag_months)
    start = local_midnight(driver_month, zone)
    end = local_midnight(_period_add(driver_month, 1), zone)
    _require(
        scenario.unit is not None
        and scenario.unit == expected.unit
        and relation.unit == f"{row.unit}/{scenario.unit}"
        and scenario.operational.parameter_id
        == expected.operational.parameter_id
        == record.parameter_id
        and expected.operational.baseline_includes is True
        and scenario.operational.baseline_includes is False
        and start <= scenario.valid_from < end
        and start <= expected.valid_from < end
        and scenario.quantity is not None
        and expected.quantity is not None
        and all(
            record.minimum_driver <= item.quantity <= record.maximum_driver
            for item in (scenario, expected)
        ),
        "driver_outside_validated_unit_lag_or_plausible_range",
    )
    operands = tuple(
        _operand(name, item, item.operational.basis)
        for name, item in zip(roles, (relation, scenario, expected), strict=True)
    )
    return (
        _decimal(operands[0].value) * (_decimal(operands[1].value) - _decimal(operands[2].value))
    ), operands


METHODS = {
    "incremental_quantity": _orders,
    "proportional_change": _percent,
    "historical_driver_response": _driver,
}


def _endpoint(
    row: ForecastRow,
    condition_id: str,
    description: str,
    delta: Decimal,
    operands: tuple[CalculationOperand, ...],
    *,
    basis_ids: tuple[str, ...] = (),
    option_applied: bool | None = None,
) -> CalculatedCondition:
    endpoint = _decimal(row.baseline) + delta
    _require(
        isfinite(row.baseline)
        and row.baseline >= 0
        and isfinite(float(delta))
        and isfinite(float(endpoint))
        and endpoint >= 0,
        "invalid_or_negative_endpoint",
    )
    return CalculatedCondition(
        condition_id=condition_id,
        description=description,
        delta=float(delta),
        endpoint=float(endpoint),
        operands=operands,
        basis_record_ids=tuple(
            dict.fromkeys((*basis_ids, *(op.record_version_id for op in operands)))
        ),
        option_applied=option_applied,
    )


def _ordered(low: CalculatedCondition, high: CalculatedCondition):
    _require(low.endpoint <= high.endpoint, "reversed_conditions")
    _require(low.description.casefold() != high.description.casefold(), "conditions_must_differ")
    return low, high


def direction(conditions: tuple[CalculatedCondition, ...]) -> str:
    low, high = (condition.delta for condition in conditions)
    if low < 0 < high:
        return "mixed"
    if high > 0:
        return "increase"
    return "decrease" if low < 0 else "none"


def calculate_evidence(
    proposal: AssessmentProposal,
    group: ExposureGroup,
    row: ForecastRow,
    snapshot: ForecastSnapshot,
    evidence: dict[str, EvidenceRevision],
    target: str,
) -> tuple[CalculatedCondition, CalculatedCondition]:
    _require(proposal.target_type == "demand" and bool(row.unit), "unsupported_target_or_unit")
    _require(
        not any(
            flag
            in {
                "incompatible_unit",
                "ambiguous_customer_item_mapping",
                "conflicting_order_evidence",
                "partial_event_validity",
                "partial_regional_outlook",
            }
            for flag in group.flags
        ),
        "incomplete_or_conflicting_group_coverage",
    )
    _require(not proposal.contrary_ids, "conflicting_effect_requires_resolution")
    candidates = tuple(
        evidence[key]
        for key in group.evidence_ids
        if evidence[key].operational
        and evidence[key].operational.dataset in {"order_amendment", "contract_option"}
    )
    method = METHODS[proposal.method_id]
    related_events = {
        *group.event_ids,
        *(
            evidence[key].source_event_id
            for key in group.evidence_ids
            if evidence[key].source_event_id
        ),
    }
    conditions = []
    previous = []
    driver_basis = None
    for condition in proposal.conditions:
        selected = {
            name: evidence[ref.record_version_id] for name, ref in condition.operands.items()
        }
        delta, operands = method(selected, row, snapshot, target, candidates)
        label = condition.description
        basis_ids = ()
        includes = None
        if proposal.method_id == "incremental_quantity":
            options = [item for item in candidates if item.operational.status == "contingent"]
            _require(len(options) == 1, "two_order_conditions_require_one_dated_contingency")
            option = options[0]
            _require(bool(option.operational.condition), "order_option_missing_condition")
            includes = option.evidence_id in {item.evidence_id for item in selected.values()}
            previous.append(includes)
            basis_ids = (option.evidence_id,)
            label = (
                f"Condition applies: {option.operational.condition}"
                if includes
                else f"Condition does not apply: {option.operational.condition}"
            )
        else:
            driver = selected[
                "percent" if proposal.method_id == "proportional_change" else "scenario_driver"
            ]
            _require(
                driver.source_event_id in related_events and bool(driver.operational.condition),
                "scenario_input_requires_dated_condition_for_this_event",
            )
            label = driver.operational.condition
            if proposal.method_id == "historical_driver_response":
                pair = tuple(
                    selected[name].evidence_id for name in ("sensitivity", "expected_driver")
                )
                _require(
                    driver_basis is None or driver_basis == pair,
                    "driver_conditions_require_the_same_relationship_and_expected_value",
                )
                driver_basis = pair
        conditions.append(
            _endpoint(
                row,
                condition.condition_id,
                label,
                delta,
                operands,
                basis_ids=basis_ids,
                option_applied=includes,
            )
        )
    if proposal.method_id == "incremental_quantity":
        _require(previous[0] != previous[1], "order_conditions_must_change_contingency")
    return _ordered(*conditions)


def calculate_assumption(
    assumption: PlannerAssumption,
    row: ForecastRow,
) -> tuple[CalculatedCondition, CalculatedCondition]:
    values = assumption.values
    scope = assumption.scope
    _require(
        row.period == scope.period and row.series_id in scope.series_ids and row.unit == scope.unit,
        "assumption_row_outside_confirmed_scope",
    )
    percent = values.method_id == "proportional_change"
    _require(not percent or row.baseline > 0, "percent_requires_nonzero_baseline")
    conditions = []
    for name, condition_id, label, value in zip(
        ("low", "high"),
        assumption.condition_ids,
        (values.low_condition, values.high_condition),
        (values.low, values.high),
        strict=True,
    ):
        amount = _decimal(value)
        delta = _decimal(row.baseline) * amount / 100 if percent else amount
        operand = CalculationOperand(
            name=name,
            record_version_id=assumption.assumption_id,
            field=name,
            value=value,
            unit="%" if percent else row.unit,
            source_ref=f"assumption:{assumption.assumption_id}",
            basis=f"Conditional planner assumption by {assumption.actor}: {values.rationale}",
        )
        conditions.append(_endpoint(row, condition_id, label, delta, (operand,)))
    return _ordered(*conditions)
