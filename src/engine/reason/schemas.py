"""Runtime records shared by the advisory services."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, StrictFloat, StrictStr, StringConstraints

from engine.config.base import StrictModel
from engine.reason.config import (
    OperationalDataset,
    ProviderRate,
    TargetConfig,
    WeatherClimatology,
    WeatherRole,
)


class ContextError(RuntimeError):
    """An invalid forecast, context input, or operator action."""


def content_digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


class ContextGap(StrictModel):
    code: str
    message: str
    series_id: str | None = None
    period: date | None = None
    fact_version_ids: tuple[str, ...] = ()


class ForecastRow(StrictModel):
    client_id: str
    forecast_run_id: int
    series_id: str
    period: date
    target_start: AwareDatetime
    target_end: AwareDatetime
    cutoff_period: date
    horizon: int
    model: str
    baseline: float
    item_id: str | None = None
    customer_id: str | None = None
    unit: str | None = None
    fact_version_ids: tuple[str, ...] = ()
    superseded_fact_version_ids: tuple[str, ...] = ()


class ForecastSnapshot(StrictModel):
    snapshot_id: str
    client_id: str
    forecast_run_id: int
    input_digest: str
    config_digest: str
    metadata_digest: str
    captured_at: AwareDatetime
    metadata_observed_at: AwareDatetime
    forecast_created_at_local: datetime
    forecast_available_at: AwareDatetime | None = None
    timezone: str
    granularity: str = "monthly"
    target: TargetConfig | None = None
    gaps: tuple[ContextGap, ...] = ()
    question_ids: tuple[str, ...] = ()
    rows: tuple[ForecastRow, ...]

    def iter_rows(self, series_id: str | None = None) -> Iterator[ForecastRow]:
        return (row for row in self.rows if series_id is None or row.series_id == series_id)


class EntityRef(StrictModel):
    type: str
    id: str


FactValue = StrictStr | StrictFloat | StrictBool | TargetConfig
LocationRole = WeatherRole
AnswerType = Literal["text", "number", "boolean", "object_id", "target"]


class CompanyFact(StrictModel):
    client_id: str
    fact_version_id: str
    source_id: str
    record_id: str
    source_ref: str
    source_version: str | None = None
    content_digest: str
    subject: EntityRef
    relation: str
    object: EntityRef | None = None
    value: FactValue | None = None
    location_role: LocationRole | None = None
    cardinality: Literal["one", "many"] = "many"
    effective_from: AwareDatetime
    effective_to: AwareDatetime | None = None
    known_at: AwareDatetime
    captured_at: AwareDatetime
    status: Literal["active", "withdrawn", "confirmed"]
    actor: str | None = None
    resolves_fact_ids: tuple[str, ...] = ()

    @property
    def field_key(self) -> tuple[EntityRef, str, LocationRole | None]:
        return self.subject, self.relation, self.location_role


class FactConflict(StrictModel):
    subject: EntityRef
    relation: str
    location_role: LocationRole | None = None
    fact_version_ids: tuple[str, ...]


class QuestionSpec(StrictModel):
    subject: EntityRef
    field: str
    purpose: Literal["missing_fact", "fact_conflict", "mapping_confirmation"]
    message: str
    answer_type: AnswerType
    object_type: str | None = None
    location_role: LocationRole | None = None
    cardinality: Literal["one", "many"] = "one"
    affected_months: tuple[date, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    candidate_fact_ids: tuple[str, ...] = ()


class QuestionAnswer(StrictModel):
    answer_id: str
    question_id: str
    revision: int
    value: FactValue
    actor: str
    confirmed_at: AwareDatetime
    fact_version_id: str
    supersedes_answer_id: str | None = None
    spec: QuestionSpec | None = None


class Question(StrictModel):
    client_id: str
    question_id: str
    spec: QuestionSpec
    status: Literal["pending", "confirmed"]
    opened_at: AwareDatetime
    answer: QuestionAnswer | None = None


class SyncResult(StrictModel):
    new: int = 0
    reused: int = 0
    conflicted: int = 0
    invalid: int = 0


class ContextSyncError(ContextError):
    def __init__(self, message: str, invalid: int = 0) -> None:
        super().__init__(message)
        self.invalid = invalid


SourceKind = Literal["erp", "news", "weather", "enso"]
CheckStatus = Literal["found", "no_match", "unavailable", "not_applicable"]
CoverageStatus = Literal["complete", "partial", "none"]


class QueryScope(StrictModel):
    client_id: str
    source_kind: SourceKind
    source_id: str
    query_hash: str
    entity_ids: tuple[EntityRef, ...] = ()
    public_aliases: tuple[str, ...] = ()
    coordinates: tuple[float, float] | None = None
    location_role: WeatherRole | None = None
    valid_from: AwareDatetime
    valid_to: AwareDatetime
    months: tuple[date, ...]
    filters: tuple[tuple[str, str], ...] = ()
    page: int = 0


class VariableCoverage(StrictModel):
    variable: Literal["temperature_2m", "precipitation"]
    status: CoverageStatus
    complete_members: int
    expected_members: int


class MonthCoverage(StrictModel):
    period: date
    status: CoverageStatus
    reason: str | None = None
    variables: tuple[VariableCoverage, ...] = ()


class SourceCheck(StrictModel):
    check_id: str
    scope: QueryScope
    provider: str
    started_at: AwareDatetime
    ended_at: AwareDatetime
    status: CheckStatus
    coverage: CoverageStatus
    reason: str | None = None
    covered_from: AwareDatetime | None = None
    covered_to: AwareDatetime | None = None
    months: tuple[MonthCoverage, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    requests: int = 0
    attempts: int = 0
    retries: int = 0
    cache_hits: int = 0
    reused_check_id: str | None = None
    provider_status: str | None = None
    retryable: bool = False
    more_pages: bool = False


class OperationalRecord(StrictModel):
    dataset: OperationalDataset
    status: str
    movement: Literal["snapshot", "inflow", "outflow", "reference"]
    stock_basis: Literal["site_total", "warehouse"] | None = None
    material_id: str | None = None
    site_id: str | None = None
    warehouse_id: str | None = None
    item_id: str | None = None
    customer_id: str | None = None
    series_id: str | None = None
    supplier_id: str | None = None
    route_id: str | None = None
    alternate_material_id: str | None = None
    finished_unit: str | None = None
    converted_unit: str | None = None
    conversion_factor: float | None = None
    lead_time_days: int | None = None
    ordered_at: AwareDatetime | None = None
    original_date: AwareDatetime | None = None
    receipt_id: str | None = None
    reservation_id: str | None = None
    parameter_id: str | None = None
    actor: str | None = None
    prior_version: str | None = None
    condition: str | None = None
    baseline_includes: bool | None = None
    target_id: str | None = None
    basis: str | None = None
    parameter_role: (
        Literal["percent_change", "sensitivity", "scenario_driver", "expected_driver"] | None
    ) = None
    validation_ref: str | None = None
    training_cutoff: AwareDatetime | None = None
    minimum_driver: float | None = Field(default=None, allow_inf_nan=False)
    maximum_driver: float | None = Field(default=None, allow_inf_nan=False)
    lag_months: int | None = Field(default=None, ge=0, le=24)
    min_horizon: int | None = Field(default=None, ge=1)
    max_horizon: int | None = Field(default=None, ge=1)
    date_precision: Literal["day", "instant"]


class NewsArticle(StrictModel):
    title: str
    publisher: str
    primary: bool = False
    provider_page_date: str | None = None
    provider_fetched_at: AwareDatetime | None = None
    duplicate_key: str


class WeatherMember(StrictModel):
    member: str
    quantity: float | None
    covered_seconds: int


class MonthlyWeather(StrictModel):
    period: date
    variable: Literal["temperature_2m", "precipitation"]
    model: str
    timezone: str
    latitude: float
    longitude: float
    grid_resolution_km: int = 36
    bias_corrected: bool = False
    interval_seconds: int
    expected_seconds: int
    members: tuple[WeatherMember, ...]
    expected_members: int
    interval_semantics: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    climatology: WeatherClimatology | None = None
    anomaly: float | None = None


class EnsoSeason(StrictModel):
    label: str
    start: date
    end: date
    statements: tuple[str, ...] = ()


class EnsoAdvisory(StrictModel):
    issue_date: date
    geographic_scope: str = "Equatorial Pacific ENSO"
    seasons: tuple[EnsoSeason, ...]


class EvidenceRevision(StrictModel):
    client_id: str
    evidence_id: str
    record_id: str
    revision: int
    source_kind: SourceKind
    source_id: str
    source_ref: str
    source_version: str | None = None
    source_event_id: str | None = None
    issued_at: AwareDatetime | None = None
    source_known_at: AwareDatetime | None = None
    first_known_at: AwareDatetime
    captured_at: AwareDatetime
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    location_id: str | None = None
    excerpt: str | None = None
    quantity: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = None
    operational: OperationalRecord | None = None
    news: NewsArticle | None = None
    weather: MonthlyWeather | None = None
    enso: EnsoAdvisory | None = None


class QueryRequestContext(StrictModel):
    client_id: str
    forecast_run_id: int = Field(ge=1)
    snapshot_id: str
    review_id: str = Field(min_length=1, max_length=200)
    months: tuple[date, ...]
    request_budget: int = Field(default=3, ge=0, le=3)
    knowledge_cutoff: AwareDatetime | None = None


class LookupResult(StrictModel):
    checks: tuple[SourceCheck, ...]
    evidence_ids: tuple[str, ...] = ()
    attempts_used: int


class ProviderUsage(StrictModel):
    provider: str
    requests: int = 0
    attempts: int = 0
    retries: int = 0
    cache_hits: int = 0


class CandidateAssociation(StrictModel):
    evidence_id: str
    check_ids: tuple[str, ...]
    entities: tuple[EntityRef, ...] = ()
    basis: str
    corroborating_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()


class ExposureMember(StrictModel):
    series_id: str
    period: date
    item_id: str | None = None
    customer_id: str | None = None
    shortfall: float | None = None


class ExposureGroup(StrictModel):
    group_id: str
    kind: Literal["supply", "weather", "customer_order", "contract_option", "order_event", "news"]
    dependency: EntityRef
    period: date
    location_id: str | None = None
    location_role: WeatherRole | None = None
    evidence_ids: tuple[str, ...]
    fact_version_ids: tuple[str, ...] = ()
    members: tuple[ExposureMember, ...]
    exceptions: tuple[ExposureMember, ...] = ()
    flags: tuple[str, ...] = ()
    event_at: AwareDatetime | None = None
    shared_shortfall: float | None = None
    unit: str | None = None
    event_ids: tuple[str, ...] = ()


class EvidenceBatch(StrictModel):
    batch_id: str
    collection_id: str
    client_id: str
    forecast_run_id: int
    snapshot_id: str
    snapshot_captured_at: AwareDatetime
    query_version: str
    config_digest: str
    started_at: AwareDatetime
    knowledge_cutoff: AwareDatetime
    frozen_at: AwareDatetime
    check_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    groups: tuple[ExposureGroup, ...]
    candidates: tuple[CandidateAssociation, ...]
    gaps: tuple[str, ...] = ()
    status_counts: tuple[tuple[CheckStatus, int], ...] = ()
    complete: bool
    usage: tuple[ProviderUsage, ...] = ()


ReviewText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
MethodId = Literal["incremental_quantity", "proportional_change", "historical_driver_response"]
TargetType = Literal["demand", "cost", "material_availability", "production", "shipment_capacity"]
Direction = Literal["increase", "decrease", "mixed", "none", "unknown"]
Finding = Literal[
    "what_if_range", "possible_impact", "no_material_change_found", "incomplete_review"
]


class MappingProposal(StrictModel):
    event_id: ReviewText
    article_ids: tuple[ReviewText, ...] = Field(min_length=1)
    entities: tuple[EntityRef, ...] = Field(max_length=8)
    fact_version_ids: tuple[ReviewText, ...] = ()
    evidence_ids: tuple[ReviewText, ...] = ()
    months: tuple[date, ...]
    basis: ReviewText
    status: Literal["established", "uncertain", "irrelevant"]


class MappingDecision(MappingProposal):
    decision_id: str
    question_ids: tuple[str, ...] = ()
    requested_status: Literal["established", "uncertain", "irrelevant"]
    terminal_reason: str | None = None


class OperandReference(StrictModel):
    record_version_id: ReviewText
    field: Literal["quantity"]


class ConditionProposal(StrictModel):
    condition_id: ReviewText
    description: ReviewText
    operands: dict[str, OperandReference] = Field(max_length=16)


class AssessmentProposal(StrictModel):
    group_id: ReviewText
    event_ids: tuple[ReviewText, ...] = Field(min_length=1)
    series_ids: tuple[ReviewText, ...] | None = None
    target_type: TargetType
    periods: tuple[date, ...] = Field(min_length=1)
    direction: Direction
    finding: Finding
    method_id: MethodId | None = None
    conditions: tuple[ConditionProposal, ...] = Field(default=(), max_length=2)
    evidence_ids: tuple[ReviewText, ...] = ()
    fact_version_ids: tuple[ReviewText, ...] = ()
    contrary_ids: tuple[ReviewText, ...] = ()
    required_check_ids: tuple[ReviewText, ...] = ()
    missing_facts: tuple[ReviewText, ...] = ()
    rationale: ReviewText


class ScenarioScope(StrictModel):
    group_id: str
    event_ids: tuple[str, ...]
    target: str
    period: date
    unit: str
    series_ids: tuple[str, ...]


class ReviewQuestion(StrictModel):
    question_id: str
    purpose: Literal["company_fact", "mapping_confirmation", "scenario_assumption"]
    message: str
    status: Literal["pending", "confirmed", "cancelled"] = "pending"
    scope: ScenarioScope | None = None
    baselines: tuple[tuple[str, float], ...] = ()
    subject: EntityRef | None = None
    related_entity: EntityRef | None = None
    months: tuple[date, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    fact_version_id: str | None = None
    actor: str | None = None
    resolved_at: AwareDatetime | None = None


class AssumptionInput(StrictModel):
    method_id: Literal["incremental_quantity", "proportional_change"]
    low: StrictFloat = Field(allow_inf_nan=False)
    high: StrictFloat = Field(allow_inf_nan=False)
    low_condition: ReviewText
    high_condition: ReviewText
    rationale: ReviewText
    unit: ReviewText
    series_ids: tuple[ReviewText, ...] = Field(min_length=1)


class PlannerAssumption(StrictModel):
    assumption_id: str
    question_id: str
    revision: int
    supersedes_id: str | None = None
    scope: ScenarioScope
    values: AssumptionInput
    actor: str
    confirmed_at: AwareDatetime
    condition_ids: tuple[str, str]
    basis: Literal["conditional_planner_assumption"] = "conditional_planner_assumption"


class CalculationOperand(StrictModel):
    name: str
    record_version_id: str
    field: str
    value: float = Field(allow_inf_nan=False)
    unit: str
    source_ref: str
    basis: str


class CalculatedCondition(StrictModel):
    condition_id: str
    description: str
    delta: float = Field(allow_inf_nan=False)
    endpoint: float = Field(allow_inf_nan=False)
    operands: tuple[CalculationOperand, ...]
    basis_record_ids: tuple[str, ...] = ()
    option_applied: bool | None = None


class Assessment(StrictModel):
    review_id: str
    manifest_revision: int
    input_digest: str
    scenario_id: str
    group_id: str
    event_ids: tuple[str, ...]
    series_id: str
    item_id: str | None = None
    customer_id: str | None = None
    period: date
    horizon: int
    model: str
    target: str
    target_type: TargetType
    unit: str | None = None
    baseline: float | None = None
    forecast_unit: str | None = None
    forecast_baseline: float
    finding: Finding
    direction: Direction = "unknown"
    method_id: MethodId | None = None
    method_version: str | None = None
    conditions: tuple[CalculatedCondition, ...] = ()
    assumption_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    fact_version_ids: tuple[str, ...] = ()
    contrary_ids: tuple[str, ...] = ()
    required_check_ids: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    rationale: str
    supply_shortfall: float | None = None
    supply_balance: float | None = None
    supply_balance_scope: Literal["material_at_site"] | None = None
    supply_unit: str | None = None
    supply_at: AwareDatetime | None = None


class GroupResult(StrictModel):
    group: ExposureGroup
    input_digest: str
    proposal: AssessmentProposal | None = None
    terminal_reason: str | None = None


class ModelCall(StrictModel):
    call_id: str
    stage: Literal["triage", "assessment", "clarification"]
    work_id: str
    started_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    outcome: str = "reserved"
    request_kind: Literal["initial", "continuation", "retry"] | None = None
    token_usage_reported: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    response_tools: tuple[str, ...] = ()


class LookupAttempt(StrictModel):
    attempt_id: str
    query_hash: str
    provider: str
    started_at: AwareDatetime
    retry: bool
    check_id: str | None = None


class ReviewManifest(StrictModel):
    review_id: str
    client_id: str
    forecast_run_id: int
    snapshot_id: str
    base_batch_id: str
    forecast_digest: str
    collection_config_digest: str
    successor_of: str | None = None
    operator_action_digest: str | None = None
    budget_owner_id: str
    revision: int
    state: Literal["draft", "finalized"] = "draft"
    resume_mode: Literal["review", "operator_reassess", "operator_offline"] = "review"
    started_at: AwareDatetime
    collection_cutoff: AwareDatetime
    knowledge_cutoff: AwareDatetime
    finalized_at: AwareDatetime | None = None
    config_digest: str
    prompt_version: str
    method_version: str
    model_id: str | None = None
    model_request_limit: int
    followup_limit: int
    appended_fact_version_ids: tuple[str, ...] = ()
    appended_check_ids: tuple[str, ...] = ()
    appended_evidence_ids: tuple[str, ...] = ()
    requested_scopes: tuple[QueryScope, ...] = ()
    lookup_attempts: tuple[LookupAttempt, ...] = ()
    model_calls: tuple[ModelCall, ...] = ()
    candidates: tuple[CandidateAssociation, ...] = ()
    mappings: tuple[MappingDecision, ...] = ()
    assumptions: tuple[PlannerAssumption, ...] = ()
    questions: tuple[ReviewQuestion, ...] = ()
    groups: tuple[GroupResult, ...] = ()
    gaps: tuple[str, ...] = ()
    approved_lookups: dict | None = None
    source_access: dict[str, Literal["csv", "postgres"]] = Field(default_factory=dict)
    provider_rates: dict[str, ProviderRate] = Field(default_factory=dict)
    seal: ReviewSeal | None = None


class ReviewSeal(StrictModel):
    version: Literal["forecast-review-seal/1"] = "forecast-review-seal/1"
    manifest_digest: str
    records: dict[str, str]
