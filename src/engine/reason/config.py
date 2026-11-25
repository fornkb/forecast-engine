"""Per-client mappings for forecast context."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, StringConstraints, field_validator, model_validator

from engine.config.base import StrictModel

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Relation = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
]


class TargetConfig(StrictModel):
    id: Text
    definition: Text
    outcome_source: Text
    unit_field: Literal["uom"] | None = None
    units_by_series: dict[Text, Text] = Field(default_factory=dict)


class ContextFields(StrictModel):
    record_id: Text
    subject_id: Text
    object_id: Text | None = None
    value: Text | None = None
    effective_from: Text
    effective_to: Text | None = None
    known_at: Text | None = None
    version: Text | None = None
    status: Text | None = None
    client_id: Text | None = None


class ContextSource(StrictModel):
    id: Text
    kind: Literal["csv", "postgres"]
    dataset: Literal["relationship", "entity"]
    relation_type: Text
    subject_type: Text
    object_type: Text | None = None
    location_role: Literal["customer", "site", "supplier", "route"] | None = None
    value_type: Literal["text", "number", "boolean"] = "text"
    cardinality: Literal["one", "many"] = "many"
    required_for_forecast: bool = Field(default=True, strict=True)
    naive_timestamps: Literal["reject", "client_local"] = "reject"
    path: Path | None = None
    relation: Relation | None = None
    dsn_env: Identifier | None = None
    fields: ContextFields

    @model_validator(mode="after")
    def validate_mapping(self) -> ContextSource:
        if self.id == "operator_confirmation":
            raise ValueError("operator_confirmation is reserved for confirmed facts")
        mapping = self.fields.model_dump(exclude_none=True)
        if self.kind == "csv":
            if self.path is None or self.relation is not None or self.dsn_env is not None:
                raise ValueError("CSV context requires path and forbids relation and dsn_env")
        else:
            if self.relation is None or self.dsn_env is None or self.path is not None:
                raise ValueError(
                    "PostgreSQL context requires relation and dsn_env and forbids path"
                )
            for name in mapping.values():
                if not name.isascii() or not ("_" + name).isidentifier() or name[0].isdigit():
                    raise ValueError(f"Invalid PostgreSQL context column identifier {name!r}")
        if self.dataset == "relationship":
            if not self.fields.object_id or not self.object_type or self.fields.value:
                raise ValueError("A relationship requires object_id and object_type")
        elif not self.fields.value or self.fields.object_id or self.object_type:
            raise ValueError(
                "An entity attribute requires value and has no object_id or object_type"
            )
        if self.location_role and "location" not in (self.subject_type, self.object_type):
            raise ValueError("location_role requires a location subject or object")
        return self


OperationalDataset = Literal[
    "bom",
    "vendor_link",
    "purchase_order",
    "receipt",
    "usable_stock",
    "consumption",
    "alternate",
    "allocation",
    "order_amendment",
    "contract_option",
    "method_parameter",
    "planner_assumption",
]

_MATERIAL = {"material_id", "site_id", "unit"}
_DEMAND = {
    "customer_id",
    "item_id",
    "quantity",
    "unit",
    "status",
    "baseline_includes",
    "version",
    "prior_version",
}
OPERATIONAL_FIELDS = {
    "bom": _MATERIAL | {"item_id", "quantity", "finished_unit"},
    "vendor_link": _MATERIAL | {"supplier_id"},
    "purchase_order": _MATERIAL | {"quantity", "status"},
    "receipt": _MATERIAL | {"quantity", "status"},
    "usable_stock": _MATERIAL | {"quantity"},
    "consumption": _MATERIAL | {"quantity"},
    "alternate": _MATERIAL | {"alternate_material_id", "status"},
    "allocation": _MATERIAL | {"item_id", "quantity", "status", "effective_to"},
    "order_amendment": _DEMAND,
    "contract_option": _DEMAND,
    "method_parameter": {"parameter_id", "quantity", "unit", "actor", "status"},
    "planner_assumption": {"parameter_id", "quantity", "unit", "actor", "status"},
}
OPERATIONAL_SCENARIO_FIELDS = {
    "target_id",
    "basis",
    "parameter_role",
    "validation_ref",
    "training_cutoff",
    "minimum_driver",
    "maximum_driver",
    "lag_months",
    "min_horizon",
    "max_horizon",
}
OPERATIONAL_OPTIONAL = {
    "effective_to",
    "known_at",
    "version",
    "prior_version",
    "status",
    "client_id",
    "warehouse_id",
    "item_id",
    "customer_id",
    "series_id",
    "supplier_id",
    "route_id",
    "site_id",
    "material_id",
    "lead_time_days",
    "ordered_at",
    "quantity",
    "condition",
    "converted_unit",
    "conversion_factor",
    "receipt_id",
    "event_id",
    "reservation_id",
    "original_date",
    "actor",
    "baseline_includes",
    *OPERATIONAL_SCENARIO_FIELDS,
}


class OperationalSource(StrictModel):
    id: Text
    kind: Literal["csv", "postgres"]
    dataset: OperationalDataset
    path: Path | None = None
    relation: Relation | None = None
    dsn_env: Identifier | None = None
    naive_timestamps: Literal["reject", "client_local"] = "reject"
    fields: dict[Text, Text]
    max_rows: int = Field(default=2000, ge=1, le=10000)
    cache_seconds: int = Field(default=0, ge=0, le=86400)
    stock_basis: Literal["site_total", "warehouse"] = "site_total"

    @model_validator(mode="after")
    def validate_mapping(self) -> OperationalSource:
        required = OPERATIONAL_FIELDS[self.dataset] | {"record_id", "effective_from"}
        if self.dataset == "usable_stock" and self.stock_basis == "warehouse":
            required = required | {"warehouse_id"}
        if required - self.fields.keys() or self.fields.keys() - required - OPERATIONAL_OPTIONAL:
            raise ValueError(
                f"Invalid field mapping for {self.dataset}; requires {sorted(required)}"
            )
        if self.kind == "csv":
            if self.path is None or self.relation is not None or self.dsn_env is not None:
                raise ValueError("CSV evidence requires only path")
        elif self.relation is None or self.dsn_env is None or self.path is not None:
            raise ValueError("PostgreSQL evidence requires relation and dsn_env")
        elif any(not name.isascii() or not name.isidentifier() for name in self.fields.values()):
            raise ValueError("Invalid PostgreSQL evidence column identifier")
        return self


class ProviderPolicy(StrictModel):
    timeout_seconds: float = Field(default=10, gt=0, le=30)
    max_attempts: int = Field(default=12, ge=0, le=100)
    retries: int = Field(default=1, ge=0, le=2)
    cache_seconds: int = Field(default=21600, ge=0, le=604800)


class NewsSearch(StrictModel):
    id: Text
    public_alias: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
    ]
    geography: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)] | None
    ) = None
    entity_type: Literal["company", "item", "material", "supplier", "route", "location"] | None = (
        None
    )
    entity_id: Text | None = None
    primary_domains: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def complete_entity(self) -> NewsSearch:
        if (self.entity_type is None) != (self.entity_id is None):
            raise ValueError(
                "Public news searches require both entity_type and entity_id or neither"
            )
        return self


class BraveNewsConfig(ProviderPolicy):
    api_key_env: Identifier = "BRAVE_SEARCH_API_KEY"
    page_size: int = Field(default=10, ge=1, le=50)
    max_pages: int = Field(default=1, ge=1, le=3)
    lookback_days: int = Field(default=45, ge=1, le=365)
    searches: tuple[NewsSearch, ...] = Field(max_length=50)

    @model_validator(mode="after")
    def unique_searches(self) -> BraveNewsConfig:
        if len({search.id for search in self.searches}) != len(self.searches):
            raise ValueError("Public news search IDs must be unique")
        return self


WeatherRole = Literal["customer", "site", "supplier", "route"]


class WeatherLocation(StrictModel):
    location_id: Text
    role: WeatherRole
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class WeatherClimatology(StrictModel):
    location_id: Text
    month: int = Field(ge=1, le=12)
    variable: Literal["temperature_2m", "precipitation"]
    unit: Literal["°C", "°F", "mm", "inch"]
    value: float = Field(allow_inf_nan=False)
    first_year: int = Field(ge=1800, le=2200)
    last_year: int = Field(ge=1800, le=2200)
    model: Literal["ecmwf_seas5"]
    timezone: Text
    revision: Text
    reference: Text

    @model_validator(mode="after")
    def ordered_years(self) -> WeatherClimatology:
        if self.last_year < self.first_year:
            raise ValueError("Climatology years must be ordered")
        return self


class SeasonalWeatherConfig(ProviderPolicy):
    access: Literal["commercial", "noncommercial"] = "commercial"
    api_key_env: Identifier = "OPEN_METEO_API_KEY"
    ensemble_members: bool = False
    temperature_unit: Literal["celsius", "fahrenheit"] = "celsius"
    precipitation_unit: Literal["mm", "inch"] = "mm"
    locations: tuple[WeatherLocation, ...] = Field(default=(), max_length=50)
    applicable_roles: tuple[WeatherRole, ...] = ("customer", "site", "supplier", "route")
    climatology: tuple[WeatherClimatology, ...] = ()

    @model_validator(mode="after")
    def unique_locations(self) -> SeasonalWeatherConfig:
        locations = {(value.location_id, value.role) for value in self.locations}
        references = {
            (value.location_id, value.month, value.variable) for value in self.climatology
        }
        if len(locations) != len(self.locations) or len(references) != len(self.climatology):
            raise ValueError("Weather locations and climatology entries must be unique")
        return self


class EnsoConfig(ProviderPolicy):
    path: Annotated[
        str,
        StringConstraints(pattern=r"^/products/[A-Za-z0-9_/-]+\.s?html?$"),
    ] = "/products/analysis_monitoring/enso_advisory/ensodisc.shtml"


class EvidenceConfig(StrictModel):
    erp: tuple[OperationalSource, ...] = ()
    news: BraveNewsConfig | None = None
    weather: SeasonalWeatherConfig | None = None
    enso: EnsoConfig | None = None
    max_attempts: int = Field(default=30, ge=0, le=250)
    max_erp_attempts: int = Field(default=20, ge=0, le=100)
    erp_retries: int = Field(default=1, ge=0, le=2)
    followup_limit: int = Field(default=3, ge=0, le=3)

    @model_validator(mode="after")
    def unique_erp_sources(self) -> EvidenceConfig:
        if len({source.id for source in self.erp}) != len(self.erp):
            raise ValueError("Operational source IDs must be unique")
        return self


class AssistantConfig(StrictModel):
    model_id: Text | None = None
    api_key_env: Identifier = "ANTHROPIC_API_KEY"
    timeout_seconds: float = Field(default=30, gt=0, le=60)
    max_requests: int = Field(default=40, ge=0, le=250)
    max_tool_rounds: int = Field(default=4, ge=0, le=12)
    max_input_bytes: int = Field(default=48000, ge=4000, le=250000)
    max_output_tokens: int = Field(default=4096, ge=512, le=16384)
    retries: int = Field(default=1, ge=0, le=2)


class ProviderRate(StrictModel):
    rate_id: Text
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    per_request: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    per_million_input: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    per_million_output: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    per_million_cache_read: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    per_million_cache_creation: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class EvaluationConfig(StrictModel):
    outcome_columns: dict[Text, Text] = Field(default_factory=dict)
    rates: dict[Text, ProviderRate] = Field(default_factory=dict)


class ReasoningConfig(StrictModel):
    enabled: bool = Field(default=False, strict=True)
    timezone: Text = "UTC"
    target: TargetConfig | None = None
    sources: tuple[ContextSource, ...] = ()
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @field_validator("timezone")
    @classmethod
    def iana_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("timezone must be an available IANA zone (e.g. Asia/Kolkata)") from exc
        return value

    @model_validator(mode="after")
    def unique_sources(self) -> ReasoningConfig:
        if len({source.id for source in self.sources}) != len(self.sources):
            raise ValueError("reasoning source IDs must be unique")
        return self
