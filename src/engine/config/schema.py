"""Typed client profile.

Every client gets one YAML file validated against these models. `extra="forbid"`
throughout: a misspelled key fails at load time rather than silently falling back
to a default nobody intended.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Granularity = Literal["monthly", "weekly", "daily"]

SEASONAL_PERIODS: dict[Granularity, int] = {"monthly": 12, "weekly": 52, "daily": 7}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ColumnMap(StrictModel):
    """Source-column names for each canonical field.

    Defaults match the demand-history export; a differently-shaped source file is
    onboarded by overriding names here rather than by editing reader code.
    """

    series_id: str = "Series ID"
    customer: str = "Customer"
    item_code: str = "Item Code"
    description: str = "Description"
    item_group: str = "Item Group"
    period: str = "Demand Month"
    quantity: str = "Demand Quantity"
    uom: str = "UOM"
    source_sheet: str = "Source Sheet"
    ground_truth_pattern: str | None = "Demand Pattern"


class SourceConfig(StrictModel):
    reader: Literal["excel_demand"] = "excel_demand"
    sheet: str | None = None
    column_map: ColumnMap = Field(default_factory=ColumnMap)


class CalendarConfig(StrictModel):
    granularity: Granularity = "monthly"
    fiscal_year_start_month: int = Field(default=4, ge=1, le=12)

    @property
    def seasonal_period(self) -> int:
        return SEASONAL_PERIODS[self.granularity]


class HierarchyConfig(StrictModel):
    """Which columns identify a series, and which group it for reconciliation.

    `grouping_keys` are independent axes (product group, plant), not levels of a
    single tree — they are not assumed to nest.
    """

    series_key: list[str] = Field(default_factory=lambda: ["customer", "item_code"])
    grouping_keys: list[str] = Field(default_factory=lambda: ["item_group", "source_sheet"])


class EvaluationConfig(StrictModel):
    horizon: int = Field(default=3, ge=1)
    origins: int = Field(default=3, ge=1)
    origin_step: int = Field(default=1, ge=1)
    min_train_periods: int = Field(default=24, ge=2)


class PatternThresholds(StrictModel):
    """Cutoffs for the per-series demand-pattern classifier (see features/pattern.py).

    Seasonal strength sits high because STL finds apparent seasonality in any
    short, noisy series — with only three seasonal cycles of history, the
    seasonal component absorbs noise, and a conventional 0.6 cutoff labels most
    flat series seasonal.
    """

    zero_fraction_intermittent: float = Field(default=0.30, ge=0, le=1)
    adi_intermittent: float = Field(default=1.32, gt=0)
    seasonal_strength: float = Field(default=0.88, ge=0, le=1)
    trend_strength: float = Field(default=0.60, ge=0, le=1)
    volatility_cv: float = Field(default=0.15, ge=0)


class LightGBMParams(StrictModel):
    objective: str = "l1"
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    num_boost_round: int = 400
    lambda_l2: float = 1.0
    verbosity: int = -1
    seed: int = 42


class ArbitrationConfig(StrictModel):
    """How per-series model selection is made.

    `min_improvement_pct` shrinks selection toward the champion model: with few
    scored observations per series, unconstrained per-series picking selects
    noise and loses to the champion out-of-sample.
    """

    selection_metric: Literal["mae", "wape"] = "mae"
    min_improvement_pct: float = Field(default=15.0, ge=0, le=100)


class ModelsConfig(StrictModel):
    """Model portfolio for this client.

    `baseline` is the reference every other model is scored against (Forecast
    Value Add); it must also appear in `enabled`.
    """

    enabled: list[str] = Field(
        default_factory=lambda: [
            "seasonal_naive",
            "naive",
            "auto_ets",
            "croston",
            "lightgbm_global",
        ]
    )
    baseline: str = "seasonal_naive"
    # Categorical series attributes the pooled model may learn from. Excludes
    # near-unique keys such as item_code, which would act as a series identifier.
    static_features: list[str] = Field(
        default_factory=lambda: ["customer", "item_group", "source_sheet"]
    )
    arbitration: ArbitrationConfig = Field(default_factory=ArbitrationConfig)
    lightgbm: LightGBMParams = Field(default_factory=LightGBMParams)

    @field_validator("enabled")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("models.enabled must list at least one forecaster")
        return value


class ClientProfile(StrictModel):
    client_id: str
    display_name: str
    currency: str = "INR"
    source: SourceConfig = Field(default_factory=SourceConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    patterns: PatternThresholds = Field(default_factory=PatternThresholds)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    # Reserved for later phases so their arrival does not break this schema.
    exogenous_signals: list[str] = Field(default_factory=list)
    cost_model: None = None

    @field_validator("client_id")
    @classmethod
    def _safe_identifier(cls, value: str) -> str:
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("client_id must be alphanumeric with underscores or hyphens only")
        return value

    def model_post_init(self, _context) -> None:
        if self.models.baseline not in self.models.enabled:
            raise ValueError(
                f"models.baseline {self.models.baseline!r} must also appear in models.enabled"
            )
