"""Typed client profile.

Every client gets one YAML file validated against these models. `extra="forbid"`
throughout: a misspelled key fails at load time rather than silently falling back
to a default nobody intended.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class DatabaseSourceConfig(StrictModel):
    """Connection details for a database-backed source.

    The password never appears here. `dsn_env` names an environment variable
    holding the whole DSN, so a profile can be committed and shared while the
    credential stays on the machine that runs the ingest.
    """

    dsn_env: str = "ENGINE_PG_DSN"
    # Schema-qualified view or table. In a real deployment this is a view the
    # ERP publishes — the integration contract — not a base table.
    relation: str
    # Server-side filter, applied before anything crosses the wire.
    where: str | None = None
    # Identity used for idempotent re-ingest. Defaults to the relation name,
    # which is stable and, unlike the DSN, safe to store and print.
    source_label: str | None = None

    @property
    def label(self) -> str:
        return self.source_label or f"postgres:{self.relation}"

    @property
    def query(self) -> str:
        clause = f" WHERE {self.where}" if self.where else ""
        return f"SELECT * FROM {self.relation}{clause}"


class SourceConfig(StrictModel):
    # Must match a name registered in engine.ingest; kept as a literal so a typo
    # fails at config load rather than at the end of an ingest.
    reader: Literal["excel_demand", "parquet_panel", "csv_panel", "postgres_panel"] = "excel_demand"
    sheet: str | None = None
    database: DatabaseSourceConfig | None = None
    column_map: ColumnMap = Field(default_factory=ColumnMap)

    @model_validator(mode="after")
    def _database_matches_reader(self) -> SourceConfig:
        """A database block and a database reader have to agree.

        `extra="forbid"` catches misspelled keys but not a correctly-spelled
        block attached to a reader that will never look at it — which fails
        silently, as a file read against a profile that says it uses Postgres.
        """
        if self.reader == "postgres_panel" and self.database is None:
            raise ValueError("reader 'postgres_panel' requires a source.database block")
        if self.reader != "postgres_panel" and self.database is not None:
            raise ValueError(
                f"source.database is set but reader is {self.reader!r}; "
                "it would be ignored. Use reader 'postgres_panel' or drop the block."
            )
        return self


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


class ExogenousConfig(StrictModel):
    """Context features supplied alongside demand history.

    Values are taken at the period being forecast, not at the forecast origin,
    because these are things a planner knows in advance — a planned shelf price,
    a promotion calendar, a public holiday. That is what makes them usable as
    inputs rather than merely explanatory after the fact.
    """

    features: list[str] = Field(default_factory=list)
    # Level-valued features (a price, a temperature) additionally get a ratio
    # against their value at the forecast origin, which is the signal a planner
    # actually reasons about: not "the price is 3.20" but "the price dropped 20%".
    level_features: list[str] = Field(default_factory=list)

    def model_post_init(self, _context) -> None:
        unknown = set(self.level_features) - set(self.features)
        if unknown:
            raise ValueError(
                f"exogenous.level_features {sorted(unknown)} are not listed in exogenous.features"
            )


class SegmentThresholds(StrictModel):
    """Cumulative volume-share boundaries for Head / Mid / Tail segmentation.

    Defaults follow the conventional Pareto split: the head is whatever few
    series carry the first 80% of demand, the tail the last 5%.
    """

    head_share: float = Field(default=0.80, gt=0, lt=1)
    mid_share: float = Field(default=0.95, gt=0, lt=1)

    def model_post_init(self, _context) -> None:
        if self.mid_share <= self.head_share:
            raise ValueError("segments.mid_share must be greater than segments.head_share")


class ProbabilisticConfig(StrictModel):
    """Quantiles every model is asked to produce.

    Defined once at client level rather than per model, so the forecasts being
    compared are always the same quantiles. Safety stock, reorder points and any
    service-level target need a distribution, not a point — a point forecast
    forces the uncertainty to be assumed somewhere else.
    """

    enabled: bool = True
    levels: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

    @field_validator("levels")
    @classmethod
    def _valid_levels(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("probabilistic.levels must not be empty")
        if any(not 0 < level < 1 for level in value):
            raise ValueError("probabilistic.levels must all lie strictly between 0 and 1")
        if len(set(value)) != len(value):
            raise ValueError("probabilistic.levels must not repeat")
        return sorted(value)

    @property
    def column_names(self) -> list[str]:
        return [quantile_column(level) for level in self.levels]


def quantile_column(level: float) -> str:
    """Column name for a quantile level: 0.1 -> q10, 0.975 -> q97_5."""
    text = f"{level * 100:g}".replace(".", "_")
    return f"q{text}"


class ChronosParams(StrictModel):
    """Zero-shot foundation model settings.

    The model is pretrained and never fitted here, so there are no training
    hyperparameters — only what to load and how to run it.
    """

    model_id: str = "amazon/chronos-2"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    batch_size: int = Field(default=64, ge=1)


class LightGBMParams(StrictModel):
    # Generate training rows from only the most recent N periods of each series.
    # Bounds memory on large panels, and recent history is usually the more
    # relevant signal anyway. None uses everything available.
    max_training_periods: int | None = Field(default=None, ge=2)
    # Winsorise the scaled training target at this quantile. See
    # features/panel.build_training_frame — without it, upper-quantile
    # regression inherits ratios from near-zero-mean series and produces
    # forecasts orders of magnitude above anything ever observed.
    target_cap_quantile: float | None = Field(default=0.995, gt=0.5, lt=1.0)
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
    # Which slice each series inherits its default model from. A single global
    # default is wrong for one group or another once the portfolio holds models
    # with complementary strengths across volume or demand pattern.
    stratify_by: Literal["none", "volume_segment", "pattern"] = "volume_segment"
    # The champion is the default for a *typical* series in its stratum, so it is
    # scored with the scale-free metric. Aggregate MAE within a stratum is
    # dominated by its few largest series, which answers a different question;
    # those series can still override the default through per-series selection.
    champion_metric: Literal["median_series_wape", "mae", "wape"] = "median_series_wape"


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
            "chronos2",
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
    chronos: ChronosParams = Field(default_factory=ChronosParams)

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
    segments: SegmentThresholds = Field(default_factory=SegmentThresholds)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    exogenous: ExogenousConfig = Field(default_factory=ExogenousConfig)
    probabilistic: ProbabilisticConfig = Field(default_factory=ProbabilisticConfig)

    # Reserved for the decision layer so its arrival does not break this schema.
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
