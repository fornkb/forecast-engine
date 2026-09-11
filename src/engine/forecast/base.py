"""Forecaster interface, output contract, and registry.

Every model — statistical, pooled ML, or a foundation model in a later phase —
implements the same narrow interface, so adding one is a new file plus a
registry entry and nothing else in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from engine.config.schema import ClientProfile, Granularity

FORECAST_SCHEMA: dict[str, pl.DataType] = {
    "series_id": pl.Utf8,
    "period": pl.Date,
    "horizon": pl.Int32,
    "forecast": pl.Float64,
}


@dataclass(frozen=True)
class ForecastContext:
    """Everything a forecaster may know beyond the history it is given."""

    client: ClientProfile
    series_meta: pl.DataFrame
    # Context features, wide (series_id, period, <feature>...). This deliberately
    # includes periods beyond the forecast cutoff, which is not leakage: these
    # are signals known in advance — a planned shelf price, a promotion
    # calendar, a public holiday. A driver only observable after the fact would
    # not belong here.
    exogenous: pl.DataFrame | None = None

    @property
    def seasonal_period(self) -> int:
        return self.client.calendar.seasonal_period

    @property
    def granularity(self) -> Granularity:
        return self.client.calendar.granularity

    @property
    def quantile_levels(self) -> list[float]:
        """Quantiles to produce, or empty when probabilistic output is off."""
        config = self.client.probabilistic
        return config.levels if config.enabled else []


class Forecaster(ABC):
    """Produces a forecast for every series in `history`.

    Implementations override `fit_predict`; callers use `predict`, which enforces
    the output contract centrally.
    """

    name: ClassVar[str]

    def predict(self, history: pl.DataFrame, horizon: int, ctx: ForecastContext) -> pl.DataFrame:
        raw = self.fit_predict(history, horizon, ctx)
        return enforce_contract(raw, self.name)

    @abstractmethod
    def fit_predict(
        self, history: pl.DataFrame, horizon: int, ctx: ForecastContext
    ) -> pl.DataFrame:
        """Fit on `history` (everything up to the cutoff) and forecast `horizon` periods.

        `history` has columns series_id, period, quantity.
        Returns columns series_id, period, horizon, forecast.
        """


def _clean_non_negative(column: str) -> pl.Expr:
    """Nulls and NaNs become zero; demand cannot go below it."""
    return (
        pl.when(pl.col(column).is_null() | pl.col(column).is_nan())
        .then(0.0)
        .otherwise(pl.col(column).clip(lower_bound=0.0))
        .alias(column)
    )


def enforce_contract(frame: pl.DataFrame, model_name: str) -> pl.DataFrame:
    """Guarantee shape, dtypes, and non-negativity of any forecaster's output.

    Quantile columns are optional: a model that cannot express uncertainty
    simply omits them, and the scoreboard scores it on point accuracy only.
    """
    missing = [column for column in FORECAST_SCHEMA if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecaster {model_name!r} returned no {missing} column(s)")

    quantile_columns = [c for c in frame.columns if c.startswith("q") and c[1:2].isdigit()]

    result = frame.select(
        pl.col("series_id").cast(pl.Utf8),
        pl.col("period").cast(pl.Date),
        pl.col("horizon").cast(pl.Int32),
        pl.col("forecast").cast(pl.Float64),
        *[pl.col(name).cast(pl.Float64) for name in quantile_columns],
    ).with_columns([_clean_non_negative(name) for name in ["forecast", *quantile_columns]])

    if len(quantile_columns) > 1:
        # Quantile regression fits each level independently, so nothing stops
        # the 90th percentile landing below the 10th. Sorting restores a valid
        # distribution; the alternative is a negative-width interval reaching
        # the safety-stock calculation.
        result = _enforce_monotone_quantiles(result, quantile_columns)

    return result


def conform(frame: pl.DataFrame, columns: list[str], quantile_columns: list[str]) -> pl.DataFrame:
    """Give one model's output the shared schema, in a fixed column order.

    Models differ in whether they express uncertainty, so absent quantile levels
    are filled with nulls. Order matters as much as names: a vertical concat
    matches columns by position, and a model that emits quantiles places them
    differently from nulls appended afterwards. Every multi-model frame in the
    engine goes through here so that invariant lives in exactly one place.
    """
    absent = [name for name in quantile_columns if name not in frame.columns]
    return frame.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(name) for name in absent]
    ).select(columns + quantile_columns)


def _enforce_monotone_quantiles(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    ordered = sorted(columns, key=_quantile_level)
    running = pl.col(ordered[0])
    exprs = [running.alias(ordered[0])]
    for name in ordered[1:]:
        running = pl.max_horizontal(running, pl.col(name))
        exprs.append(running.alias(name))
    return frame.with_columns(exprs)


def _quantile_level(column: str) -> float:
    return float(column[1:].replace("_", "."))


_FORECASTERS: dict[str, type[Forecaster]] = {}


def register_forecaster(cls: type[Forecaster]) -> type[Forecaster]:
    _FORECASTERS[cls.name] = cls
    return cls


def get_forecaster(name: str) -> Forecaster:
    if name not in _FORECASTERS:
        raise KeyError(f"Unknown forecaster {name!r}; registered: {sorted(_FORECASTERS)}")
    return _FORECASTERS[name]()


def available_forecasters() -> list[str]:
    return sorted(_FORECASTERS)
