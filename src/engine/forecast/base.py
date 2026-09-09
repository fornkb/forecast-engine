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


def enforce_contract(frame: pl.DataFrame, model_name: str) -> pl.DataFrame:
    """Guarantee shape, dtypes, and non-negativity of any forecaster's output."""
    missing = [column for column in FORECAST_SCHEMA if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecaster {model_name!r} returned no {missing} column(s)")

    result = frame.select(
        pl.col("series_id").cast(pl.Utf8),
        pl.col("period").cast(pl.Date),
        pl.col("horizon").cast(pl.Int32),
        pl.col("forecast").cast(pl.Float64),
    )

    # Statistical models can extrapolate below zero; demand cannot go there.
    return result.with_columns(
        pl.when(pl.col("forecast").is_null() | pl.col("forecast").is_nan())
        .then(0.0)
        .otherwise(pl.col("forecast").clip(lower_bound=0.0))
        .alias("forecast")
    )


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
