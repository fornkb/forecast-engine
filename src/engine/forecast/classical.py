"""Classical per-series forecasters, via statsforecast's Numba-compiled models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, CrostonOptimized, SeasonalNaive

from engine.config.schema import Granularity
from engine.forecast.base import ForecastContext, Forecaster, register_forecaster

_POLARS_FREQ: dict[Granularity, str] = {"monthly": "1mo", "weekly": "1w", "daily": "1d"}


class StatsForecastAdapter(Forecaster):
    """Runs one statsforecast model across the whole panel.

    A failed fit on an individual series falls back to seasonal-naive rather than
    aborting the run — one pathological series should not cost a whole backtest.
    """

    @abstractmethod
    def build_model(self, ctx: ForecastContext) -> Any: ...

    def fit_predict(
        self, history: pl.DataFrame, horizon: int, ctx: ForecastContext
    ) -> pl.DataFrame:
        panel = history.select(
            pl.col("series_id").alias("unique_id"),
            pl.col("period").alias("ds"),
            pl.col("quantity").alias("y"),
        ).sort("unique_id", "ds")

        model = self.build_model(ctx)
        engine = StatsForecast(
            models=[model],
            freq=_POLARS_FREQ[ctx.granularity],
            fallback_model=SeasonalNaive(season_length=ctx.seasonal_period),
            n_jobs=1,
        )
        predictions = engine.forecast(df=panel, h=horizon)

        model_column = next(
            column for column in predictions.columns if column not in ("unique_id", "ds")
        )

        return predictions.select(
            pl.col("unique_id").alias("series_id"),
            pl.col("ds").cast(pl.Date).alias("period"),
            pl.col("ds").rank("ordinal").over("unique_id").cast(pl.Int32).alias("horizon"),
            pl.col(model_column).alias("forecast"),
        )


@register_forecaster
class AutoETSForecaster(StatsForecastAdapter):
    """Automatic exponential smoothing — error/trend/season form selected per series."""

    name = "auto_ets"

    def build_model(self, ctx: ForecastContext) -> Any:
        return AutoETS(season_length=ctx.seasonal_period)


@register_forecaster
class CrostonForecaster(StatsForecastAdapter):
    """Croston's method for intermittent demand, with optimized smoothing."""

    name = "croston"

    def build_model(self, ctx: ForecastContext) -> Any:
        return CrostonOptimized()


@register_forecaster
class AutoARIMAForecaster(StatsForecastAdapter):
    """Automatic ARIMA. Registered but off by default — slower, rarely the winner
    on short monthly history."""

    name = "auto_arima"

    def build_model(self, ctx: ForecastContext) -> Any:
        return AutoARIMA(season_length=ctx.seasonal_period)
