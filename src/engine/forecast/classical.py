"""Classical per-series forecasters, via statsforecast's Numba-compiled models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, CrostonOptimized, SeasonalNaive
from statsforecast.utils import ConformalIntervals

from engine.config.schema import Granularity, quantile_column
from engine.forecast.base import ForecastContext, Forecaster, register_forecaster

_POLARS_FREQ: dict[Granularity, str] = {"monthly": "1mo", "weekly": "1w", "daily": "1d"}

# Windows the conformal interval calibrates on. Each one costs a refit, and each
# one also holds back `horizon` periods of history — so more windows means a
# steadier interval fitted on less data. Two is statsforecast's own default and
# the most a 36-month panel can spare at a 3-month horizon.
CONFORMAL_WINDOWS = 2


def interval_levels(quantile_levels: list[float]) -> tuple[list[int], dict[float, str]]:
    """Which statsforecast intervals to ask for, and how to read each quantile off them.

    statsforecast returns symmetric intervals: one `level` of 80 yields both the
    10th and the 90th percentile. So the quantiles a client wants are matched into
    mirror pairs around the median, and each pair becomes one requested level.

    A level with no mirror — 0.95 without 0.05 — cannot be read off a symmetric
    interval and is left out. Its column then stays absent, which `conform` turns
    into nulls, exactly as for a model that expresses no uncertainty at all.
    """
    columns: dict[float, str] = {}
    levels: set[int] = set()

    for level in quantile_levels:
        if level == 0.5:
            continue
        mirror = round(1.0 - level, 10)
        if mirror not in quantile_levels:
            continue
        spread = abs(1.0 - 2 * min(level, mirror)) * 100
        if abs(spread - round(spread)) > 1e-9:
            # statsforecast names its columns with the level as written; a
            # fractional one would not round-trip reliably.
            continue
        as_int = int(round(spread))
        levels.add(as_int)
        columns[level] = f"{'lo' if level < 0.5 else 'hi'}-{as_int}"

    return sorted(levels), columns


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

        # Conformal rather than each model's own analytic interval: these are
        # distribution-free, they come out the same way for every model here, and
        # Croston has no analytic interval to offer in the first place. Without
        # this these models produce a point forecast only — and a series whose
        # champion produces no distribution has no stock level, so it never
        # reaches a reorder screen at all.
        levels, quantile_sources = interval_levels(ctx.quantile_levels)
        interval_args: dict[str, Any] = {}
        if levels:
            interval_args = {
                "level": levels,
                "prediction_intervals": ConformalIntervals(h=horizon, n_windows=CONFORMAL_WINDOWS),
            }

        predictions = engine.forecast(df=panel, h=horizon, **interval_args)

        # The point forecast is the column named for the model alone; the interval
        # bounds are that name with a -lo-/-hi- suffix.
        model_column = next(
            column
            for column in predictions.columns
            if column not in ("unique_id", "ds") and "-lo-" not in column and "-hi-" not in column
        )

        outputs = [pl.col(model_column).alias("forecast")]
        for level in ctx.quantile_levels:
            if level == 0.5:
                # The median is the point forecast for these models; a separate
                # conformal band around it would only restate it less precisely.
                source = pl.col(model_column)
            elif level in quantile_sources:
                source = pl.col(f"{model_column}-{quantile_sources[level]}")
            else:
                continue
            outputs.append(source.alias(quantile_column(level)))

        return predictions.select(
            pl.col("unique_id").alias("series_id"),
            pl.col("ds").cast(pl.Date).alias("period"),
            pl.col("ds").rank("ordinal").over("unique_id").cast(pl.Int32).alias("horizon"),
            *outputs,
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
