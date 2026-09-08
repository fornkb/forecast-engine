"""Naive baselines.

Seasonal-naive is the reference every other model is scored against: if the
portfolio cannot beat "same month last year", the added machinery is not
earning its cost.
"""

from __future__ import annotations

import polars as pl

from engine.features.calendar import offset_expr
from engine.forecast.base import ForecastContext, Forecaster, register_forecaster


def _horizon_grid(history: pl.DataFrame, horizon: int, granularity: str) -> pl.DataFrame:
    """One row per (series, horizon) with the period being forecast."""
    cutoffs = history.group_by("series_id").agg(pl.col("period").max().alias("cutoff"))
    frames = [
        cutoffs.with_columns(
            offset_expr("cutoff", step, granularity).alias("period"),
            pl.lit(step, dtype=pl.Int32).alias("horizon"),
        )
        for step in range(1, horizon + 1)
    ]
    return pl.concat(frames)


def _last_values(history: pl.DataFrame) -> pl.DataFrame:
    return (
        history.sort("series_id", "period")
        .group_by("series_id")
        .agg(pl.col("quantity").last().alias("last_value"))
    )


@register_forecaster
class NaiveForecaster(Forecaster):
    """Repeat the last observed value across the horizon."""

    name = "naive"

    def fit_predict(
        self, history: pl.DataFrame, horizon: int, ctx: ForecastContext
    ) -> pl.DataFrame:
        grid = _horizon_grid(history, horizon, ctx.granularity)
        return grid.join(_last_values(history), on="series_id", how="left").select(
            "series_id", "period", "horizon", pl.col("last_value").alias("forecast")
        )


@register_forecaster
class SeasonalNaiveForecaster(Forecaster):
    """Repeat the value observed one full seasonal cycle earlier.

    Falls back to the last observed value when a series is shorter than one
    cycle, which keeps short and new series scoreable rather than null.
    """

    name = "seasonal_naive"

    def fit_predict(
        self, history: pl.DataFrame, horizon: int, ctx: ForecastContext
    ) -> pl.DataFrame:
        grid = _horizon_grid(history, horizon, ctx.granularity).with_columns(
            offset_expr("period", -ctx.seasonal_period, ctx.granularity).alias("lookup_period")
        )

        seasonal = grid.join(
            history.select(
                "series_id",
                pl.col("period").alias("lookup_period"),
                pl.col("quantity").alias("seasonal_value"),
            ),
            on=["series_id", "lookup_period"],
            how="left",
        )

        return seasonal.join(_last_values(history), on="series_id", how="left").select(
            "series_id",
            "period",
            "horizon",
            pl.coalesce("seasonal_value", "last_value").alias("forecast"),
        )
