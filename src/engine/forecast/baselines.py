"""Naive baselines.

Seasonal-naive is the reference every other model is scored against: if the
portfolio cannot beat "same month last year", the added machinery is not
earning its cost.
"""

from __future__ import annotations

import polars as pl

from engine.config.schema import quantile_column
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


def _residual_offsets(history: pl.DataFrame, lookback: int, levels: list[float]) -> pl.DataFrame:
    """Empirical quantiles of the rule's own past errors, per series.

    A naive point forecast becomes a naive *distribution* by adding the spread
    of the errors that same rule has historically made. Without this the
    baseline has no interval, and there is nothing to judge a model's
    uncertainty against — the same trap as scoring accuracy with no baseline.
    """
    residuals = (
        history.sort("series_id", "period")
        .with_columns(
            (pl.col("quantity") - pl.col("quantity").shift(lookback).over("series_id")).alias(
                "residual"
            )
        )
        .drop_nulls("residual")
    )

    return residuals.group_by("series_id").agg(
        [
            pl.col("residual").quantile(level).alias(f"offset_{quantile_column(level)}")
            for level in levels
        ]
    )


def _with_quantiles(
    grid: pl.DataFrame, offsets: pl.DataFrame, levels: list[float], join_on: list[str]
) -> pl.DataFrame:
    if not levels:
        return grid
    joined = grid.join(offsets, on=join_on, how="left")
    return joined.with_columns(
        [
            (pl.col("forecast") + pl.col(f"offset_{quantile_column(level)}").fill_null(0.0)).alias(
                quantile_column(level)
            )
            for level in levels
        ]
    ).drop([f"offset_{quantile_column(level)}" for level in levels])


@register_forecaster
class NaiveForecaster(Forecaster):
    """Repeat the last observed value across the horizon."""

    name = "naive"

    def fit_predict(
        self, history: pl.DataFrame, horizon: int, ctx: ForecastContext
    ) -> pl.DataFrame:
        grid = _horizon_grid(history, horizon, ctx.granularity)
        point = grid.join(_last_values(history), on="series_id", how="left").select(
            "series_id", "period", "horizon", pl.col("last_value").alias("forecast")
        )

        levels = ctx.quantile_levels
        if not levels:
            return point

        # A last-value forecast drifts further the longer the horizon, so the
        # error spread is measured separately at each step.
        offsets = pl.concat(
            [
                _residual_offsets(history, step, levels).with_columns(
                    pl.lit(step, dtype=pl.Int32).alias("horizon")
                )
                for step in range(1, horizon + 1)
            ]
        )
        return _with_quantiles(point, offsets, levels, ["series_id", "horizon"])


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

        point = seasonal.join(_last_values(history), on="series_id", how="left").select(
            "series_id",
            "period",
            "horizon",
            pl.coalesce("seasonal_value", "last_value").alias("forecast"),
        )

        levels = ctx.quantile_levels
        if not levels:
            return point

        # Within one seasonal cycle the error of "same period last cycle" does
        # not widen with the horizon, so a single spread per series suffices.
        offsets = _residual_offsets(history, ctx.seasonal_period, levels)
        return _with_quantiles(point, offsets, levels, ["series_id"])
