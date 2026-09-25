"""Stock levels from a forecast distribution.

The engine already produces quantiles; this turns them into the three numbers a
reorder screen needs. Nothing here estimates a standard deviation or assumes a
distribution shape — the quantile *is* the answer to "how much covers demand
this often", which is the whole reason for forecasting one.
"""

from __future__ import annotations

import polars as pl

from engine.config.schema import CostModelConfig

POLICY_COLUMNS = [
    "series_id",
    "model",
    "service_level",
    "window_periods",
    "expected_demand",
    "target_stock",
    "safety_stock",
    "reorder_point",
]


def _window_sum(frame: pl.DataFrame, horizons: int, column: str, alias: str) -> pl.DataFrame:
    if horizons <= 0:
        # A zero-period window covers nothing. Real for a lead time of zero:
        # replenishment is instant, so nothing has to be held to bridge it.
        return frame.select("series_id", "model").unique().with_columns(pl.lit(0.0).alias(alias))
    return (
        frame.filter(pl.col("horizon") <= horizons)
        .group_by("series_id", "model")
        .agg(pl.col(column).sum().alias(alias))
    )


def build_policy(
    forecasts: pl.DataFrame, quantiles: pl.DataFrame, cost_model: CostModelConfig
) -> pl.DataFrame:
    """Per-series stock levels for one forecast run.

    `target_stock` is what to order up to at each review; it covers lead time
    plus review period. `reorder_point` covers lead time alone, which is the
    trigger level a continuous-review policy needs. Under periodic review only
    the first is used; both are reported because which policy a client runs is
    the client's business, not the engine's.

    Summing per-period quantiles overstates the quantile of the summed demand —
    for a service level above 50% the sum of quantiles exceeds the quantile of
    the sum — so `target_stock` errs toward holding too much rather than too
    little. Correcting it properly needs a joint distribution across horizons,
    which the models do not produce.
    """
    level = cost_model.service_level
    window = cost_model.protection_window

    at_level = quantiles.filter((pl.col("level") - level).abs() < 1e-9)
    if at_level.is_empty():
        raise ValueError(
            f"No quantiles stored at level {level:.4g}, which the cost model implies. "
            "Re-run the forecast after adding that level to probabilistic.levels."
        )

    expected = _window_sum(forecasts, window, "forecast", "expected_demand")
    target = _window_sum(at_level, window, "value", "target_stock")
    reorder = _window_sum(at_level, cost_model.lead_time_periods, "value", "reorder_point")

    return (
        expected.join(target, on=["series_id", "model"], how="inner")
        .join(reorder, on=["series_id", "model"], how="left")
        .with_columns(
            pl.lit(level).alias("service_level"),
            pl.lit(window, dtype=pl.Int32).alias("window_periods"),
            # Clipped at zero: a model whose upper quantile falls below its own
            # point forecast is miscalibrated, and a negative safety stock is not
            # an instruction anyone can act on.
            (pl.col("target_stock") - pl.col("expected_demand")).clip(0).alias("safety_stock"),
            pl.col("reorder_point").fill_null(0.0),
        )
        .select(POLICY_COLUMNS)
        .sort("series_id", "model")
    )


def coverage_gap(forecasts: pl.DataFrame, policy: pl.DataFrame) -> pl.DataFrame:
    """Series that got a forecast but cannot get a stock policy, by model.

    Arbitration picks a model on accuracy alone. Some models produce no
    quantiles, and a series whose champion is one of them has no distribution to
    stock against — so it silently vanishes from the reorder screen. Naming them
    is the point: an item with no policy is an item nobody reorders.
    """
    covered = policy["series_id"].unique().to_list()
    return (
        forecasts.filter(~pl.col("series_id").is_in(covered))
        .group_by("model")
        .agg(pl.col("series_id").n_unique().alias("series"))
        .sort("series", descending=True)
    )
