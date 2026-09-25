"""What a forecast cost, in the currency of decisions rather than of error.

Accuracy metrics treat over- and under-forecasting as equally bad. Inventory
does not: a stockout and a month of holding are different amounts of pain, and
their ratio is what decides how much to keep. This module scores each model on
that instead.

The reduction is a per-period newsvendor. At each origin the model's quantile at
the critical ratio is the stock level; the actual is what showed up; holding cost
falls on the excess and shortage cost on the shortfall. Inventory carried between
periods is deliberately not simulated — doing it needs an opening stock, an order
calendar and receipt lags, none of which exist in the data, all of which would be
invented. Because the carryover a model would face is the same whichever model
produced the forecast, the comparison between models stays fair, which is the
number worth having.

One consequence is worth stating: under this reduction perfect foresight costs
exactly nothing, so a model's cost *is* its regret against a perfect forecast.
"""

from __future__ import annotations

import polars as pl

from engine.config.schema import CostModelConfig

GLOBAL_SCOPE = "__all__"
METRIC_COLUMNS = ["model", "scope_type", "scope_value", "metric", "value", "n_obs"]


def with_decision_costs(scored: pl.DataFrame, cost_model: CostModelConfig) -> pl.DataFrame:
    """Add per-row holding, shortage and total cost.

    `scored` is backtest quantiles at one level, each row carrying the actual for
    the period the decision was made for.
    """
    excess = (pl.col("value") - pl.col("actual")).clip(0)
    shortfall = (pl.col("actual") - pl.col("value")).clip(0)
    return scored.with_columns(
        excess.alias("units_held"),
        shortfall.alias("units_short"),
        (excess * cost_model.holding_cost_per_unit).alias("holding_cost"),
        (shortfall * cost_model.shortage_cost_per_unit).alias("shortage_cost"),
    ).with_columns((pl.col("holding_cost") + pl.col("shortage_cost")).alias("decision_cost"))


def aggregate(costed: pl.DataFrame, group_by: list[str]) -> pl.DataFrame:
    """Total the decision costs, plus two different service measures.

    `cycle_service_level` is the share of periods in which stock covered demand.
    That is what stocking to a quantile promises, so it is the calibration check:
    stock to the 90th percentile and 90% of periods should not stock out.

    `fill_rate` is the share of demanded *units* that were served. It is always
    the higher of the two and is not a calibration check — in a period that does
    stock out you still serve most of the units, so a well-calibrated 90th
    percentile typically delivers a fill rate in the high nineties. It is here
    because it is the number a planner quotes, not because it validates anything.
    """
    return (
        costed.group_by(group_by)
        .agg(
            pl.col("decision_cost").sum().alias("decision_cost"),
            pl.col("holding_cost").sum().alias("holding_cost"),
            pl.col("shortage_cost").sum().alias("shortage_cost"),
            pl.col("units_held").sum().alias("units_held"),
            pl.col("units_short").sum().alias("units_short"),
            pl.col("actual").sum().alias("total_demand"),
            (pl.col("actual") <= pl.col("value")).mean().alias("cycle_service_level"),
            pl.len().alias("n_obs"),
        )
        .with_columns(
            pl.when(pl.col("total_demand") > 0)
            .then(1 - pl.col("units_short") / pl.col("total_demand"))
            .otherwise(None)
            .alias("fill_rate"),
            pl.when(pl.col("total_demand") > 0)
            .then(pl.col("decision_cost") / pl.col("total_demand"))
            .otherwise(None)
            .alias("cost_per_unit_demand"),
        )
        .sort("decision_cost")
    )


def add_decision_value_add(
    summary: pl.DataFrame, baseline: str, scope_column: str | None = None
) -> pl.DataFrame:
    """Percentage of the baseline's decision cost avoided, within each scope.

    The decision counterpart of Forecast Value Add, and the number that answers
    "what is the better forecast worth" — in cost, not in error.

    `scope_column` matters: a segmented summary holds one baseline row per
    segment, and Head items must be compared against the baseline's Head cost
    rather than against its total.
    """
    if baseline not in summary["model"].to_list():
        return summary.with_columns(pl.lit(None, dtype=pl.Float64).alias("dva_pct"))

    join_keys = [scope_column] if scope_column else []
    reference = summary.filter(pl.col("model") == baseline).select(
        *join_keys, pl.col("decision_cost").alias("baseline_cost")
    )
    joined = (
        summary.join(reference, on=join_keys, how="left")
        if join_keys
        else summary.join(reference, how="cross")
    )

    return joined.with_columns(
        # A baseline that costs nothing cannot be improved on, and dividing by it
        # would report an infinite win.
        pl.when(pl.col("baseline_cost") > 0)
        .then(100 * (pl.col("baseline_cost") - pl.col("decision_cost")) / pl.col("baseline_cost"))
        .otherwise(None)
        .alias("dva_pct")
    ).drop("baseline_cost")


def to_long(summary: pl.DataFrame, scope_type: str, scope_column: str | None) -> pl.DataFrame:
    """Reshape a summary into the store's one-row-per-metric form."""
    metrics = [
        "decision_cost",
        "holding_cost",
        "shortage_cost",
        "units_held",
        "units_short",
        "cycle_service_level",
        "fill_rate",
        "cost_per_unit_demand",
        "dva_pct",
    ]
    present = [name for name in metrics if name in summary.columns]
    scope = (pl.col(scope_column).cast(pl.Utf8) if scope_column else pl.lit(GLOBAL_SCOPE)).alias(
        "scope_value"
    )

    return (
        summary.with_columns(pl.lit(scope_type).alias("scope_type"), scope)
        .unpivot(
            index=["model", "scope_type", "scope_value", "n_obs"],
            on=present,
            variable_name="metric",
            value_name="value",
        )
        .select(METRIC_COLUMNS)
    )
