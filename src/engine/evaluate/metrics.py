"""Accuracy metrics and Forecast Value Add.

Metrics are reported in native units (MAE, WAPE) rather than percentage errors:
roughly a tenth of this panel's observations are true zeros, and MAPE-family
metrics are undefined or explosive there.
"""

from __future__ import annotations

import polars as pl

from engine.config.schema import quantile_column

ERROR_METRICS = ("mae", "rmse", "wape", "bias")


def with_errors(predictions: pl.DataFrame) -> pl.DataFrame:
    return predictions.with_columns(
        (pl.col("forecast") - pl.col("actual")).alias("error"),
        (pl.col("forecast") - pl.col("actual")).abs().alias("abs_error"),
        ((pl.col("forecast") - pl.col("actual")) ** 2).alias("sq_error"),
    )


def aggregate(predictions: pl.DataFrame, group_by: list[str]) -> pl.DataFrame:
    """Wide metric frame: one row per group with every error metric."""
    scored = with_errors(predictions)
    return scored.group_by(group_by).agg(
        pl.col("abs_error").mean().alias("mae"),
        pl.col("sq_error").mean().sqrt().alias("rmse"),
        pl.when(pl.col("actual").sum() > 0)
        .then(pl.col("abs_error").sum() / pl.col("actual").sum())
        .otherwise(None)
        .alias("wape"),
        pl.col("error").mean().alias("bias"),
        pl.len().alias("n_obs"),
    )


def aggregate_series_level(predictions: pl.DataFrame, group_by: list[str]) -> pl.DataFrame:
    """Median of per-series WAPE — a scale-free companion to the aggregate metrics.

    Aggregate MAE and WAPE are volume-weighted, so on this panel a handful of
    very large series decide a whole slice's number. Both views are reported:
    the weighted one for business impact, this one for per-series model quality.
    """
    per_series = (
        with_errors(predictions)
        .group_by([*group_by, "series_id"])
        .agg(
            pl.col("abs_error").sum().alias("abs_error_sum"),
            pl.col("actual").sum().alias("actual_sum"),
        )
        .filter(pl.col("actual_sum") > 0)
        .with_columns((pl.col("abs_error_sum") / pl.col("actual_sum")).alias("series_wape"))
    )
    return per_series.group_by(group_by).agg(
        pl.col("series_wape").median().alias("median_series_wape")
    )


def pinball_expr(level: float) -> pl.Expr:
    """Quantile (pinball) loss for one level.

    Under-forecasting a high quantile is penalised more heavily than
    over-forecasting it, which is what makes this the right loss for a service
    level: running out is not symmetric with holding a little extra.
    """
    column = quantile_column(level)
    return (
        pl.when(pl.col("actual") >= pl.col(column))
        .then(level * (pl.col("actual") - pl.col(column)))
        .otherwise((1 - level) * (pl.col(column) - pl.col("actual")))
    )


def aggregate_probabilistic(
    predictions: pl.DataFrame, group_by: list[str], levels: list[float]
) -> pl.DataFrame | None:
    """Pinball loss per level, its mean, and empirical interval coverage.

    Returns None when the frame carries no quantiles — a model that only emits a
    point forecast is scored on point accuracy alone rather than penalised with
    a placeholder.
    """
    available = [level for level in levels if quantile_column(level) in predictions.columns]
    if not available:
        return None

    scored = predictions.filter(
        pl.any_horizontal([pl.col(quantile_column(level)).is_not_null() for level in available])
    )
    if scored.is_empty():
        return None

    exprs = [
        pinball_expr(level).mean().alias(f"pinball_{quantile_column(level)}") for level in available
    ]

    # The mean pinball loss across levels approximates CRPS, which is the
    # single number for "is this whole distribution any good".
    exprs.append(
        pl.mean_horizontal([pinball_expr(level) for level in available])
        .mean()
        .alias("mean_pinball")
    )

    if len(available) >= 2:
        low, high = quantile_column(min(available)), quantile_column(max(available))
        nominal = max(available) - min(available)
        exprs.append(
            ((pl.col("actual") >= pl.col(low)) & (pl.col("actual") <= pl.col(high)))
            .mean()
            .alias("interval_coverage")
        )
        exprs.append(pl.lit(nominal).alias("nominal_coverage"))

    return scored.group_by(group_by).agg(exprs)


def to_long(metrics: pl.DataFrame, scope_type: str, scope_column: str | None) -> pl.DataFrame:
    """Reshape a wide metric frame into the store's long metric format."""
    scope_value = (
        pl.col(scope_column).cast(pl.Utf8) if scope_column else pl.lit("all", dtype=pl.Utf8)
    )
    value_columns = [c for c in metrics.columns if c not in ("model", "n_obs", scope_column)]

    return (
        metrics.with_columns(scope_value.alias("scope_value"))
        .unpivot(
            index=["model", "scope_value", "n_obs"],
            on=value_columns,
            variable_name="metric",
            value_name="value",
        )
        .select(
            "model",
            pl.lit(scope_type).alias("scope_type"),
            "scope_value",
            "metric",
            "value",
            "n_obs",
        )
    )


def add_forecast_value_add(
    metrics: pl.DataFrame, baseline_model: str, scope_column: str | None
) -> pl.DataFrame:
    """Percentage error reduction against the baseline, within each scope.

    Positive FVA means the model beat the baseline; negative means the added
    machinery is making forecasts worse than "same period last cycle".
    """
    join_keys = [scope_column] if scope_column else []
    baseline = metrics.filter(pl.col("model") == baseline_model).select(
        *join_keys,
        pl.col("mae").alias("baseline_mae"),
        pl.col("wape").alias("baseline_wape"),
    )

    joined = (
        metrics.join(baseline, on=join_keys, how="left")
        if join_keys
        else metrics.join(baseline, how="cross")
    )

    result = joined.with_columns(
        pl.when(pl.col("baseline_mae") > 0)
        .then((1 - pl.col("mae") / pl.col("baseline_mae")) * 100)
        .otherwise(None)
        .alias("fva_mae_pct"),
        pl.when(pl.col("baseline_wape") > 0)
        .then((1 - pl.col("wape") / pl.col("baseline_wape")) * 100)
        .otherwise(None)
        .alias("fva_wape_pct"),
    ).drop("baseline_mae", "baseline_wape")

    if "median_series_wape" not in metrics.columns:
        return result

    baseline_median = metrics.filter(pl.col("model") == baseline_model).select(
        *join_keys, pl.col("median_series_wape").alias("baseline_median_wape")
    )
    result = (
        result.join(baseline_median, on=join_keys, how="left")
        if join_keys
        else result.join(baseline_median, how="cross")
    )
    return result.with_columns(
        pl.when(pl.col("baseline_median_wape") > 0)
        .then((1 - pl.col("median_series_wape") / pl.col("baseline_median_wape")) * 100)
        .otherwise(None)
        .alias("fva_median_wape_pct")
    ).drop("baseline_median_wape")
