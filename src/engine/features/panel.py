"""Supervised panel construction for the global (cross-series) model.

Features are always computed "as of" a period t and the target is taken at t+h,
so no row can see its own future. Every level-valued feature is divided by the
series' expanding mean, because demand here spans four orders of magnitude
across series — a pooled model trained on raw units would simply learn to rank
series by size.
"""

from __future__ import annotations

import polars as pl

from engine.config.schema import Granularity
from engine.features.calendar import CALENDAR_FEATURES, add_calendar_features, offset_expr

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 12)
DEFAULT_WINDOWS: tuple[int, ...] = (3, 6, 12)


def feature_columns(
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    static_features: tuple[str, ...] = (),
) -> list[str]:
    """Ordered feature list — the contract between panel building and training."""
    names = [f"s_lag_{lag}" for lag in lags]
    for window in windows:
        names += [f"s_roll_mean_{window}", f"s_roll_std_{window}", f"zero_frac_{window}"]
    names += ["horizon", "periods_observed", *CALENDAR_FEATURES]
    names += [f"{col}_code" for col in static_features]
    return names


def add_history_features(
    history: pl.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pl.DataFrame:
    """Attach lag, rolling, and scale columns computed as of each row's period."""
    frame = history.sort("series_id", "period")

    level_columns: list[str] = []
    exprs: list[pl.Expr] = []

    for lag in lags:
        name = f"lag_{lag}"
        level_columns.append(name)
        exprs.append(pl.col("quantity").shift(lag).over("series_id").alias(name))

    for window in windows:
        mean_name, std_name = f"roll_mean_{window}", f"roll_std_{window}"
        level_columns += [mean_name, std_name]
        exprs += [
            pl.col("quantity")
            .rolling_mean(window_size=window, min_samples=1)
            .over("series_id")
            .alias(mean_name),
            pl.col("quantity")
            .rolling_std(window_size=window, min_samples=2)
            .over("series_id")
            .alias(std_name),
            (pl.col("quantity") == 0)
            .cast(pl.Float64)
            .rolling_mean(window_size=window, min_samples=1)
            .over("series_id")
            .alias(f"zero_frac_{window}"),
        ]

    frame = frame.with_columns(exprs)

    frame = frame.with_columns(
        pl.col("quantity").cum_count().over("series_id").cast(pl.Int32).alias("periods_observed"),
        (
            pl.col("quantity").cum_sum().over("series_id")
            / pl.col("quantity").cum_count().over("series_id")
        ).alias("scale_raw"),
    )
    frame = frame.with_columns(
        pl.when(pl.col("scale_raw") > 0).then(pl.col("scale_raw")).otherwise(1.0).alias("scale")
    )

    return frame.with_columns(
        [(pl.col(col) / pl.col("scale")).alias(f"s_{col}") for col in level_columns]
    )


def build_training_frame(
    base: pl.DataFrame, horizon: int, granularity: Granularity
) -> pl.DataFrame:
    """One row per (series, origin period t, horizon h) with target taken at t+h."""
    frames = []
    for step in range(1, horizon + 1):
        stepped = base.with_columns(
            pl.col("quantity").shift(-step).over("series_id").alias("target"),
            pl.col("period").shift(-step).over("series_id").alias("target_period"),
            pl.lit(step, dtype=pl.Int32).alias("horizon"),
        )
        # A row `step` positions ahead is only a valid target if it is also
        # `step` periods ahead in calendar time — guards against gaps.
        frames.append(
            stepped.filter(
                pl.col("target").is_not_null()
                & (pl.col("target_period") == offset_expr("period", step, granularity))
            )
        )

    panel = pl.concat(frames)
    panel = panel.with_columns((pl.col("target") / pl.col("scale")).alias("target_scaled"))
    return add_calendar_features(panel, "target_period", granularity)


def build_inference_frame(
    base: pl.DataFrame, horizon: int, granularity: Granularity
) -> pl.DataFrame:
    """One row per (series, horizon) anchored on each series' last observed period."""
    last_rows = base.filter(pl.col("period") == pl.col("period").max().over("series_id"))

    frames = [
        last_rows.with_columns(
            offset_expr("period", step, granularity).alias("target_period"),
            pl.lit(step, dtype=pl.Int32).alias("horizon"),
        )
        for step in range(1, horizon + 1)
    ]
    panel = pl.concat(frames)
    return add_calendar_features(panel, "target_period", granularity)


def encode_static_features(dim: pl.DataFrame, static_features: tuple[str, ...]) -> pl.DataFrame:
    """Stable integer codes for categorical series attributes.

    Derived once from the series dimension so training and inference frames share
    an encoding regardless of which rows each contains.
    """
    if not static_features:
        return dim.select("series_id")

    encoded = dim.select("series_id", *static_features)
    for column in static_features:
        levels = encoded[column].unique().sort().to_list()
        lookup = pl.DataFrame(
            {column: levels, f"{column}_code": list(range(len(levels)))},
            schema={column: pl.Utf8, f"{column}_code": pl.Int32},
        )
        encoded = encoded.join(lookup, on=column, how="left")

    return encoded.select("series_id", *[f"{col}_code" for col in static_features])
