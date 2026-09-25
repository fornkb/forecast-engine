"""Conformal correction of forecast quantiles.

A model's 90th percentile is a claim: demand should fall below it nine times in
ten. Measured on held-out origins, the models here miss that claim by a few
points, in both directions — and they miss it for reasons that cannot be fixed
in the models. The naive family estimates its interval from roughly two dozen
residuals per series, where the empirical upper quantile is biased low. The
gradient-boosted model winsorizes its training target, clipping the very tail
its upper quantile is trying to find. Chronos-2's quantiles come out of
pretrained weights.

So the correction is applied afterwards and identically to all of them: measure
how far each model's quantile actually sat from the demand it was meant to cover,
and shift it by that much. If a model is already calibrated the shift is zero and
nothing changes.

The offset is measured on the backtest's earlier origins and never on its last
one, which stays an untouched holdout — the same split arbitration uses, for the
same reason.

Offsets are expressed as a multiple of each series' mean demand rather than in
units. Demand here spans four orders of magnitude, so a single unit offset would
swamp the small series and vanish on the large ones. Dividing by the series' own
level also keeps the correction defined for intermittent series, where a
multiplicative correction on a zero forecast has nothing to multiply.
"""

from __future__ import annotations

import polars as pl

from engine.config.schema import quantile_column

OFFSET_COLUMNS = ["model", "level", "scaled_offset", "n_obs"]
# A series whose mean demand is this small has no meaningful level to scale by;
# its rows cannot inform an offset without dominating it.
MIN_MEAN_DEMAND = 1e-9


def fit_offsets(
    scored: pl.DataFrame, series_level: pl.DataFrame, exclude_origin: int | None = None
) -> pl.DataFrame:
    """Per model and level, how far the quantile has to move to keep its promise.

    `scored` is backtest quantiles carrying the actual for each period. The
    conformity score is (actual - quantile) / mean_demand; the offset is that
    score's own quantile at the same level. A model covering demand exactly as
    often as it claims scores an offset of zero.

    `exclude_origin` drops the holdout origin so the correction is never fitted
    on the data it will be judged against.
    """
    frame = scored
    if exclude_origin is not None:
        frame = frame.filter(pl.col("origin") != exclude_origin)
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "model": pl.Utf8,
                "level": pl.Float64,
                "scaled_offset": pl.Float64,
                "n_obs": pl.UInt32,
            }
        )

    joined = frame.join(
        series_level.select("series_id", "mean_demand"), on="series_id", how="inner"
    ).filter(pl.col("mean_demand") > MIN_MEAN_DEMAND)

    return (
        joined.with_columns(
            ((pl.col("actual") - pl.col("value")) / pl.col("mean_demand")).alias("score")
        )
        .group_by("model", "level")
        .agg(
            # The level's own quantile of the score: the amount that would have
            # made this quantile cover demand exactly as often as it promises.
            pl.map_groups(
                exprs=[pl.col("score"), pl.col("level").first()],
                function=lambda s: s[0].quantile(s[1][0]),
                returns_scalar=True,
            ).alias("scaled_offset"),
            pl.len().alias("n_obs"),
        )
        .with_columns(pl.col("scaled_offset").fill_null(0.0))
        .select(OFFSET_COLUMNS)
        .sort("model", "level")
    )


def apply_offsets(
    quantiles: pl.DataFrame, offsets: pl.DataFrame, series_level: pl.DataFrame
) -> pl.DataFrame:
    """Shift each quantile by its model's offset, scaled to the series' level.

    A model or level with no offset is left exactly as it was, so a partially
    calibrated portfolio is not silently half-corrected in units nobody chose.
    Corrected quantiles are clipped at zero: demand is not negative.
    """
    if offsets.is_empty():
        return quantiles

    return (
        quantiles.join(
            offsets.select("model", "level", "scaled_offset"), on=["model", "level"], how="left"
        )
        .join(series_level.select("series_id", "mean_demand"), on="series_id", how="left")
        .with_columns(
            (
                pl.col("value")
                + pl.col("scaled_offset").fill_null(0.0) * pl.col("mean_demand").fill_null(0.0)
            )
            .clip(0)
            .alias("value")
        )
        .drop("scaled_offset", "mean_demand")
    )


def coverage(scored: pl.DataFrame, group_by: list[str]) -> pl.DataFrame:
    """Share of periods the quantile actually covered, per group.

    The number a quantile's level is a promise about, and the only way to tell
    whether a correction helped.
    """
    return (
        scored.group_by(group_by)
        .agg(
            (pl.col("actual") <= pl.col("value")).mean().alias("coverage"),
            pl.len().alias("n_obs"),
        )
        .sort(group_by)
    )


def apply_to_wide(
    frame: pl.DataFrame,
    offsets: pl.DataFrame,
    series_level: pl.DataFrame,
    levels: list[float],
) -> pl.DataFrame:
    """The same correction, applied to the per-level columns models emit.

    The long form above is what evaluation uses; a forward forecast carries one
    column per level, and correcting it here keeps every later consumer — the
    export, the store, the stock policy — reading already-corrected numbers
    rather than each having to remember to ask.
    """
    if offsets.is_empty():
        return frame

    out = frame.join(
        series_level.select("series_id", "mean_demand"), on="series_id", how="left"
    ).with_columns(pl.col("mean_demand").fill_null(0.0))

    for level in levels:
        column = quantile_column(level)
        if column not in out.columns:
            continue
        at_level = offsets.filter((pl.col("level") - level).abs() < 1e-9).select(
            "model", pl.col("scaled_offset").alias("_offset")
        )
        if at_level.is_empty():
            continue
        out = (
            out.join(at_level, on="model", how="left")
            .with_columns(
                # Null offset means this model was never calibrated; leaving the
                # quantile alone is better than shifting it by an invented amount.
                pl.when(pl.col(column).is_null())
                .then(pl.col(column))
                .otherwise(
                    (
                        pl.col(column) + pl.col("_offset").fill_null(0.0) * pl.col("mean_demand")
                    ).clip(0)
                )
                .alias(column)
            )
            .drop("_offset")
        )

    return out.drop("mean_demand")
