"""Volume segmentation (Pareto / ABC analysis).

Phase 1's claim is that foundation models earn their place on *long-tail* series
rather than across the board, so the scoreboard needs a volume axis to test that
against. Segments are assigned by cumulative share of total demand: the head
carries most of the volume in few series, the tail carries little volume across
many.
"""

from __future__ import annotations

import polars as pl

SEGMENTS = ("Head", "Mid", "Tail")

# Cumulative sums accumulate floating-point error, so a series sitting exactly on
# a boundary can compute to 0.9500000000000001 and fall a segment lower than the
# thresholds say it should. Compare with a tolerance rather than exactly.
BOUNDARY_TOLERANCE = 1e-9


def assign_volume_segments(
    profiles: pl.DataFrame, head_share: float = 0.80, mid_share: float = 0.95
) -> pl.DataFrame:
    """Label each series Head / Mid / Tail by cumulative share of total volume.

    A series belongs to the head if the cumulative share up to and including it
    stays within `head_share`; the series that crosses a boundary falls into the
    next segment down.
    """
    if profiles.is_empty():
        return profiles.with_columns(pl.lit(None, dtype=pl.Utf8).alias("volume_segment"))

    total = float(profiles["total_volume"].sum())
    if total <= 0:
        # No volume anywhere — segmentation would be arbitrary.
        return profiles.with_columns(pl.lit("Tail", dtype=pl.Utf8).alias("volume_segment"))

    ranked = profiles.sort("total_volume", descending=True).with_columns(
        (pl.col("total_volume").cum_sum() / total).alias("cumulative_share")
    )

    return (
        ranked.with_columns(
            pl.when(pl.col("cumulative_share") <= head_share + BOUNDARY_TOLERANCE)
            .then(pl.lit("Head"))
            .when(pl.col("cumulative_share") <= mid_share + BOUNDARY_TOLERANCE)
            .then(pl.lit("Mid"))
            .otherwise(pl.lit("Tail"))
            .alias("volume_segment")
        )
        .drop("cumulative_share")
        .sort("series_id")
    )


def segment_summary(profiles: pl.DataFrame) -> pl.DataFrame:
    """Series count and volume share per segment — the shape of the Pareto curve."""
    total = float(profiles["total_volume"].sum())
    return (
        profiles.group_by("volume_segment")
        .agg(
            pl.len().alias("series"),
            pl.col("total_volume").sum().alias("volume"),
        )
        .with_columns(
            (pl.col("volume") / total * 100).round(1).alias("volume_pct"),
        )
        .sort("volume", descending=True)
    )
