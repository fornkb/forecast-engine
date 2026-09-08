"""Per-series model arbitration.

Selection is shrunk toward a champion model rather than taken at face value: with
only a handful of scored observations per series, the lowest per-series error is
substantially noise, and unconstrained per-series picking scores worse
out-of-sample than simply using the best overall model. A series therefore
switches away from the champion only when its own backtest shows a challenger
winning by a margin.

Blending is deliberately not done here — it is only worth adding once it
demonstrably beats the winning single model, which is a Phase 1 question.
"""

from __future__ import annotations

import polars as pl


def select_per_series(
    series_scores: pl.DataFrame,
    metric: str,
    champion: str,
    min_improvement: float = 0.0,
) -> pl.DataFrame:
    """Choose a model per series, defaulting to `champion`.

    `series_scores` has columns series_id, model, score (lower is better).
    `min_improvement` is the fraction by which a challenger must beat the
    champion on that series before it is selected (0.1 = 10% lower error).
    Ties break on model name so reruns cannot silently change the selection.
    """
    champion_scores = series_scores.filter(pl.col("model") == champion).select(
        "series_id", pl.col("score").alias("champion_score")
    )

    best = (
        series_scores.sort("series_id", "score", "model", nulls_last=True)
        .group_by("series_id", maintain_order=True)
        .first()
        .rename({"model": "challenger", "score": "challenger_score"})
    )

    combined = best.join(champion_scores, on="series_id", how="left")

    switches = pl.col("champion_score").is_not_null() & (
        pl.col("challenger_score") <= pl.col("champion_score") * (1 - min_improvement)
    )

    return combined.select(
        "series_id",
        pl.when(switches).then(pl.col("challenger")).otherwise(pl.lit(champion)).alias("model"),
        pl.when(switches)
        .then(pl.col("challenger_score"))
        .otherwise(pl.col("champion_score"))
        .alias("score"),
        pl.lit(metric).alias("metric"),
    )


def pick_champion(overall_scores: pl.DataFrame, metric_column: str = "mae") -> str:
    """The single best model across all series — the default every series inherits."""
    return overall_scores.sort(metric_column, "model").select("model").head(1).item()


def apply_selection(forecasts: pl.DataFrame, selection: pl.DataFrame) -> pl.DataFrame:
    """Keep only each series' selected model from a multi-model forecast frame."""
    return forecasts.join(
        selection.select("series_id", "model"), on=["series_id", "model"], how="inner"
    ).sort("series_id", "period")


def selection_summary(selection: pl.DataFrame) -> pl.DataFrame:
    """How often each model was chosen — the portfolio's composition at a glance."""
    total = selection.height
    return (
        selection.group_by("model")
        .len()
        .rename({"len": "series_won"})
        .with_columns((pl.col("series_won") / total * 100).round(1).alias("share_pct"))
        .sort("series_won", descending=True)
    )
