"""Per-series model arbitration.

Selection is shrunk toward a champion rather than taken at face value: with only
a handful of scored observations per series, the lowest per-series error is
substantially noise, and unconstrained per-series picking scores worse
out-of-sample than simply using the best overall model.

The champion is chosen per stratum, not globally. Once a portfolio contains
models with genuinely complementary strengths — a pooled learner that wins on
high-volume series with long history, a foundation model that wins on the long
tail — a single global default is wrong for one group or the other by
construction. Pooling within a stratum keeps the estimate reliable (many series,
not one) while still letting the default differ where the data says it should.

Blending is deliberately not done here — it is only worth adding once it
demonstrably beats the winning single model.
"""

from __future__ import annotations

import polars as pl

GLOBAL_STRATUM = "__all__"


def pick_champions(stratum_scores: pl.DataFrame, metric_column: str = "mae") -> dict[str, str]:
    """Best model within each stratum.

    `stratum_scores` has columns stratum, model, and the metric column (lower is
    better). Ties break on model name so reruns cannot silently change the
    selection.
    """
    winners = (
        stratum_scores.sort("stratum", metric_column, "model", nulls_last=True)
        .group_by("stratum", maintain_order=True)
        .first()
    )
    return {row["stratum"]: row["model"] for row in winners.iter_rows(named=True)}


def select_per_series(
    series_scores: pl.DataFrame,
    metric: str,
    champions: dict[str, str],
    strata: pl.DataFrame | None = None,
    min_improvement: float = 0.0,
) -> pl.DataFrame:
    """Choose a model per series, defaulting to that series' stratum champion.

    `series_scores` has columns series_id, model, score (lower is better).
    `strata` maps series_id to stratum; when omitted every series falls under
    `GLOBAL_STRATUM`. `min_improvement` is the fraction by which a challenger
    must beat the champion on that series before it is selected (0.1 = 10%
    lower error).
    """
    champion_lookup = pl.DataFrame(
        {"stratum": list(champions), "champion": [champions[k] for k in champions]},
        schema={"stratum": pl.Utf8, "champion": pl.Utf8},
    )

    if strata is None:
        series_champion = (
            series_scores.select("series_id")
            .unique()
            .with_columns(pl.lit(GLOBAL_STRATUM).alias("stratum"))
        )
    else:
        series_champion = strata.select("series_id", "stratum")

    series_champion = series_champion.join(champion_lookup, on="stratum", how="left").select(
        "series_id", "champion"
    )

    champion_scores = (
        series_scores.join(series_champion, on="series_id", how="inner")
        .filter(pl.col("model") == pl.col("champion"))
        .select("series_id", "champion", pl.col("score").alias("champion_score"))
    )

    best = (
        series_scores.sort("series_id", "score", "model", nulls_last=True)
        .group_by("series_id", maintain_order=True)
        .first()
        .rename({"model": "challenger", "score": "challenger_score"})
    )

    combined = best.join(champion_scores, on="series_id", how="left")

    # No champion score for this series (its stratum champion never scored it):
    # fall back to the challenger rather than dropping the series.
    switches = pl.col("champion").is_null() | (
        pl.col("challenger_score") <= pl.col("champion_score") * (1 - min_improvement)
    )

    return combined.select(
        "series_id",
        pl.when(switches).then(pl.col("challenger")).otherwise(pl.col("champion")).alias("model"),
        pl.when(switches)
        .then(pl.col("challenger_score"))
        .otherwise(pl.col("champion_score"))
        .alias("score"),
        pl.lit(metric).alias("metric"),
    )


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
