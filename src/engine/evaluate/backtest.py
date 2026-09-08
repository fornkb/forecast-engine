"""Rolling-origin backtesting.

The training window expands as the origin walks forward; at every origin each
model sees only data up to that cutoff. The final origin is held back from model
arbitration so the arbitrated portfolio can be scored on an origin it was not
selected on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from engine.config.schema import ClientProfile
from engine.forecast.base import ForecastContext, get_forecaster
from engine.utils.console import console


class BacktestError(RuntimeError):
    """Raised when history is too short for the configured evaluation plan."""


@dataclass(frozen=True)
class BacktestPlan:
    cutoffs: list[date]
    horizon: int

    @property
    def n_origins(self) -> int:
        return len(self.cutoffs)


def build_plan(history: pl.DataFrame, client: ClientProfile) -> BacktestPlan:
    """Choose cutoffs so every origin has a full horizon of actuals after it."""
    config = client.evaluation
    periods = history["period"].unique().sort().to_list()
    n_periods = len(periods)

    last_index = n_periods - config.horizon - 1
    if last_index < 0:
        raise BacktestError(
            f"History has {n_periods} periods, too few for a horizon of {config.horizon}"
        )

    indices = [
        last_index - config.origin_step * offset for offset in range(config.origins - 1, -1, -1)
    ]
    if indices[0] < 0:
        raise BacktestError(
            f"History has {n_periods} periods, too few for {config.origins} origins "
            f"of horizon {config.horizon} at step {config.origin_step}"
        )

    shortest_train = indices[0] + 1
    if shortest_train < config.min_train_periods:
        raise BacktestError(
            f"Earliest origin would train on {shortest_train} periods, below the configured "
            f"minimum of {config.min_train_periods}. Reduce origins or min_train_periods."
        )

    return BacktestPlan(cutoffs=[periods[i] for i in indices], horizon=config.horizon)


def truncate_history(history: pl.DataFrame, max_periods: int) -> pl.DataFrame:
    """Keep only each series' most recent `max_periods` observations.

    Used to simulate cold start: this panel has no genuinely new products, so the
    only honest way to test whether a model earns its place on short history is
    to take history away and watch what happens to the ranking.
    """
    return history.filter(
        pl.col("period").rank("ordinal", descending=True).over("series_id") <= max_periods
    )


def run(
    history: pl.DataFrame,
    series_meta: pl.DataFrame,
    client: ClientProfile,
    plan: BacktestPlan,
    max_history: int | None = None,
) -> pl.DataFrame:
    """Forecast every enabled model at every origin and attach actuals.

    `max_history` truncates each series' training window to that many periods,
    leaving the evaluation periods untouched.

    Returns columns: model, origin, cutoff_period, series_id, period, horizon,
    actual, forecast.
    """
    ctx = ForecastContext(client=client, series_meta=series_meta)
    actuals = history.select("series_id", "period", pl.col("quantity").alias("actual"))
    results: list[pl.DataFrame] = []

    for origin_index, cutoff in enumerate(plan.cutoffs, start=1):
        train = history.filter(pl.col("period") <= cutoff)
        if max_history is not None:
            train = truncate_history(train, max_history)
        console.print(
            f"   [dim]origin {origin_index}/{plan.n_origins} — "
            f"train through {cutoff}, {train.height} rows[/dim]"
        )

        for model_name in client.models.enabled:
            forecaster = get_forecaster(model_name)
            predictions = forecaster.predict(train, plan.horizon, ctx)

            scored = predictions.join(actuals, on=["series_id", "period"], how="inner")
            if scored.height != predictions.height:
                missing = predictions.height - scored.height
                console.print(
                    f"   [yellow]warning[/yellow] {model_name}: {missing} forecast rows had no "
                    "actual to score against"
                )

            results.append(
                scored.with_columns(
                    pl.lit(model_name).alias("model"),
                    pl.lit(origin_index, dtype=pl.Int32).alias("origin"),
                    pl.lit(cutoff).cast(pl.Date).alias("cutoff_period"),
                )
            )

    return pl.concat(results).select(
        "model",
        "origin",
        "cutoff_period",
        "series_id",
        "period",
        "horizon",
        "actual",
        "forecast",
    )
