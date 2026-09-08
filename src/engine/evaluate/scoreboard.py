"""Scoreboard: metric aggregation, arbitration, persistence, and terminal rendering.

Results are persisted rather than only printed, so a later phase's portfolio can
be compared against this baseline run instead of against a remembered number.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from rich.table import Table

from engine.config.schema import ClientProfile
from engine.evaluate.metrics import (
    add_forecast_value_add,
    aggregate,
    aggregate_series_level,
    to_long,
)
from engine.forecast.arbitration import pick_champion, select_per_series, selection_summary
from engine.store import EngineStore
from engine.utils.console import console

PORTFOLIO_MODEL = "portfolio"


@dataclass(frozen=True)
class ScoreboardResult:
    metrics_long: pl.DataFrame
    selection: pl.DataFrame
    champion: str
    holdout_origin: int
    selection_is_in_sample: bool


def _scope(
    predictions: pl.DataFrame, scope_type: str, scope_column: str | None, baseline: str
) -> pl.DataFrame:
    group_by = ["model", scope_column] if scope_column else ["model"]
    wide = aggregate(predictions, group_by)
    if scope_type != "series":
        wide = wide.join(aggregate_series_level(predictions, group_by), on=group_by, how="left")
    wide = add_forecast_value_add(wide, baseline, scope_column)
    return to_long(wide, scope_type, scope_column)


def compute(
    predictions: pl.DataFrame, profiles: pl.DataFrame, client: ClientProfile
) -> ScoreboardResult:
    baseline = client.models.baseline
    labelled = predictions.join(
        profiles.select("series_id", "pattern"), on="series_id", how="left"
    ).with_columns(pl.col("pattern").fill_null("Unclassified"))

    holdout_origin = int(labelled["origin"].max())
    selection_source = labelled.filter(pl.col("origin") < holdout_origin)
    selection_is_in_sample = selection_source.is_empty()
    if selection_is_in_sample:
        # Single-origin run: selection and scoring cannot be separated.
        selection_source = labelled

    arbitration = client.models.arbitration
    series_scores = aggregate(selection_source, ["model", "series_id"]).select(
        "series_id", "model", pl.col(arbitration.selection_metric).alias("score")
    )
    champion = pick_champion(aggregate(selection_source, ["model"]), arbitration.selection_metric)
    selection = select_per_series(
        series_scores,
        arbitration.selection_metric,
        champion=champion,
        min_improvement=arbitration.min_improvement_pct / 100,
    )

    holdout = labelled.filter(pl.col("origin") == holdout_origin)
    portfolio = holdout.join(
        selection.select("series_id", "model"), on=["series_id", "model"], how="inner"
    ).with_columns(pl.lit(PORTFOLIO_MODEL).alias("model"))
    holdout_with_portfolio = pl.concat([holdout, portfolio])

    metrics_long = pl.concat(
        [
            _scope(labelled, "overall", None, baseline),
            _scope(labelled, "pattern", "pattern", baseline),
            _scope(labelled, "horizon", "horizon", baseline),
            _scope(labelled, "series", "series_id", baseline),
            _scope(holdout_with_portfolio, "holdout", None, baseline),
            _scope(holdout_with_portfolio, "holdout_pattern", "pattern", baseline),
        ]
    )

    return ScoreboardResult(
        metrics_long=metrics_long,
        selection=selection,
        champion=champion,
        holdout_origin=holdout_origin,
        selection_is_in_sample=selection_is_in_sample,
    )


def persist(store: EngineStore, run_id: int, result: ScoreboardResult) -> None:
    store.write_metrics(run_id, result.metrics_long)
    store.write_model_selection(run_id, result.selection)


# --------------------------------------------------------------------- render


def _pivot_scope(metrics: pl.DataFrame, scope_type: str, metric: str) -> pl.DataFrame:
    return (
        metrics.filter((pl.col("scope_type") == scope_type) & (pl.col("metric") == metric))
        .select("model", "scope_value", "value")
        .pivot(values="value", index="model", on="scope_value")
    )


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def _fva_cell(value: float | None) -> str:
    if value is None:
        return "-"
    colour = "green" if value > 0 else "red" if value < 0 else "white"
    return f"[{colour}]{value:+.1f}%[/{colour}]"


def render_accuracy(metrics: pl.DataFrame, baseline: str) -> None:
    rows = (
        metrics.filter(pl.col("scope_type") == "overall")
        .select("model", "metric", "value", "n_obs")
        .pivot(values="value", index=["model", "n_obs"], on="metric")
        .sort("mae")
    )

    table = Table(title="Accuracy across all origins", title_justify="left", header_style="bold")
    table.add_column("Model")
    table.add_column("MAE", justify="right")
    table.add_column("WAPE", justify="right")
    table.add_column("Bias", justify="right")
    table.add_column("FVA (MAE)", justify="right")
    table.add_column("Median series WAPE", justify="right")
    table.add_column("FVA (median)", justify="right")

    for row in rows.iter_rows(named=True):
        is_baseline = row["model"] == baseline
        label = f"{row['model']}  [dim](baseline)[/dim]" if is_baseline else row["model"]
        table.add_row(
            label,
            _fmt(row.get("mae")),
            _fmt(row.get("wape"), 3),
            _fmt(row.get("bias")),
            "[dim]—[/dim]" if is_baseline else _fva_cell(row.get("fva_mae_pct")),
            _fmt(row.get("median_series_wape"), 3),
            "[dim]—[/dim]" if is_baseline else _fva_cell(row.get("fva_median_wape_pct")),
        )
    console.print(table)
    console.print(
        "[dim]MAE/WAPE are volume-weighted — a few very large series dominate them. "
        "Median series WAPE weights every series equally.[/dim]"
    )


def render_pattern_slices(metrics: pl.DataFrame, baseline: str, metric: str, label: str) -> None:
    pivot = _pivot_scope(metrics, "pattern", metric)
    if pivot.is_empty():
        return

    scopes = [c for c in pivot.columns if c != "model"]
    table = Table(
        title=f"Forecast Value Add by demand pattern (vs {baseline}, {label})",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Model")
    for scope in scopes:
        table.add_column(scope, justify="right")

    for row in pivot.iter_rows(named=True):
        if row["model"] == baseline:
            continue
        table.add_row(row["model"], *[_fva_cell(row.get(scope)) for scope in scopes])
    console.print(table)


def render_holdout(metrics: pl.DataFrame, baseline: str, in_sample: bool) -> None:
    rows = (
        metrics.filter(pl.col("scope_type") == "holdout")
        .select("model", "metric", "value")
        .pivot(values="value", index="model", on="metric")
        .sort("mae")
    )

    note = (
        "selection was in-sample (single origin)"
        if in_sample
        else "arbitration chosen on earlier origins only"
    )
    table = Table(
        title=f"Final held-out origin — {note}",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Model")
    table.add_column("MAE", justify="right")
    table.add_column("WAPE", justify="right")
    table.add_column("FVA (MAE)", justify="right")
    table.add_column("Median series WAPE", justify="right")

    for row in rows.iter_rows(named=True):
        style = "bold" if row["model"] == PORTFOLIO_MODEL else None
        table.add_row(
            row["model"],
            _fmt(row.get("mae")),
            _fmt(row.get("wape"), 3),
            "[dim]—[/dim]" if row["model"] == baseline else _fva_cell(row.get("fva_mae_pct")),
            _fmt(row.get("median_series_wape"), 3),
            style=style,
        )
    console.print(table)


def render_portfolio_by_pattern(metrics: pl.DataFrame, baseline: str) -> None:
    """The arbitrated portfolio's per-pattern result on the held-out origin.

    This is the view the Phase 0 gate is judged on: the portfolio must not lose
    to the baseline on any demand pattern.
    """
    slices = metrics.filter(
        (pl.col("scope_type") == "holdout_pattern") & (pl.col("model") == PORTFOLIO_MODEL)
    )
    if slices.is_empty():
        return

    wide = slices.select("scope_value", "metric", "value").pivot(
        values="value", index="scope_value", on="metric"
    )

    table = Table(
        title=f"Portfolio vs {baseline} by pattern — held-out origin",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Pattern")
    table.add_column("MAE", justify="right")
    table.add_column("FVA (MAE)", justify="right")
    table.add_column("Median series WAPE", justify="right")
    table.add_column("FVA (median)", justify="right")

    for row in wide.sort("scope_value").iter_rows(named=True):
        table.add_row(
            row["scope_value"],
            _fmt(row.get("mae")),
            _fva_cell(row.get("fva_mae_pct")),
            _fmt(row.get("median_series_wape"), 3),
            _fva_cell(row.get("fva_median_wape_pct")),
        )
    console.print(table)


def render_selection(selection: pl.DataFrame) -> None:
    summary = selection_summary(selection)
    table = Table(
        title="Arbitration — model chosen per series",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Model")
    table.add_column("Series won", justify="right")
    table.add_column("Share", justify="right")
    for row in summary.iter_rows(named=True):
        table.add_row(row["model"], str(row["series_won"]), f"{row['share_pct']:.1f}%")
    console.print(table)


def render_pattern_validation(matrix: pl.DataFrame, agreement: float) -> None:
    colour = "green" if agreement >= 0.75 else "red"
    console.print(
        f"\n[bold]Pattern classifier[/bold] — agreement with source labels: "
        f"[{colour}]{agreement * 100:.1f}%[/{colour}] [dim](gate: 75%)[/dim]"
    )

    pivot = (
        matrix.pivot(values="n", index="ground_truth", on="predicted", aggregate_function="sum")
        .fill_null(0)
        .sort("ground_truth")
    )
    predicted_columns = [c for c in pivot.columns if c != "ground_truth"]

    table = Table(title="Confusion matrix", title_justify="left", header_style="bold")
    table.add_column("actual \\ predicted")
    for column in predicted_columns:
        table.add_column(column, justify="right")

    for row in pivot.iter_rows(named=True):
        cells = []
        for column in predicted_columns:
            value = row[column]
            if value == 0:
                cells.append("[dim]·[/dim]")
            elif column == row["ground_truth"]:
                cells.append(f"[green]{value}[/green]")
            else:
                cells.append(f"[yellow]{value}[/yellow]")
        table.add_row(row["ground_truth"], *cells)
    console.print(table)
