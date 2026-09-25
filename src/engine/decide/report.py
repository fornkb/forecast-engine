"""Terminal rendering for the decision layer."""

from __future__ import annotations

import polars as pl
from rich.table import Table

from engine.config.schema import CostModelConfig
from engine.utils.console import console


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _dva_cell(value: float | None) -> str:
    if value is None:
        return "-"
    colour = "green" if value > 0 else "red" if value < 0 else "white"
    return f"[{colour}]{value:+.1f}%[/{colour}]"


def render_cost_model(cost_model: CostModelConfig) -> None:
    console.print(
        f"   cost ratio [bold]{cost_model.shortage_cost_per_unit:g}[/bold] short : "
        f"[bold]{cost_model.holding_cost_per_unit:g}[/bold] held "
        f"-> service level [bold]{cost_model.service_level:.1%}[/bold]; "
        f"protection window {cost_model.protection_window} "
        f"period(s) (lead {cost_model.lead_time_periods} + review "
        f"{cost_model.review_period_periods})"
    )


def render_decision_cost(summary: pl.DataFrame, baseline: str) -> None:
    """Cost of each model's stocking decisions, against the baseline's.

    Costs are in the units the cost model is expressed in. They are not currency
    unless the configured costs were.
    """
    table = Table(title="Decision cost — held-out origins, lower is better")
    table.add_column("Model")
    table.add_column("Decision cost", justify="right")
    table.add_column("Held", justify="right")
    table.add_column("Short", justify="right")
    table.add_column("Fill rate", justify="right")
    table.add_column("DVA", justify="right")

    for row in summary.iter_rows(named=True):
        name = row["model"]
        table.add_row(
            f"[bold]{name}[/bold]" if name == baseline else name,
            _fmt(row["decision_cost"], 0),
            _fmt(row["units_held"], 0),
            _fmt(row["units_short"], 0),
            "-" if row["fill_rate"] is None else f"{row['fill_rate']:.1%}",
            _dva_cell(row.get("dva_pct")),
        )
    console.print(table)
    console.print(
        f"   [dim]{baseline} is the baseline. DVA is the share of its decision cost "
        "avoided. Holding and shortage are weighted by the configured ratio, so a "
        "model can hold more and still cost less.[/dim]"
    )


def render_calibration(summary: pl.DataFrame, target: float) -> None:
    """Whether stocking to a quantile actually achieves that service level.

    This is the check that makes the rest trustworthy. A model whose 90th
    percentile serves 70% of demand is not merely inaccurate — its stock levels
    are wrong in a direction no accuracy metric reports.
    """
    off = summary.filter(
        pl.col("fill_rate").is_not_null() & ((pl.col("fill_rate") - target).abs() > 0.05)
    )
    if off.is_empty():
        console.print(
            f"   [green]calibration ok[/green] — every model's fill rate is within "
            f"5 points of the {target:.0%} target"
        )
        return
    console.print(
        f"   [yellow]calibration warning[/yellow] — stocking to the {target:.0%} "
        "quantile did not deliver that fill rate:"
    )
    for row in off.sort("model").iter_rows(named=True):
        gap = row["fill_rate"] - target
        console.print(
            f"     {row['model']}: {row['fill_rate']:.1%} "
            f"({gap:+.1%} vs target) — its intervals are "
            + ("too wide" if gap > 0 else "too narrow")
        )


def render_by_scope(summary: pl.DataFrame, scope_column: str, baseline: str) -> None:
    table = Table(title=f"Decision cost by {scope_column.replace('_', ' ')}")
    table.add_column(scope_column.replace("_", " ").title())
    table.add_column("Best model")
    table.add_column("Decision cost", justify="right")
    table.add_column("Fill rate", justify="right")
    table.add_column("DVA", justify="right")

    for scope in summary[scope_column].unique().sort():
        rows = summary.filter(pl.col(scope_column) == scope).sort("decision_cost")
        best = rows.row(0, named=True)
        table.add_row(
            str(scope),
            f"[bold]{best['model']}[/bold]",
            _fmt(best["decision_cost"], 0),
            "-" if best["fill_rate"] is None else f"{best['fill_rate']:.1%}",
            _dva_cell(best.get("dva_pct")),
        )
    console.print(table)


def render_policy(policy: pl.DataFrame, preview_rows: int = 8) -> None:
    table = Table(title=f"Stock policy — first {preview_rows} of {policy.height:,} series")
    for name in ("Series", "Model", "Expected", "Target stock", "Safety stock", "Reorder pt"):
        table.add_column(name, justify="right" if name != "Series" else "left")

    for row in policy.head(preview_rows).iter_rows(named=True):
        table.add_row(
            row["series_id"],
            row["model"],
            _fmt(row["expected_demand"], 0),
            _fmt(row["target_stock"], 0),
            _fmt(row["safety_stock"], 0),
            _fmt(row["reorder_point"], 0),
        )
    console.print(table)


def render_coverage(gap: pl.DataFrame, total_series: int, covered: int) -> None:
    """Say plainly how much of the portfolio the decision layer cannot act on."""
    if gap.is_empty():
        console.print(f"   [green]policy covers all {total_series} series[/green]")
        return
    missing = total_series - covered
    console.print(
        f"   [yellow]no stock policy for {missing} of {total_series} series "
        f"({missing / total_series:.1%})[/yellow] — their arbitrated model "
        "produces no quantiles, so there is no distribution to stock against:"
    )
    for row in gap.iter_rows(named=True):
        console.print(f"     {row['model']}: {row['series']} series")
    console.print(
        "     [dim]these items get a forecast but never reach a reorder screen. "
        "Arbitration selects on accuracy and does not yet know that a model "
        "without quantiles cannot support a decision.[/dim]"
    )
