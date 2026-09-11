"""Terminal interface to the engine.

engine ingest   --client teknic_mumbai --file data/raw/Demand_History_3_Years.xlsx
engine features --client teknic_mumbai
engine backtest --client teknic_mumbai
engine report   --client teknic_mumbai
engine forecast --client teknic_mumbai --horizon 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from engine.config import ClientProfile, ConfigError, list_clients, load_client
from engine.config.schema import quantile_column
from engine.evaluate import backtest as backtest_module
from engine.evaluate import scoreboard
from engine.features.pattern import (
    agreement_rate,
    profile_series,
    validate_against_labels,
)
from engine.features.segments import assign_volume_segments, segment_summary
from engine.forecast.arbitration import apply_selection
from engine.forecast.base import ForecastContext, conform, get_forecaster
from engine.ingest import get_reader, read_exogenous
from engine.store import EngineStore
from engine.utils.console import console, log_step

FORECAST_OUTPUT_COLUMNS = ["series_id", "period", "horizon", "model", "forecast"]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ERP-native demand forecasting engine (Phase 0: baseline portfolio).",
)

ClientOption = Annotated[str, typer.Option("--client", "-c", help="Client profile id.")]


def _load(client_id: str) -> tuple[ClientProfile, EngineStore]:
    try:
        client = load_client(client_id)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    return client, EngineStore(client.client_id)


def _require_data(store: EngineStore) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not store.table_exists("fact_demand"):
        console.print("[red]No data ingested yet. Run `engine ingest` first.[/red]")
        raise typer.Exit(code=1)
    fact = store.load_fact()
    if fact.is_empty():
        console.print("[red]No demand rows in the store. Run `engine ingest` first.[/red]")
        raise typer.Exit(code=1)
    return fact, store.load_dim()


def _ensure_profiles(store: EngineStore, client: ClientProfile, fact: pl.DataFrame) -> pl.DataFrame:
    if store.table_exists("series_profile"):
        profiles = store.load_series_profiles()
        if not profiles.is_empty():
            return profiles
    console.print("[dim]   no series profiles found — computing them now[/dim]")
    profiles = profile_series(fact, client.patterns, client.calendar.seasonal_period)
    store.write_series_profiles(profiles)
    return profiles


@app.command()
def clients() -> None:
    """List configured client profiles."""
    found = list_clients()
    if not found:
        console.print("[yellow]No client profiles in config/clients[/yellow]")
        return
    for client_id in found:
        profile = load_client(client_id)
        console.print(f"  [bold]{client_id}[/bold] — {profile.display_name}")


@app.command()
def ingest(
    client: ClientOption,
    file: Annotated[Path, typer.Option("--file", "-f", help="Source file to ingest.")],
) -> None:
    """Load a source file into the canonical store (idempotent per source file)."""
    profile, store = _load(client)

    with log_step(f"Ingesting {file.name} for {profile.display_name}"):
        reader = get_reader(profile.source.reader)
        result = reader.read(file, profile)
        store.replace_source(result.dim_series, result.fact_demand)

    report = result.report
    console.print(
        f"   [green]reconciled[/green] {report.canonical_rows:,} rows "
        f"({report.series_count} series, {report.period_min} to {report.period_max}); "
        f"quantity sum {report.canonical_quantity_sum:,.0f}"
    )
    console.print(f"   [dim]store: {store.path}[/dim]")


@app.command()
def exogenous(
    client: ClientOption,
    file: Annotated[Path, typer.Option("--file", "-f", help="Wide exogenous file.")],
) -> None:
    """Load context features (prices, promotions, calendars) into the store."""
    profile, store = _load(client)

    with log_step(f"Ingesting context features from {file.name}"):
        frame = read_exogenous(file, profile)
        store.replace_exogenous(frame)

    console.print(
        f"   [green]{frame.height:,} values[/green] across "
        f"{frame['feature'].n_unique()} features "
        f"({', '.join(profile.exogenous.features)}) "
        f"for {frame['series_id'].n_unique():,} series"
    )


@app.command()
def features(client: ClientOption) -> None:
    """Profile every series and classify its demand pattern."""
    profile, store = _load(client)
    fact, dim = _require_data(store)

    with log_step(f"Profiling {fact['series_id'].n_unique()} series"):
        profiles = profile_series(fact, profile.patterns, profile.calendar.seasonal_period)
        profiles = assign_volume_segments(
            profiles, profile.segments.head_share, profile.segments.mid_share
        )
        store.write_series_profiles(profiles)

    counts = profiles.group_by("pattern").len().sort("len", descending=True)
    console.print(
        "   detected patterns: "
        + ", ".join(f"{row['pattern']} {row['len']}" for row in counts.iter_rows(named=True))
    )
    console.print(
        "   volume segments: "
        + ", ".join(
            f"{row['volume_segment']} {row['series']} series / {row['volume_pct']}% of volume"
            for row in segment_summary(profiles).iter_rows(named=True)
        )
    )

    matrix = validate_against_labels(profiles, dim)
    if matrix is None:
        console.print(
            "   [dim]source carries no pattern labels — nothing to validate against[/dim]"
        )
        return

    run_id = store.start_run("pattern_validation", {"client_id": profile.client_id})
    store.write_pattern_validation(run_id, matrix)
    scoreboard.render_pattern_validation(matrix, agreement_rate(matrix))


@app.command()
def backtest(
    client: ClientOption,
    horizon: Annotated[int | None, typer.Option(help="Override the configured horizon.")] = None,
    origins: Annotated[
        int | None, typer.Option(help="Override the configured origin count.")
    ] = None,
    max_history: Annotated[
        int | None,
        typer.Option(help="Truncate each series to this many periods of history."),
    ] = None,
) -> None:
    """Run the rolling-origin backtest and persist the scoreboard."""
    profile, store = _load(client)
    fact, dim = _require_data(store)
    profiles = _ensure_profiles(store, profile, fact)

    if horizon is not None or origins is not None:
        evaluation = profile.evaluation.model_copy(
            update={
                "horizon": horizon or profile.evaluation.horizon,
                "origins": origins or profile.evaluation.origins,
            }
        )
        profile = profile.model_copy(update={"evaluation": evaluation})

    try:
        plan = backtest_module.build_plan(fact, profile)
    except backtest_module.BacktestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]{profile.display_name}[/bold] — {plan.n_origins} origins, "
        f"horizon {plan.horizon}, models: {', '.join(profile.models.enabled)}"
    )

    context = store.load_exogenous(profile.exogenous.features)
    if context is not None:
        console.print(
            f"   context features: {', '.join(profile.exogenous.features)} "
            f"({context.height:,} series-periods)"
        )

    with log_step("Backtesting"):
        predictions = backtest_module.run(
            fact, dim, profile, plan, max_history=max_history, exogenous=context
        )

    with log_step("Scoring"):
        result = scoreboard.compute(predictions, profiles, profile)
        run_id = store.start_run(
            "backtest",
            {
                "horizon": plan.horizon,
                "origins": plan.n_origins,
                "cutoffs": [str(c) for c in plan.cutoffs],
                "models": profile.models.enabled,
                "baseline": profile.models.baseline,
                "champions": result.champions,
            },
        )
        store.write_backtest_predictions(run_id, predictions)
        scoreboard.persist(store, run_id, result)

    console.print(
        "   default model per "
        + f"{profile.models.arbitration.stratify_by}: "
        + ", ".join(f"{k} -> {v}" for k, v in sorted(result.champions.items()))
    )
    console.print(f"   [dim]run {run_id} persisted[/dim]\n")
    _render_run(store, run_id, profile)


@app.command()
def coldstart(
    client: ClientOption,
    depths: Annotated[
        str, typer.Option(help="Comma-separated history lengths to test.")
    ] = "3,6,12,24,36",
) -> None:
    """Measure how each model's accuracy holds up as history is taken away.

    This panel contains no genuinely new products, so cold start is simulated by
    truncating each series' training window and re-running the same backtest.
    """
    profile, store = _load(client)
    fact, dim = _require_data(store)
    profiles = _ensure_profiles(store, profile, fact)

    try:
        depth_list = sorted({int(d) for d in depths.split(",") if d.strip()})
    except ValueError as exc:
        console.print(f"[red]Could not parse --depths {depths!r}: expected integers[/red]")
        raise typer.Exit(code=1) from exc

    plan = backtest_module.build_plan(fact, profile)
    rows: list[dict[str, object]] = []

    for depth in depth_list:
        with log_step(f"Backtesting with {depth} periods of history"):
            predictions = backtest_module.run(fact, dim, profile, plan, max_history=depth)
        result = scoreboard.compute(predictions, profiles, profile)
        overall = result.metrics_long.filter(
            (pl.col("scope_type") == "overall") & (pl.col("metric") == "median_series_wape")
        )
        for row in overall.iter_rows(named=True):
            rows.append({"depth": depth, "model": row["model"], "score": row["value"]})

    scoreboard.render_cold_start(pl.DataFrame(rows), profile.models.baseline)


@app.command()
def report(
    client: ClientOption,
    run_id: Annotated[int | None, typer.Option(help="Defaults to the latest backtest run.")] = None,
) -> None:
    """Print the scoreboard for a stored backtest run."""
    profile, store = _load(client)
    target = run_id or store.latest_run("backtest")
    if target is None:
        console.print("[red]No backtest runs found. Run `engine backtest` first.[/red]")
        raise typer.Exit(code=1)
    _render_run(store, target, profile)


def _render_run(store: EngineStore, run_id: int, profile: ClientProfile) -> None:
    info = store.run_info(run_id)
    metrics = store.load_metrics(run_id)
    selection = store.load_model_selection(run_id)

    if metrics.is_empty():
        console.print(f"[red]Run {run_id} has no metrics.[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]Run {run_id}[/bold] · {profile.display_name} · "
        f"[dim]{info['created_at'] if info else ''}[/dim]"
    )
    baseline = profile.models.baseline
    scoreboard.render_accuracy(metrics, baseline)
    scoreboard.render_slices(metrics, baseline, "fva_mae_pct", "MAE, volume-weighted")
    scoreboard.render_slices(metrics, baseline, "fva_median_wape_pct", "median series WAPE")
    scoreboard.render_slices(
        metrics,
        baseline,
        "fva_median_wape_pct",
        "median series WAPE",
        scope_type="segment",
        dimension="volume segment",
    )
    holdout_in_sample = metrics.filter(pl.col("scope_type") == "holdout").is_empty()
    scoreboard.render_holdout(metrics, baseline, holdout_in_sample)
    scoreboard.render_probabilistic(metrics, baseline)
    scoreboard.render_portfolio_by_scope(metrics, baseline)
    scoreboard.render_portfolio_by_scope(
        metrics, baseline, scope_type="holdout_segment", dimension="volume segment"
    )
    if not selection.is_empty():
        scoreboard.render_selection(selection)


@app.command()
def forecast(
    client: ClientOption,
    horizon: Annotated[int, typer.Option(help="Periods ahead to forecast.")] = 3,
    model: Annotated[
        str | None,
        typer.Option(help="Force one model instead of per-series arbitration."),
    ] = None,
) -> None:
    """Forecast forward from the end of history using the arbitrated portfolio."""
    profile, store = _load(client)
    fact, dim = _require_data(store)
    ctx = ForecastContext(client=profile, series_meta=dim)

    selection = None
    if model is None:
        latest = store.latest_run("backtest")
        if latest is not None:
            stored = store.load_model_selection(latest)
            selection = stored if not stored.is_empty() else None
        if selection is None:
            console.print(
                "[yellow]No arbitration available — run `engine backtest` first. "
                f"Falling back to {profile.models.baseline}.[/yellow]"
            )

    models = (
        [model]
        if model
        else sorted(selection["model"].unique().to_list())
        if selection is not None
        else [profile.models.baseline]
    )

    quantile_columns = [quantile_column(level) for level in ctx.quantile_levels]

    frames = []
    for model_name in models:
        with log_step(f"Forecasting with {model_name}"):
            predictions = get_forecaster(model_name).predict(fact, horizon, ctx)
            frames.append(
                conform(
                    predictions.with_columns(pl.lit(model_name).alias("model")),
                    FORECAST_OUTPUT_COLUMNS,
                    quantile_columns,
                )
            )

    combined = pl.concat(frames)
    final = apply_selection(combined, selection) if selection is not None else combined

    cutoff = fact["period"].max()
    final = final.with_columns(pl.lit(cutoff).cast(pl.Date).alias("cutoff_period"))

    run_id = store.start_run("forecast", {"horizon": horizon, "models": models})
    store.write_forecasts(run_id, final)
    n_quantiles = store.write_forecast_quantiles(run_id, final, ctx.quantile_levels)

    console.print(
        f"   [green]{final.height:,} forecasts[/green] for {final['series_id'].n_unique()} series "
        f"({horizon} periods from {cutoff}) — run {run_id}"
    )
    if n_quantiles:
        console.print(
            f"   [green]{n_quantiles:,} quantile values[/green] at levels "
            f"{profile.probabilistic.levels}"
        )

    preview_columns = ["series_id", "period", "horizon", "model", "forecast", *quantile_columns]
    preview = final.sort("series_id", "period").head(8)
    console.print(preview.select([c for c in preview_columns if c in preview.columns]))


if __name__ == "__main__":
    app()
