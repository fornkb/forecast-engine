"""Terminal interface to the engine.

engine ingest   --client teknic_mumbai --file data/raw/Demand_History_3_Years.xlsx
engine features --client teknic_mumbai
engine backtest --client teknic_mumbai
engine report   --client teknic_mumbai
engine forecast --client teknic_mumbai --horizon 3
"""

from __future__ import annotations

import socket
import webbrowser
from pathlib import Path
from typing import Annotated

import duckdb
import polars as pl
import typer

from engine.config import ClientProfile, ConfigError, list_clients, load_client
from engine.config.schema import quantile_column
from engine.decide import report as decide_report
from engine.decide.cost import add_decision_value_add, with_decision_costs
from engine.decide.cost import aggregate as decision_aggregate
from engine.decide.cost import to_long as decision_to_long
from engine.decide.policy import build_policy, coverage_gap
from engine.evaluate import backtest as backtest_module
from engine.evaluate import scoreboard
from engine.features.pattern import (
    agreement_rate,
    profile_series,
    validate_against_labels,
)
from engine.features.segments import assign_volume_segments, segment_summary
from engine.forecast import calibration
from engine.forecast.arbitration import apply_selection
from engine.forecast.base import ForecastContext, conform, get_forecaster
from engine.ingest import IngestError, get_reader, read_exogenous
from engine.reason.cli import register_reason_commands
from engine.store import (
    EngineStore,
    PublishError,
    export_forecasts,
    publish_forecasts,
    publish_policy,
)
from engine.store.export import SUPPORTED as EXPORT_FORMATS
from engine.utils.console import console, log_step

FORECAST_OUTPUT_COLUMNS = ["series_id", "period", "horizon", "model", "forecast"]
PREVIEW_ROWS = 8

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


register_reason_commands(app, lambda client: _load(client))


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
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Source file. Omit for database-backed readers."),
    ] = None,
) -> None:
    """Load a source into the canonical store (idempotent per source)."""
    profile, store = _load(client)

    reader = get_reader(profile.source.reader)
    if reader.requires_path and file is None:
        console.print(f"[red]Reader {profile.source.reader!r} reads a file; pass --file.[/red]")
        raise typer.Exit(code=1)
    if not reader.requires_path and file is not None:
        # Silently ignoring it would let someone believe they had loaded a file
        # while the engine read the database.
        console.print(
            f"[red]Reader {profile.source.reader!r} reads from the database "
            "configured in the client profile; --file would be ignored.[/red]"
        )
        raise typer.Exit(code=1)

    origin = file.name if file else profile.source.database.label
    try:
        with log_step(f"Ingesting {origin} for {profile.display_name}"):
            result = reader.read(file, profile)
            store.replace_source(result.dim_series, result.fact_demand)
    except IngestError as exc:
        # A bad source is an operator problem — an unset variable, a renamed
        # column — and reads as one. A traceback buries the message.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

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

    try:
        with log_step(f"Ingesting context features from {file.name}"):
            frame = read_exogenous(file, profile)
            store.replace_exogenous(frame)
    except IngestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

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
        # The decision layer scores a stock level chosen at each origin, so it
        # needs the distribution the model produced there, not just the point.
        store.write_backtest_quantiles(run_id, predictions, profile.probabilistic.levels)
        scoreboard.persist(store, run_id, result)

        if profile.probabilistic.enabled and profile.probabilistic.calibrate:
            # Fitted on the earlier origins only. The final one stays untouched so
            # the coverage reported against it is not coverage of its own fit.
            stored = store.load_backtest_quantiles(run_id)
            offsets = calibration.fit_offsets(
                stored, profiles, exclude_origin=int(stored["origin"].max())
            )
            store.write_quantile_calibration(run_id, offsets)

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
    export: Annotated[
        Path | None,
        typer.Option(
            "--export", "-e", help="Also write every forecast to a .csv or .parquet file."
        ),
    ] = None,
) -> None:
    """Forecast forward from the end of history using the arbitrated portfolio."""
    if export is not None and export.suffix.lower() not in EXPORT_FORMATS:
        console.print(
            f"[red]Cannot export to {export.name!r}: use a "
            f"{' or '.join(EXPORT_FORMATS)} file name.[/red]"
        )
        raise typer.Exit(code=1)

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

    calibrated_from = None
    if profile.probabilistic.enabled and profile.probabilistic.calibrate:
        source_run = store.latest_run("backtest")
        offsets = (
            store.load_quantile_calibration(source_run)
            if source_run is not None
            else pl.DataFrame()
        )
        if offsets.is_empty():
            console.print(
                "[yellow]   probabilistic.calibrate is on but no calibration is stored — "
                "run `engine backtest` to fit one. Quantiles are uncorrected.[/yellow]"
            )
        else:
            # Corrected here, once, so the store, the export and the stock policy
            # all read the same numbers rather than each remembering to ask.
            final = calibration.apply_to_wide(
                final,
                offsets,
                _ensure_profiles(store, profile, fact),
                ctx.quantile_levels,
            )
            calibrated_from = source_run

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
            + (
                f", calibrated from backtest run {calibrated_from}"
                if calibrated_from is not None
                else " (uncalibrated)"
            )
        )

    if export is not None:
        rows = export_forecasts(final, dim, export)
        console.print(f"   [green]exported {rows:,} rows[/green] to {export.resolve()}")

    preview_columns = ["series_id", "period", "horizon", "model", "forecast", *quantile_columns]
    preview = final.sort("series_id", "period").head(PREVIEW_ROWS)
    console.print(
        f"\n   [dim]Preview — first {PREVIEW_ROWS} of {final.height:,} rows. Every row is saved "
        "in the store; add --export forecasts.csv to get them as a file.[/dim]"
    )
    console.print(preview.select([c for c in preview_columns if c in preview.columns]))


@app.command()
def decide(
    client: ClientOption,
    run_id: Annotated[
        int | None, typer.Option(help="Backtest run to score. Defaults to the latest.")
    ] = None,
) -> None:
    """Score each model on what its stocking decisions cost, and set stock levels."""
    profile, store = _load(client)
    cost_model = profile.cost_model
    if cost_model is None:
        console.print(
            f"[red]Client {client!r} has no cost_model block, so there is no cost to "
            "measure against. Add one to the client profile.[/red]"
        )
        raise typer.Exit(code=1)

    if run_id is None:
        run_id = store.latest_run("backtest")
        if run_id is None:
            console.print("[red]No backtest run found. Run `engine backtest` first.[/red]")
            raise typer.Exit(code=1)

    decide_report.render_cost_model(cost_model)

    scored = store.load_backtest_quantiles(run_id, level=cost_model.service_level)
    if scored.is_empty():
        console.print(
            f"[red]Backtest run {run_id} stored no quantiles at level "
            f"{cost_model.service_level:.4g}. Re-run `engine backtest` — runs from "
            "before the decision layer did not persist them.[/red]"
        )
        raise typer.Exit(code=1)

    # Only the final origin, for the same reason accuracy is reported on it: the
    # earlier ones were used to arbitrate, so scoring on them flatters whichever
    # model the arbitration picked.
    holdout = scored.filter(pl.col("origin") == scored["origin"].max())

    # The stored quantiles are raw. A forward forecast applies the calibration, so
    # the decision reported here is the one the engine would actually take.
    offsets = store.load_quantile_calibration(run_id)
    if profile.probabilistic.calibrate and not offsets.is_empty():
        raw = holdout
        holdout = calibration.apply_offsets(
            holdout, offsets, _ensure_profiles(store, profile, store.load_fact())
        )
        decide_report.render_calibration_effect(
            calibration.coverage(raw, ["model"]),
            calibration.coverage(holdout, ["model"]),
            cost_model.service_level,
        )

    with log_step(f"Costing {holdout.height:,} decisions from backtest run {run_id}"):
        costed = with_decision_costs(holdout, cost_model)
        overall = add_decision_value_add(
            decision_aggregate(costed, ["model"]), profile.models.baseline
        )
        metrics = decision_to_long(overall, "holdout", None)

        profiles = _ensure_profiles(store, profile, store.load_fact())
        segments = profiles.select("series_id", "volume_segment")
        by_segment = add_decision_value_add(
            decision_aggregate(
                costed.join(segments, on="series_id", how="left"),
                ["model", "volume_segment"],
            ),
            profile.models.baseline,
            "volume_segment",
        )
        metrics = pl.concat(
            [metrics, decision_to_long(by_segment, "volume_segment", "volume_segment")]
        )

        # forecast_run_id is recorded so a publish can tell whether a stored
        # policy belongs to the forecast it is publishing beside it.
        decision_run = store.start_run(
            "decision",
            {
                "backtest_run_id": run_id,
                "forecast_run_id": store.latest_run("forecast"),
                "service_level": cost_model.service_level,
                "cost_model": cost_model.model_dump(),
            },
        )
        store.write_decision_metrics(decision_run, metrics)

    decide_report.render_decision_cost(overall, profile.models.baseline)
    decide_report.render_calibration(overall, cost_model.service_level)
    decide_report.render_by_scope(by_segment, "volume_segment", profile.models.baseline)

    forecast_run = store.latest_run("forecast")
    if forecast_run is None:
        console.print(
            "\n   [yellow]no forecast run yet — run `engine forecast` to get stock "
            "levels for the coming periods[/yellow]"
        )
        return

    forward = store.load_forecasts(forecast_run)
    policy = build_policy(forward, store.load_forecast_quantiles(forecast_run), cost_model)
    store.write_decision_policy(decision_run, policy)
    console.print()
    decide_report.render_coverage(
        coverage_gap(forward, policy),
        forward["series_id"].n_unique(),
        policy["series_id"].n_unique(),
    )
    decide_report.render_policy(policy, PREVIEW_ROWS)
    console.print(
        f"   [dim]run {decision_run}, from forecast run {forecast_run}. "
        "Target stock covers the protection window; summing per-period quantiles "
        "makes it slightly conservative.[/dim]"
    )


@app.command()
def publish(
    client: ClientOption,
    run_id: Annotated[
        int | None, typer.Option(help="Forecast run to publish. Defaults to the latest.")
    ] = None,
) -> None:
    """Write a stored forecast into the ERP database."""
    profile, store = _load(client)

    if run_id is None:
        run_id = store.latest_run("forecast")
        if run_id is None:
            console.print("[red]No forecast run found. Run `engine forecast` first.[/red]")
            raise typer.Exit(code=1)

    try:
        with log_step(f"Publishing run {run_id} for {profile.display_name}"):
            report = publish_forecasts(store, profile, run_id)
    except PublishError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"   [green]{report.forecast_rows:,} forecasts[/green] for "
        f"{report.series_count} series ({report.horizon} periods from "
        f"{report.cutoff_period}) -> {report.target}"
    )
    if report.quantile_rows:
        console.print(f"   [green]{report.quantile_rows:,} quantile values[/green]")
    console.print(
        "   [dim]the ERP reads forecast_output.v_current_forecast, which now "
        f"resolves to run {report.run_id}[/dim]"
    )

    # The policy published is the one derived from *this* forecast, not whichever
    # is newest. Publishing a reorder point beside a forecast it was not built
    # from would be worse than publishing no reorder point.
    decision_run = store.latest_run_from("decision", "forecast_run_id", report.run_id)
    if decision_run is None:
        console.print(
            "   [yellow]no stock policy published[/yellow] — run `engine decide` "
            "after this forecast to publish reorder points beside it"
        )
        return

    try:
        with log_step(f"Publishing stock policy from decision run {decision_run}"):
            policy = publish_policy(store, profile, decision_run, report.run_id)
    except PublishError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"   [green]{policy.series_count} stock levels[/green] at a "
        f"{policy.service_level:.0%} service level over {policy.window_periods} "
        f"period(s) -> {policy.target}"
    )
    console.print(
        "   [dim]the ERP reads forecast_output.v_current_stock_policy for what to reorder[/dim]"
    )


@app.command()
def ui(
    client: ClientOption,
    port: Annotated[int, typer.Option(help="Local port for the viewer.")] = 4213,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Start the viewer without opening a browser.")
    ] = False,
) -> None:
    """Open this client's results in DuckDB's browser viewer, read-only."""
    profile, store = _load(client)
    if not store.path.exists():
        console.print("[red]No data ingested yet. Run `engine ingest` first.[/red]")
        raise typer.Exit(code=1)

    # A second viewer on a busy port does not fail — it silently leaves the old
    # one answering, so the browser shows whatever that earlier viewer shows.
    if _port_in_use(port):
        console.print(
            f"[red]Port {port} is already in use, most likely by an earlier viewer.[/red] "
            "Close it (press Enter in its terminal), or pick another port with "
            f"--port {port + 1}."
        )
        raise typer.Exit(code=1)

    # The viewer keeps its own notebooks in a separate database it creates on
    # first use. Opening the engine file with read_only=True makes the whole
    # session read-only, so that database can never be created and the viewer
    # fails with 'Catalog "_duckdb_ui" does not exist'. Instead the session is
    # writable and only the engine's file is attached read-only.
    con = duckdb.connect()
    escaped = str(store.path).replace("'", "''")
    con.execute(f"ATTACH '{escaped}' AS engine (READ_ONLY)")
    con.execute("USE engine")
    con.execute("INSTALL ui")
    con.execute("LOAD ui")
    con.execute(f"SET ui_local_port = {port}")
    con.execute("CALL start_ui_server()")

    url = f"http://localhost:{port}"
    if not no_browser:
        webbrowser.open(url)

    console.print(f"[bold]{profile.display_name}[/bold] — viewer open at [cyan]{url}[/cyan]")
    console.print(
        "   [dim]Read-only. Close it before running other engine commands — "
        "while it is open they cannot write to the database.[/dim]"
    )
    try:
        input("   Press Enter to close the viewer. ")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        con.execute("CALL stop_ui_server()")
        con.close()
        console.print("   [dim]viewer closed[/dim]")


def _port_in_use(port: int) -> bool:
    """True if something is already listening on localhost at this port."""
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    app()
