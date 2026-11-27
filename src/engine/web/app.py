"""Forecast Engine v2 web app: a console for the ERP provider's team.

Portfolio of client companies → one client's summary, forecast, external-factor
reviews, company profile and settings. Every action calls the same
function as its CLI command (`engine forecast`, `engine reason*`). Slow work runs
as a background job the page follows.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.adjust import active
from engine.reason.forecast import UNASSIGNED, assign_family
from engine.reason.llm import LLMClient, LLMError
from engine.reason.outcome import outcome
from engine.reason.profile import (
    SCAN_ID,
    load_catalog,
    load_profile,
    profile_hash,
    profile_path,
    resolve_all,
    save_profile,
)
from engine.reason.report import CHANNEL_LABEL, TIER_LABEL, render_html, render_markdown, totals
from engine.reason.run import Options, apply_override, run_onboarding, run_review
from engine.reason.schemas import Driver, Override, ReasoningError, ReasonRun
from engine.web import insights
from engine.web.charts import BAND_LABEL, compact, sparkline, trend_chart
from engine.web.icons import icon
from engine.web.jobs import Busy, JobRunner
from engine.web.pipeline import run_engine_command
from engine.web.preview import SampleModel, preview_workspace
from engine.web.services import (
    NotFound,
    Workspace,
    client_status,
    client_view,
    parse_profile,
    profile_text,
    scenario_path,
    scenarios,
)

PRODUCT = "Forecast Engine v2"
HERE = Path(__file__).parent
MODES = ("live", "record", "replay")
READ_ONLY = (
    "Preview mode: profiles are read-only and drafting needs a real model. "
    "Start the app without --preview to change them."
)
NO_PIPELINE = (
    "Refreshing the forecast from the browser is off in preview mode. "
    "Start the app normally, or run `engine forecast` in a terminal."
)
LLMFactory = Callable[[ClientProfile], LLMClient | None]
Pipeline = Callable[..., str]


def _cause_label(value: object) -> str:
    """A driver id as people read it; the open scan and new factors get their own names."""
    text = str(value)
    if text == SCAN_ID:
        return "Open news scan"
    if text.startswith("new:"):
        return "New: " + text[4:].replace("_", " ").capitalize()
    return text.replace("_", " ").title()


def _flag_label(flag: str) -> str:
    for prefix, label in (("accepted_by_", "accepted by "), ("edited_by_", "edited by ")):
        if flag.startswith(prefix):
            return label + flag[len(prefix) :]
    return flag.replace("_", " ")


def _decision(flags: list[str]) -> str:
    if "dismissed" in flags:
        return "dismissed"
    if any(f.startswith("edited_by_") for f in flags):
        return "edited"
    if any(f.startswith("accepted_by_") for f in flags):
        return "accepted"
    return ""


def _decision_label(flags: list[str]) -> str:
    """The planner's latest decision on a finding, e.g. "Edited by Asha"."""
    for flag in flags:
        if flag == "dismissed":
            return "Dismissed"
        for prefix, label in (("accepted_by_", "Accepted by "), ("edited_by_", "Edited by ")):
            if flag.startswith(prefix):
                return label + flag[len(prefix) :]
    return ""


def _check_flags(flags: list[str]) -> list[str]:
    """Flags set by Python's checks, without the planner's own decisions."""
    return [
        f for f in flags if f != "dismissed" and not f.startswith(("accepted_by_", "edited_by_"))
    ]


def _signed(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "–"
    return f"{value:+.{digits}f}%"


def _dot(status) -> str:
    if status.error:
        return "bad"
    if not status.enabled:
        return "off"
    return "ok" if status.ready else "warn"


def create_app(
    workspace: Workspace | None = None,
    jobs: JobRunner | None = None,
    llm_factory: LLMFactory | None = None,
    preview: bool = False,
    pipeline: Pipeline | None = None,
) -> FastAPI:
    """`preview=True`: sample answers instead of the AI, on copied stores, read-only profiles.

    `pipeline` runs `engine forecast` / `engine backtest`; by default the real
    commands, and only when the app uses the default config and data folders.
    """
    if preview:
        workspace = workspace or preview_workspace()
        llm_factory = llm_factory or (lambda _client: SampleModel())
    ws = workspace or Workspace()
    if pipeline is None and not preview and ws.config_dir is None and ws.store_root is None:
        pipeline = run_engine_command
    runner = jobs or JobRunner()
    app = FastAPI(title=PRODUCT, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    env = templates.env
    env.filters["compact"] = compact
    env.filters["num"] = lambda v: "–" if v is None else f"{v:,.0f}"
    env.filters["pct"] = lambda v: f"{v:+g}%"
    env.filters["signed"] = _signed
    env.filters["month"] = lambda d: d.strftime("%b %Y") if d else "–"
    env.filters["mon"] = lambda d: d.strftime("%b") if d else "–"
    env.filters["day"] = lambda d: d.strftime("%d %b %Y") if d else "–"
    env.filters["when"] = lambda d: d.strftime("%d %b %Y, %H:%M") if d else "–"
    env.filters["money"] = lambda v: "–" if v is None else f"${v:,.4f}" if v < 1 else f"${v:,.2f}"
    env.filters["flag"] = _flag_label
    env.filters["title_id"] = _cause_label
    env.globals.update(
        PRODUCT=PRODUCT,
        CHANNEL_LABEL=CHANNEL_LABEL,
        TIER_LABEL=TIER_LABEL,
        BAND_LABEL=BAND_LABEL,
        MODES=MODES,
        decision=_decision,
        decision_label=_decision_label,
        check_flags=_check_flags,
        sparkline=sparkline,
        icon=icon,
        preview=preview,
    )

    # ----------------------------------------------------------------- helpers

    def planner(request: Request) -> str:
        name = unquote(request.cookies.get("planner", "")).strip()
        return re.sub(r"[^\w .@-]", "", name)[:60]

    def actor(request: Request) -> str:
        return planner(request) or "web user"

    def sidebar() -> list[dict]:
        items = []
        for cid in ws.client_ids():
            status = client_status(ws, cid, detail=False)
            items.append({"id": cid, "name": status.name, "dot": _dot(status)})
        return sorted(items, key=lambda i: (i["dot"] == "off", i["name"].lower()))

    def page(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
        context.setdefault("client_id", None)
        context.setdefault("tab", "")
        context.setdefault("section", "client" if context["client_id"] else "")
        if context["client_id"] and "head" not in context:
            context["head"] = client_status(ws, context["client_id"], detail=False)
        return templates.TemplateResponse(
            request,
            name,
            {
                "planner": planner(request),
                "ok": request.query_params.get("ok", ""),
                "err": request.query_params.get("err", ""),
                "clients_nav": sidebar(),
                "today_label": datetime.now().strftime("%a %d %b %Y"),
                **context,
            },
            status_code=status_code,
        )

    def back(path: str, ok: str = "", err: str = "", anchor: str = "") -> RedirectResponse:
        query = f"?ok={quote(ok)}" if ok else (f"?err={quote(err)}" if err else "")
        return RedirectResponse(f"{path}{query}{'#' + anchor if anchor else ''}", status_code=303)

    def model_for(client: ClientProfile) -> LLMClient | None:
        return llm_factory(client) if llm_factory else None

    def load_run(client_id: str, reason_id: str):
        client, store = ws.load(client_id)
        run = records.load_run(store, reason_id)
        if run is None:
            raise NotFound(f"Review {reason_id} was not found for {client_id}.")
        return client, store, run

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        # The app spends money on AI calls, so refuse form posts from other sites.
        if request.method == "POST":
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return PlainTextResponse("Cross-site request refused.", status_code=403)
        return await call_next(request)

    @app.exception_handler(NotFound)
    async def not_found(request: Request, exc: NotFound):
        return page(request, "error.html", status_code=404, message=str(exc))

    # --------------------------------------------------------------- portfolio

    @app.get("/", response_class=HTMLResponse)
    def portfolio(request: Request):
        views = [client_view(ws, cid) for cid in ws.client_ids()]
        order = {"bad": 0, "warn": 1, "info": 2, "ok": 3, "off": 4}
        views.sort(key=lambda v: (order[v.level], v.status.name.lower()))
        active_views = [v for v in views if v.status.enabled]
        kpis = {
            "clients": len(views),
            "active": len(active_views),
            "attention": sum(1 for v in active_views if v.level in ("bad", "warn")),
            "open": sum(v.digest.open for v in active_views if v.digest),
            "changes": sum(v.digest.outcome.changes for v in active_views if v.digest),
            "new_factors": sum(v.digest.outcome.new_factors for v in active_views if v.digest),
            "reviewed": sum(1 for v in active_views if v.reviewed_current),
        }
        feed = [
            (v, item)
            for v in active_views
            for item in v.attention
            if item.level in ("bad", "warn", "info")
        ]
        return page(
            request,
            "portfolio.html",
            section="portfolio",
            views=views,
            kpis=kpis,
            feed=feed[:12],
            today=datetime.now(UTC).date(),
        )

    # ------------------------------------------------------------------ client

    @app.get("/c/{client_id}", response_class=HTMLResponse)
    def summary(request: Request, client_id: str):
        ws.load(client_id)
        view = client_view(ws, client_id)
        chart, impact = None, []
        if view.forecast is not None:
            months = [insights.MonthPoint(**vars(m)) for m in view.forecast.total]
            band = "model"
            if view.reviewed_current:
                totals_by_month: dict[date, list[float]] = {}
                for b in view.digest.run.bands:
                    cell = totals_by_month.setdefault(b.period, [0.0, 0.0])
                    cell[0] += b.low
                    cell[1] += b.high
                for m in months:
                    if m.period in totals_by_month:
                        m.low, m.high = totals_by_month[m.period]
                band = "whatif"
            chart = trend_chart(view.forecast.history[-18:], months, band, "Total demand")
        if view.digest is not None:
            impact = {row["id"]: row for row in insights.family_impact(view.digest.run)}
        return page(
            request,
            "summary.html",
            client_id=client_id,
            tab="summary",
            head=view.status,
            v=view,
            chart=chart,
            impact=impact,
            job=runner.running(client_id),
            can_pipeline=pipeline is not None,
        )

    @app.get("/c/{client_id}/forecast", response_class=HTMLResponse)
    def forecast_page(request: Request, client_id: str):
        ws.load(client_id)
        view = client_view(ws, client_id)
        chart = None
        if view.forecast is not None:
            chart = trend_chart(view.forecast.history, view.forecast.total, "model", "Total demand")
        return page(
            request,
            "forecast.html",
            client_id=client_id,
            tab="forecast",
            head=view.status,
            v=view,
            f=view.forecast,
            acc=view.accuracy,
            chart=chart,
            job=runner.running(client_id),
            can_pipeline=pipeline is not None,
        )

    @app.get("/c/{client_id}/forecast.csv")
    def forecast_csv(client_id: str):
        _, store = ws.load(client_id)
        run_id = store.latest_run("forecast") if store.path.exists() else None
        if run_id is None:
            raise NotFound("No saved forecast to export.")
        points = store.load_forecasts(run_id)
        quantiles = insights._quantile_map(store.load_forecast_quantiles(run_id), 0.1)
        upper = insights._quantile_map(store.load_forecast_quantiles(run_id), 0.9)
        dim = {row["series_id"]: row for row in store.load_dim().iter_rows(named=True)}
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["series_id", "customer", "item_code", "item_group", "unit", "period", "horizon",
             "model", "forecast", "p10", "p90"]
        )  # fmt: skip
        for row in points.iter_rows(named=True):
            meta, key = dim.get(row["series_id"], {}), (row["series_id"], row["period"])
            low, high = quantiles.get(key), upper.get(key)
            writer.writerow(
                [row["series_id"], meta.get("customer") or "", meta.get("item_code") or "",
                 meta.get("item_group") or "", meta.get("uom") or "", row["period"],
                 row["horizon"], row["model"], round(row["forecast"] or 0.0, 2),
                 "" if low is None else round(low, 2), "" if high is None else round(high, 2)]
            )  # fmt: skip
        return PlainTextResponse(
            buffer.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{client_id}_forecast_{run_id}.csv"'
            },
        )

    def _pipeline_job(client_id: str, kind: str, **options):
        ws.load(client_id)
        here = f"/c/{client_id}/forecast"
        if pipeline is None:
            return back(here, err=NO_PIPELINE)
        try:
            job = runner.submit(client_id, kind, lambda: pipeline(kind, client_id, **options))
        except Busy as exc:
            return back(here, err=str(exc))
        return back(f"/c/{client_id}/jobs/{job.id}")

    @app.post("/c/{client_id}/forecast/refresh")
    def refresh_forecast(client_id: str, horizon: int = Form(6)):
        if not 1 <= horizon <= 24:
            return back(f"/c/{client_id}/forecast", err="Choose 1 to 24 months.")
        return _pipeline_job(client_id, "forecast", horizon=horizon)

    @app.post("/c/{client_id}/backtest")
    def run_backtest(client_id: str):
        return _pipeline_job(client_id, "backtest")

    @app.get("/c/{client_id}/reviews", response_class=HTMLResponse)
    def reviews(request: Request, client_id: str):
        client, store = ws.load(client_id)
        status = client_status(ws, client_id)
        rows = []
        if status.enabled:
            for reason_id, *_ in records.list_runs(store):
                run = records.load_run(store, reason_id)
                if run is not None:
                    rows.append(insights.review_digest(run))
        return page(
            request,
            "reviews.html",
            client_id=client_id,
            tab="reviews",
            head=status,
            s=status,
            rows=rows,
            scenarios=scenarios(client) if status.enabled else [],
            job=runner.running(client_id),
        )

    @app.post("/c/{client_id}/review")
    def start_review(
        client_id: str,
        action: str = Form("run"),
        llm_mode: str = Form("live"),
        scenario: str = Form(""),
        as_of: str = Form(""),
        full: bool = Form(False),
    ):
        client, store = ws.load(client_id)
        here = f"/c/{client_id}/reviews"
        try:
            if llm_mode not in MODES:
                raise ReasoningError(f"Unknown AI mode {llm_mode!r}")
            options = Options(
                llm_mode=llm_mode,
                scenario=scenario_path(client, scenario),
                dry_run=action == "estimate",
                full=full,
                as_of=date.fromisoformat(as_of) if as_of else None,
                client=model_for(client),
            )
            job = runner.submit(
                client_id,
                "estimate" if options.dry_run else "review",
                lambda: run_review(client, store, None, options),
            )
        except (ReasoningError, ValueError, Busy) as exc:
            return back(here, err=str(exc))
        return back(f"/c/{client_id}/jobs/{job.id}")

    @app.get("/c/{client_id}/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, client_id: str, job_id: str):
        ws.load(client_id)
        job = runner.get(job_id)
        if job is None or job.client_id != client_id:
            raise NotFound("This job is no longer in memory (the server was restarted).")
        tab = "forecast" if job.kind in ("forecast", "backtest") else "reviews"
        if job.status == "done":
            if job.kind == "review":
                return back(f"/c/{client_id}/runs/{job.result.reason_id}")
            if job.kind in ("onboard", "answers"):
                return back(f"/c/{client_id}/profile", ok=job.result)
            if job.kind == "forecast":
                return back(f"/c/{client_id}/forecast", ok="Forecast refreshed.")
            if job.kind == "backtest":
                return back(f"/c/{client_id}/forecast", ok="Backtest finished. Accuracy updated.")
            return page(
                request, "estimate.html", client_id=client_id, tab=tab, run=job.result, job=job
            )
        return page(request, "job.html", client_id=client_id, tab=tab, job=job)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = runner.get(job_id)
        if job is None:
            return JSONResponse({"status": "missing"}, status_code=404)
        return {"status": job.status, "seconds": job.seconds, "label": job.label}

    @app.get("/c/{client_id}/runs/{reason_id}", response_class=HTMLResponse)
    def run_page(request: Request, client_id: str, reason_id: str):
        _, store, run = load_run(client_id, reason_id)
        decisions = {o.finding_id: o for o in records.overrides(store, reason_id)}
        return page(
            request,
            "run.html",
            client_id=client_id,
            tab="reviews",
            decisions=decisions,
            digest=insights.review_digest(run),
            impact=insights.family_impact(run),
            **_run_view(run),
        )

    @app.post("/c/{client_id}/runs/{reason_id}/findings/{finding_id}")
    def decide(
        request: Request,
        client_id: str,
        reason_id: str,
        finding_id: str,
        action: str = Form(...),
        low: str = Form(""),
        high: str = Form(""),
        note: str = Form(""),
    ):
        client, store, _ = load_run(client_id, reason_id)
        here = f"/c/{client_id}/runs/{reason_id}"
        try:
            if action not in ("accept", "dismiss", "edit"):
                raise ReasoningError(f"Unknown action {action!r}")
            low_pct = float(low) if low.strip() else None
            high_pct = float(high) if high.strip() else None
            if action == "edit" and (low_pct is None or high_pct is None):
                raise ReasoningError("Editing needs both a low and a high percentage.")
            if action == "edit" and low_pct > high_pct:
                raise ReasoningError("Low must not be above high.")
            apply_override(
                client,
                store,
                Override(
                    reason_id=reason_id,
                    finding_id=finding_id,
                    action=action,
                    low_pct=low_pct if action == "edit" else None,
                    high_pct=high_pct if action == "edit" else None,
                    actor=actor(request),
                    note=note.strip()[:500],
                    at=datetime.now(UTC),
                ),
            )
        except (ReasoningError, ValueError) as exc:
            return back(here, err=str(exc), anchor=finding_id)
        verb = {"accept": "accepted", "dismiss": "dismissed", "edit": "edited"}[action]
        message = f"{finding_id} {verb}. Bands recalculated, no AI call."
        return back(here, ok=message, anchor=finding_id)

    @app.post("/c/{client_id}/runs/{reason_id}/findings/{finding_id}/watch")
    def watch(request: Request, client_id: str, reason_id: str, finding_id: str):
        """Accept a new factor and add it to the watch list, so later reviews track it."""
        here = f"/c/{client_id}/runs/{reason_id}"
        if preview:
            return back(here, err=READ_ONLY, anchor=finding_id)
        client, store, run = load_run(client_id, reason_id)
        finding = next((f for f in run.findings if f.finding_id == finding_id), None)
        if finding is None or not finding.new_factor:
            return back(here, err=f"{finding_id} is not a new factor.", anchor=finding_id)
        try:
            profile = load_profile(client)
            driver_id = finding.driver_id.removeprefix("new:")
            if any(d.id == driver_id for d in profile.drivers):
                return back(here, ok=f"{finding.new_factor} is already watched.", anchor=finding_id)
            driver = Driver(
                id=driver_id,
                template="topic_news",
                families=[finding.family_id],
                mechanism=finding.reasoning[:400],
                params={"topic": finding.watch_query or finding.new_factor},
            )
            updated = profile.model_copy(update={"drivers": [*profile.drivers, driver]})
            save_profile(client, store, updated, actor(request))
            if not insights.decided(finding.flags):
                apply_override(
                    client,
                    store,
                    Override(
                        reason_id=reason_id,
                        finding_id=finding_id,
                        action="accept",
                        actor=actor(request),
                        note="added to the watch list",
                        at=datetime.now(UTC),
                    ),
                )
        except ReasoningError as exc:
            return back(here, err=str(exc), anchor=finding_id)
        message = f"Added “{finding.new_factor}” to the watch list. Later reviews will track it."
        return back(here, ok=message, anchor=finding_id)

    @app.get("/c/{client_id}/runs/{reason_id}/download/{fmt}")
    def download(client_id: str, reason_id: str, fmt: str):
        _, _, run = load_run(client_id, reason_id)
        name = f"{client_id}_{reason_id}"
        if fmt == "html":
            body, media = render_html(run), "text/html"
        elif fmt == "md":
            body, media = render_markdown(run), "text/markdown"
        elif fmt == "json":
            body, media = run.model_dump_json(indent=2), "application/json"
        elif fmt == "csv":
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(
                ["series_id", "family_id", "period", "baseline", "low", "high", "findings"]
            )
            for b in run.bands:
                writer.writerow(
                    [b.series_id, b.family_id, b.period, round(b.baseline, 2), round(b.low, 2),
                     round(b.high, 2), " ".join(b.finding_ids)]
                )  # fmt: skip
            body, media = buffer.getvalue(), "text/csv"
        else:
            raise NotFound(f"Unknown format {fmt!r}")
        return PlainTextResponse(
            body,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"'},
        )

    # ----------------------------------------------------------------- profile

    def profile_context(client: ClientProfile, store, text: str | None = None) -> dict:
        context: dict = {"text": profile_text(client) if text is None else text}
        context["path"] = profile_path(client)
        context["catalog"] = sorted(load_catalog(client).values(), key=lambda t: t.id)
        context["versions"] = records.profile_versions(store)
        try:
            profile = load_profile(client) if profile_path(client).exists() else None
        except ReasoningError as exc:
            profile, context["load_error"] = None, str(exc)
        context["profile"] = profile
        context["resolved"] = resolve_all(profile, load_catalog(client)) if profile else []
        coverage: Counter = Counter()
        if profile and store.path.exists():
            try:
                dim = store.load_dim()
                columns = dim.select("series_id", "item_code", "item_group")
                for series, code, group in columns.iter_rows():
                    coverage[assign_family(profile, series, code or "", group or "")] += 1
            except Exception:  # no ERP data yet
                coverage.clear()
        context["coverage"] = coverage
        context["unassigned"] = coverage.get(UNASSIGNED, 0)
        context["has_erp"] = bool(coverage)
        return context

    @app.get("/c/{client_id}/profile", response_class=HTMLResponse)
    def profile_page(request: Request, client_id: str):
        client, store = ws.load(client_id)
        if client.reasoning is None or not client.reasoning.enabled:
            return back(f"/c/{client_id}/settings")
        return page(
            request,
            "profile.html",
            client_id=client_id,
            tab="profile",
            job=runner.running(client_id),
            **profile_context(client, store),
        )

    @app.post("/c/{client_id}/profile", response_class=HTMLResponse)
    def save_profile_text(request: Request, client_id: str, yaml_text: str = Form(...)):
        if preview:
            return back(f"/c/{client_id}/profile", err=READ_ONLY)
        client, store = ws.load(client_id)
        text = yaml_text.replace("\r\n", "\n")
        try:
            profile = parse_profile(text)
        except ReasoningError as exc:
            return page(
                request,
                "profile.html",
                status_code=422,
                client_id=client_id,
                tab="profile",
                job=runner.running(client_id),
                edit_error=str(exc),
                **profile_context(client, store, text),
            )
        path = profile_path(client)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")  # keeps the admin's comments
        records.save_profile_version(store, profile_hash(profile), actor(request), text)
        return back(f"/c/{client_id}/profile", ok="Profile saved.")

    @app.post("/c/{client_id}/profile/confirm")
    def confirm_profile(request: Request, client_id: str):
        if preview:
            return back(f"/c/{client_id}/profile", err=READ_ONLY)
        client, store = ws.load(client_id)
        try:
            text = profile_text(client)
            updated = re.sub(r"(?m)^status:\s*draft\s*$", "status: confirmed", text)
            if updated != text:
                profile = parse_profile(updated)
                profile_path(client).write_text(updated, encoding="utf-8")
                records.save_profile_version(store, profile_hash(profile), actor(request), updated)
            else:
                profile = load_profile(client).model_copy(update={"status": "confirmed"})
                save_profile(client, store, profile, actor(request))
        except ReasoningError as exc:
            return back(f"/c/{client_id}/profile", err=str(exc))
        return back(f"/c/{client_id}/profile", ok="Profile confirmed.")

    def _onboard_job(request: Request, client_id: str, kind: str, **kwargs):
        if preview:
            return back(f"/c/{client_id}/profile", err=READ_ONLY)
        client, store = ws.load(client_id)
        mode = kwargs.pop("llm_mode")
        who = actor(request)
        if mode not in MODES:
            return back(f"/c/{client_id}/profile", err=f"Unknown AI mode {mode!r}")

        def work() -> str:
            draft, _ = run_onboarding(
                client, store, llm_mode=mode, actor=who, llm=model_for(client), **kwargs
            )
            questions = len(draft.open_questions)
            return (
                f"Saved a {draft.status} profile: {len(draft.families)} families, "
                f"{len(draft.drivers)} drivers"
                + (f", {questions} questions for you below." if questions else ".")
            )

        try:
            job = runner.submit(client_id, kind, work)
        except (Busy, ReasoningError, LLMError) as exc:
            return back(f"/c/{client_id}/profile", err=str(exc))
        return back(f"/c/{client_id}/jobs/{job.id}")

    @app.post("/c/{client_id}/onboard")
    def onboard(
        request: Request,
        client_id: str,
        about: str = Form(""),
        llm_mode: str = Form("live"),
        replace: bool = Form(False),
    ):
        client, _ = ws.load(client_id)
        if profile_path(client).exists() and not replace:
            return back(
                f"/c/{client_id}/profile",
                err="A profile exists. Tick “replace the current profile” to draft a new one.",
            )
        return _onboard_job(request, client_id, "onboard", about=about, llm_mode=llm_mode)

    @app.post("/c/{client_id}/onboard/answers")
    def onboard_answers(
        request: Request,
        client_id: str,
        answers: str = Form(...),
        llm_mode: str = Form("live"),
        confirm: bool = Form(False),
    ):
        if not answers.strip():
            return back(f"/c/{client_id}/profile", err="Write your answers first.")
        return _onboard_job(
            request, client_id, "answers", answers=answers, confirm=confirm, llm_mode=llm_mode
        )

    # ---------------------------------------------------------- settings, cost

    @app.get("/c/{client_id}/settings", response_class=HTMLResponse)
    def settings_page(request: Request, client_id: str):
        client, store = ws.load(client_id)
        status = client_status(ws, client_id)
        spend = insights.month_spend(store) if store.path.exists() else (0.0, 0)
        return page(
            request,
            "settings.html",
            client_id=client_id,
            tab="settings",
            head=status,
            s=status,
            client=client,
            config=client.reasoning,
            spend=spend,
        )

    return app


def _run_view(run: ReasonRun) -> dict:
    by_family: dict[str, list] = {}
    for finding in run.findings:
        by_family.setdefault(finding.family_id, []).append(finding)
    counts = Counter(f.channel for f in run.findings if "dismissed" not in f.flags)
    families = [
        {
            "summary": family,
            "chart": trend_chart(family.history, family.months, "whatif", family.name),
            "findings": by_family.get(family.family_id, []),
            "moved": any(active(f) for f in by_family.get(family.family_id, [])),
        }
        for family in run.families
    ]
    return {
        "run": run,
        "families": families,
        "refs": {ref.id: ref for ref in run.references},
        "used": totals(run),
        "result": outcome(run),
        "counts": counts,
        "changed": [d.driver_id for d in run.drivers if d.status == "changed"],
        "unchanged": [d.driver_id for d in run.drivers if d.status == "unchanged"],
    }
