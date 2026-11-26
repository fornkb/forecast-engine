"""One monthly reasoning run, end to end, with hard limits on model calls.

    profile + saved forecast → signals → change filter → (triage) → reasoning → checks → bands

Typical cost: 0 model calls when nothing changed, 1 when something did, 2 when
there is a lot of news to screen. The same inputs twice return the saved run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.adjust import bands, check, effective, number
from engine.reason.config import ReasoningConfig
from engine.reason.connectors import Context, http_get
from engine.reason.connectors.http import HttpTransport
from engine.reason.connectors.scenario import ScenarioPack
from engine.reason.forecast import ForecastView, latest_forecast_run, load_forecast
from engine.reason.llm import LLMClient, LLMError, Mode, get_client
from engine.reason.llm.cost import cost, estimate_tokens
from engine.reason.profile import (
    _config_root,
    load_catalog,
    load_profile,
    profile_hash,
    resolve_all,
)
from engine.reason.reasoner import PROMPT_VERSION, build_packets, header, system_prompt
from engine.reason.schemas import (
    Finding,
    LLMCall,
    Override,
    ReasoningError,
    ReasoningOut,
    ReasonRun,
    TriageOut,
    Usage,
    digest,
)
from engine.reason.signals import collect, new_articles
from engine.reason.triage import SYSTEM as TRIAGE_SYSTEM
from engine.reason.triage import apply_triage, prefilter, round_robin, triage_prompt
from engine.store.duckdb_store import EngineStore


def require_reasoning(client: ClientProfile) -> ReasoningConfig:
    config = client.reasoning
    if config is None or not config.enabled:
        raise ReasoningError(
            f"Reasoning is not enabled for {client.client_id}. Add `reasoning: {{enabled: true}}` "
            "to its client profile."
        )
    return config


def _resolve_path(client: ClientProfile, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else _config_root(client).parent / path


def recordings_path(client: ClientProfile) -> Path:
    """Where `--llm record` saves answers and `--llm replay` reads them."""
    configured = _resolve_path(client, require_reasoning(client).llm.recordings_dir)
    return configured or _config_root(client).parent / "data" / "recordings" / client.client_id


@dataclass
class Options:
    llm_mode: Mode = "live"
    scenario: Path | None = None
    dry_run: bool = False
    full: bool = False
    as_of: date | None = None
    http: HttpTransport = http_get
    client: LLMClient | None = None  # injected in tests


class _Caller:
    """Makes model calls within the run's budget and records every one in the ledger."""

    def __init__(
        self, config: ReasoningConfig, store: EngineStore, llm: LLMClient | None, reason_id: str
    ) -> None:
        self.config, self.store, self.llm, self.reason_id = config, store, llm, reason_id
        self.calls: list[LLMCall] = []

    @property
    def remaining(self) -> int:
        return self.config.budget.max_calls_per_run - len(self.calls)

    def __call__(self, stage: str, model: str, system: str, prompt: str, schema):
        if self.llm is None:
            raise LLMError("no model client")
        started, clock = datetime.now(UTC), time.perf_counter()
        try:
            result = self.llm.generate(
                system=system,
                prompt=prompt,
                schema=schema,
                model=model,
                max_output_tokens=self.config.llm.max_output_tokens,
                effort=self.config.llm.effort,
                name=stage,
            )
            usage, outcome, key = (
                result.usage,
                ("replayed" if result.replayed else "ok"),
                result.request_hash,
            )
        except LLMError as exc:
            result, usage, outcome, key = None, Usage(), f"error:{exc.reason}"[:200], ""
        call = LLMCall(
            call_id=uuid4().hex[:12],
            stage=stage,
            provider=self.config.llm.provider,
            model=model,
            started_at=started,
            seconds=round(time.perf_counter() - clock, 2),
            usage=usage,
            cost=0.0 if outcome == "replayed" else cost(usage, self.config.llm.rates.get(model)),
            outcome=outcome,
            request_hash=key,
        )
        self.calls.append(call)
        records.save_call(self.store, self.reason_id, call)
        return result, call


def _carry(previous: ReasonRun | None, driver_ids: set[str], months: list[date]) -> list[Finding]:
    """Earlier findings for unchanged drivers, kept for months still in the horizon."""
    if previous is None:
        return []
    kept = []
    for finding in previous.findings:
        still = [m for m in finding.months if m in months]
        if finding.driver_id in driver_ids and still and "dismissed" not in finding.flags:
            kept.append(
                finding.model_copy(
                    update={
                        "months": still,
                        "carried_from": finding.carried_from or previous.reason_id,
                    }
                )
            )
    return kept


def _measured_index(packet_refs: dict, states) -> dict[str, dict]:
    by_summary = {
        s.summary: s.data for state in states for s in state.signals if s.kind == "measured"
    }
    return {key: by_summary.get(ref.text, {}) for key, ref in packet_refs.items() if key[0] == "M"}


def run_review(
    client: ClientProfile,
    store: EngineStore,
    forecast_run_id: int | None = None,
    options: Options | None = None,
) -> ReasonRun:
    options = options or Options()
    config = require_reasoning(client)
    profile = load_profile(client)
    catalog = load_catalog(client)
    drivers = resolve_all(profile, catalog)
    run_id = forecast_run_id or latest_forecast_run(store)
    view: ForecastView = load_forecast(client, store, profile, run_id)

    pack_path = _resolve_path(client, options.scenario or config.sources.scenario_pack)
    scenario = ScenarioPack(pack_path) if pack_path else None
    zone = ZoneInfo(config.timezone)
    as_of = (
        options.as_of
        or (scenario.as_of if scenario and scenario.as_of else None)
        or datetime.now(zone).date()
    )
    moment = (
        datetime.now(UTC)
        if as_of == datetime.now(zone).date()
        else datetime(as_of.year, as_of.month, as_of.day, 12, tzinfo=UTC)
    )

    # --full ignores the last run: every driver is reassessed with all its articles.
    previous = None if options.full else records.latest_run(store)
    ctx = Context(
        as_of=moment,
        timezone=config.timezone,
        months=view.months,
        sources=config.sources,
        http=options.http,
        history=view.history,
    )
    states, stats = collect(
        drivers, ctx, store, previous, scenario=scenario, cache_hours=config.sources.cache_hours
    )

    by_id = {d.id: d for d in drivers}
    usable = [s for s in states if s.status in ("changed", "unchanged")]
    if options.full or previous is None:
        assess = list(usable)
    else:
        assess = [s for s in usable if s.status == "changed"]
    unchanged = [s.driver_id for s in usable if s not in assess]
    llm = config.llm
    if not llm.model:
        raise ReasoningError("Set reasoning.llm.model in the client profile.")
    inputs_hash = digest(
        [
            PROMPT_VERSION,
            view.digest,
            profile_hash(profile),
            as_of,
            [(s.driver_id, s.status, s.fingerprint, s.seen_urls[:50]) for s in states],
            llm.provider,
            llm.model,
            llm.triage_model,
            llm.effort,
            config.budget.model_dump(),
            options.full,
            scenario.name if scenario else None,
        ]
    )
    if not options.dry_run:
        saved = records.run_by_inputs(store, inputs_hash)
        if saved is not None:
            return saved
    reason_id = f"R{as_of:%Y%m%d}-{inputs_hash[:6]}"

    client_llm = options.client
    needs_model = bool(assess) and not options.dry_run and config.budget.max_calls_per_run > 0
    if client_llm is None and needs_model:
        client_llm = get_client(llm, options.llm_mode, recordings_path(client))
    caller = _Caller(config, store, client_llm, reason_id)
    gaps, not_assessed = [], []

    # Articles: only ones no earlier run has seen, after the free filters.
    articles = prefilter(
        {s.driver_id: new_articles(s, previous) for s in assess},
        config.sources.news.max_articles_per_driver,
    )
    total = sum(len(v) for v in articles.values())
    triage_model = llm.triage_model or llm.model
    estimate_calls = 0
    if total > config.budget.triage_threshold:
        prompt, index = triage_prompt(
            profile.company.name,
            {d: (by_id[d].driver.mechanism or by_id[d].template.description) for d in articles},
            articles,
        )
        estimate_calls += 1
        if options.dry_run or not needs_model or caller.remaining < 2:
            articles = round_robin(articles, config.budget.max_articles)
        else:
            result, _ = caller("triage", triage_model, TRIAGE_SYSTEM, prompt, TriageOut)
            if result is None:
                gaps.append("article triage failed; kept the newest articles instead")
                articles = round_robin(articles, config.budget.max_articles)
            else:
                articles = apply_triage(result.data, index, config.budget.max_articles)
    else:
        articles = round_robin(articles, config.budget.max_articles)

    # One packet for all changed drivers (split only if it would exceed the token budget).
    head = header(profile, view, as_of.isoformat())
    blocks = []
    for state in assess:
        driver = by_id[state.driver_id]
        signals = [s for s in state.signals if s.kind != "news" or not s.ok]
        earlier = [
            f
            for f in (previous.findings if previous else [])
            if f.driver_id == state.driver_id and set(f.months) & set(view.months)
        ]
        blocks.append((driver, signals, articles.get(state.driver_id, []), earlier))
    system = system_prompt(config.budget.max_judgment_pct)
    packets, skipped = build_packets(
        head,
        blocks,
        unchanged,
        max_tokens=config.budget.max_input_tokens,
        max_packets=max(0, caller.remaining if not options.dry_run else 10),
    )
    not_assessed += [f"{d}: over the call budget" for d in skipped]

    findings, quiet, references, summaries = [], [], {}, []
    families_by_driver = {d.id: d.families for d in drivers}
    if options.dry_run:
        tokens = sum(estimate_tokens(system + p.text) for p in packets)
        rates = llm.rates.get(llm.model)
        guess = Usage(input_tokens=tokens, output_tokens=1500 * len(packets))
        estimate = {
            "calls": len(packets) + estimate_calls,
            "input_tokens": tokens,
            "output_tokens_guess": guess.output_tokens,
            "cost_guess": cost(guess, rates),
            "drivers_to_assess": [s.driver_id for s in assess],
            "unchanged": unchanged,
        }
    else:
        estimate = {}
        for packet in packets:
            result, call = caller("reason", llm.model, system, packet.text, ReasoningOut)
            references.update(packet.references)
            if result is None:
                not_assessed += [
                    f"{d}: model call failed ({call.outcome})" for d in packet.driver_ids
                ]
                continue
            measured = _measured_index(packet.references, states)
            checked, calm, notes = check(
                result.data,
                packet.references,
                {d: families_by_driver[d] for d in packet.driver_ids},
                view.months,
                config.budget.max_judgment_pct,
                measured,
            )
            findings += checked
            quiet += calm
            gaps += notes
            summaries.append(result.data.summary)

    findings = number([*findings, *_carry(previous, set(unchanged), view.months)])
    band_rows, families, stacked = bands(view, findings)
    errored = [s for s in states if s.status == "error"]
    gaps += stacked
    gaps += [
        f"{s.driver_id}: {next((x.error for x in s.signals if not x.ok), 'error')}" for s in errored
    ]
    gaps += [f"{s.driver_id}: no data this month" for s in states if s.status == "no_data"]
    if view.unassigned:
        gaps.append(f"{len(view.unassigned)} SKUs match no family in the profile")
    status = (
        "dry_run" if options.dry_run else ("partial" if not_assessed or errored else "complete")
    )
    run = ReasonRun(
        reason_id=reason_id,
        client_id=client.client_id,
        forecast_run_id=run_id,
        created_at=datetime.now(UTC),
        as_of=as_of,
        status=status,
        inputs_hash=inputs_hash,
        profile_hash=profile_hash(profile),
        prompt_version=PROMPT_VERSION,
        provider=llm.provider,
        model=llm.model,
        scenario=f"{scenario.name} ({scenario.note})" if scenario else None,
        months=view.months,
        summary=" ".join(summaries),
        families=families,
        drivers=states,
        findings=findings,
        quiet=quiet,
        not_assessed=not_assessed,
        references=sorted(references.values(), key=lambda r: (r.id[0], int(r.id[1:]))),
        bands=band_rows,
        calls=caller.calls,
        gaps=list(dict.fromkeys(gaps)),
        unassigned_series=len(view.unassigned),
        estimate={**estimate, "sources": stats},
    )
    if not options.dry_run:
        records.save_run(store, run)
    return run


def apply_override(client: ClientProfile, store: EngineStore, override: Override) -> ReasonRun:
    """Planner accepts, edits or dismisses a finding: bands are recomputed, no model call."""
    require_reasoning(client)
    run = records.load_run(store, override.reason_id)
    if run is None:
        raise ReasoningError(f"Reasoning run {override.reason_id} not found")
    if override.finding_id not in {f.finding_id for f in run.findings}:
        raise ReasoningError(f"{override.finding_id} is not a finding in {run.reason_id}")
    records.save_override(store, override)
    profile = load_profile(client)
    view = load_forecast(client, store, profile, run.forecast_run_id)
    original = [
        f.model_copy(
            update={
                "flags": [
                    x
                    for x in f.flags
                    if not x.startswith(("dismissed", "edited_by_", "accepted_by_"))
                ]
            }
        )
        for f in run.findings
    ]
    findings = effective(original, records.overrides(store, run.reason_id))
    band_rows, families, _ = bands(view, findings)
    updated = run.model_copy(
        update={"findings": findings, "bands": band_rows, "families": families}
    )
    records.save_run(store, updated)
    return updated


__all__ = ["Options", "apply_override", "recordings_path", "require_reasoning", "run_review"]
