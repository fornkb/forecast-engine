"""`engine reason*` commands. Thin wrappers: the same functions serve a future ERP portal."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from engine.reason import store as records
from engine.reason.llm import LLMError, get_client
from engine.reason.llm.cost import cost
from engine.reason.profile import (
    apply_answers,
    draft_profile,
    load_profile,
    profile_path,
    save_profile,
)
from engine.reason.report import render_html, render_markdown, totals
from engine.reason.run import (
    Options,
    apply_override,
    recordings_path,
    require_reasoning,
    run_review,
)
from engine.reason.schemas import LLMCall, Override, ReasoningError, ReasonRun

console = Console()
ClientOption = Annotated[str, typer.Option("--client", "-c", help="Client profile id.")]


class LLMMode(StrEnum):
    live = "live"
    record = "record"
    replay = "replay"


class Action(StrEnum):
    accept = "accept"
    dismiss = "dismiss"
    edit = "edit"


def _fail(exc: Exception) -> typer.Exit:
    console.print(f"[red]{exc}[/red]")
    return typer.Exit(code=1)


def print_run(run: ReasonRun) -> None:
    used = totals(run)
    status = {
        "changed": "[green]changed[/green]",
        "unchanged": "unchanged",
        "no_data": "[yellow]no data[/yellow]",
        "error": "[red]error[/red]",
    }
    table = Table("Driver", "Status", "Signals", title=f"Drivers · run {run.reason_id}")
    for state in run.drivers:
        detail = "; ".join(
            (s.summary if s.ok else f"{s.kind}: {s.error}")[:90] for s in state.signals
        )
        table.add_row(state.driver_id, status[state.status], detail or "–")
    console.print(table)
    if run.status == "dry_run":
        estimate = run.estimate
        console.print(
            f"[bold]Dry run[/bold] · would make {estimate.get('calls', 0)} AI call(s) · "
            f"~{estimate.get('input_tokens', 0):,} input tokens · cost guess "
            f"{estimate.get('cost_guess') if estimate.get('cost_guess') is not None else 'unknown'}"
        )
        return
    table = Table(
        "ID", "Driver", "Family", "Channel", "Effect", "Months", "Evidence", title="Findings"
    )
    for f in run.findings:
        effect = "–" if f.low_pct == f.high_pct == 0 else f"{f.low_pct:+g}%..{f.high_pct:+g}%"
        table.add_row(
            f.finding_id,
            f.driver_id,
            f.family_id,
            f.channel,
            effect,
            ", ".join(m.strftime("%Y-%m") for m in f.months),
            f.tier + (" (carried)" if f.carried_from else ""),
        )
    console.print(table)
    if run.summary:
        console.print(f"[bold]Summary:[/bold] {run.summary}")
    for gap in [*run.not_assessed, *run.gaps]:
        console.print(f"[yellow]gap:[/yellow] {gap}")
    cost = "unknown" if used["cost"] is None else f"{used['cost']:.4f} {used['currency']}"
    console.print(
        f"{run.status} · {used['calls']} AI call(s) · {used['input_tokens']:,} in / "
        f"{used['output_tokens']:,} out tokens · cost {cost}"
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    console.print(f"Wrote {path}")


def register_reason_commands(app: typer.Typer, load: Callable) -> None:
    @app.command("reason")
    def reason(
        client: ClientOption,
        run_id: Annotated[
            int | None, typer.Option("--run-id", help="Forecast run; default latest.")
        ] = None,
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="Collect signals, estimate cost, no AI call.")
        ] = False,
        full: Annotated[
            bool, typer.Option("--full", help="Reassess every driver, ignore the last run.")
        ] = False,
        llm: Annotated[
            LLMMode, typer.Option("--llm", help="live, record (save answers) or replay (offline).")
        ] = LLMMode.live,
        scenario: Annotated[
            Path | None,
            typer.Option("--scenario", help="Scenario pack JSON instead of live sources."),
        ] = None,
        as_of: Annotated[
            str | None, typer.Option("--as-of", help="Review date YYYY-MM-DD.")
        ] = None,
        html: Annotated[
            Path | None, typer.Option("--html", help="Also write the HTML report here.")
        ] = None,
        as_json: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Review a saved monthly forecast against external factors (0-3 AI calls)."""
        profile, store = load(client)
        try:
            run = run_review(
                profile,
                store,
                run_id,
                Options(
                    llm_mode=llm.value,
                    scenario=scenario,
                    dry_run=dry_run,
                    full=full,
                    as_of=date.fromisoformat(as_of) if as_of else None,
                ),
            )
        except (ReasoningError, ValueError) as exc:
            raise _fail(exc) from exc
        if as_json:
            typer.echo(run.model_dump_json(indent=2))
        else:
            print_run(run)
        if html and not dry_run:
            _write(html, render_html(run))

    @app.command("reason-report")
    def reason_report(
        client: ClientOption,
        reason_id: Annotated[
            str | None, typer.Option("--reason-id", help="Default: latest run.")
        ] = None,
        html: Annotated[Path | None, typer.Option("--html")] = None,
        markdown: Annotated[Path | None, typer.Option("--markdown")] = None,
        as_json: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Reopen a saved run offline (no AI call)."""
        _, store = load(client)
        run = records.load_run(store, reason_id) if reason_id else records.latest_run(store)
        if run is None:
            raise _fail(ReasoningError("No saved reasoning run. Run `engine reason` first."))
        if as_json:
            typer.echo(run.model_dump_json(indent=2))
        elif not (html or markdown):
            print_run(run)
        if html:
            _write(html, render_html(run))
        if markdown:
            _write(markdown, render_markdown(run))

    @app.command("reason-adjust")
    def reason_adjust(
        client: ClientOption,
        reason_id: Annotated[str, typer.Option("--reason-id")],
        finding: Annotated[str, typer.Option("--finding", help="Finding id, e.g. F2.")],
        action: Annotated[Action, typer.Option("--action")],
        by: Annotated[str, typer.Option("--by", help="Planner's name for the audit trail.")],
        low: Annotated[float | None, typer.Option("--low", help="Edited low %.")] = None,
        high: Annotated[float | None, typer.Option("--high", help="Edited high %.")] = None,
        note: Annotated[str, typer.Option("--note")] = "",
    ) -> None:
        """Accept, edit or dismiss a finding. Bands are recomputed with no AI call."""
        profile, store = load(client)
        if action is Action.edit and (low is None or high is None):
            raise _fail(ReasoningError("--action edit needs --low and --high"))
        try:
            run = apply_override(
                profile,
                store,
                Override(
                    reason_id=reason_id,
                    finding_id=finding,
                    action=action.value,
                    low_pct=low,
                    high_pct=high,
                    actor=by,
                    note=note,
                    at=datetime.now(UTC),
                ),
            )
        except ReasoningError as exc:
            raise _fail(exc) from exc
        print_run(run)

    @app.command("reason-onboard")
    def reason_onboard(
        client: ClientOption,
        about: Annotated[
            Path | None, typer.Option("--about", help="Text file describing the company.")
        ] = None,
        answers: Annotated[
            Path | None, typer.Option("--answers", help="Admin answers to the draft's questions.")
        ] = None,
        confirm: Annotated[
            bool, typer.Option("--confirm", help="Mark the profile confirmed.")
        ] = False,
        actor: Annotated[
            str, typer.Option("--by", help="Admin's name for the audit trail.")
        ] = "admin",
        llm: Annotated[LLMMode, typer.Option("--llm")] = LLMMode.live,
    ) -> None:
        """Draft the company profile from ERP data, or apply the admin's answers (1 AI call)."""
        profile, store = load(client)
        try:
            config = require_reasoning(profile)
            if not config.llm.model:
                raise ReasoningError("Set reasoning.llm.model in the client profile.")
            model = get_client(config.llm, llm.value, recordings_path(profile))
            started = datetime.now(UTC)
            if answers:
                current = load_profile(profile)
                draft, result = apply_answers(
                    profile,
                    store,
                    model,
                    config.llm.model,
                    config.llm.max_output_tokens,
                    current,
                    answers.read_text(encoding="utf-8"),
                )
                stage = "onboard_answers"
            else:
                if profile_path(profile).exists() and not typer.confirm(
                    f"{profile_path(profile)} exists. Replace it with a new draft?", default=False
                ):
                    raise typer.Exit()
                draft, result = draft_profile(
                    profile,
                    store,
                    model,
                    config.llm.model,
                    config.llm.max_output_tokens,
                    about.read_text(encoding="utf-8") if about else "",
                )
                stage = "onboard"
        except (ReasoningError, LLMError) as exc:
            raise _fail(exc) from exc
        records.save_call(
            store,
            None,
            LLMCall(
                call_id=result.request_hash[:12] or "onboard",
                stage=stage,
                provider=result.provider,
                model=result.model,
                started_at=started,
                seconds=round((datetime.now(UTC) - started).total_seconds(), 2),
                usage=result.usage,
                cost=0.0
                if result.replayed
                else cost(result.usage, config.llm.rates.get(result.model)),
                outcome="replayed" if result.replayed else "ok",
                request_hash=result.request_hash,
            ),
        )
        if confirm:
            draft = draft.model_copy(update={"status": "confirmed"})
        path = save_profile(profile, store, draft, actor)
        console.print(
            f"Saved {draft.status} profile to {path}: {len(draft.families)} families, "
            f"{len(draft.drivers)} drivers."
        )
        if draft.notes:
            console.print(f"[yellow]Checks:[/yellow] {draft.notes}")
        if draft.open_questions:
            console.print(
                "\n[bold]Questions for the admin[/bold] (answer them in a text file, then "
                f"run `engine reason-onboard --client {client} --answers FILE`):"
            )
            for number, question in enumerate(draft.open_questions, 1):
                console.print(f"  {number}. {question}")

    @app.command("reason-cost")
    def reason_cost(
        client: ClientOption, as_json: Annotated[bool, typer.Option("--json")] = False
    ) -> None:
        """AI calls, tokens and cost recorded for this client."""
        _, store = load(client)
        ledger = records.call_ledger(store)
        if as_json:
            typer.echo(json.dumps(ledger, default=str, indent=2))
            return
        table = Table(
            "When",
            "Run",
            "Stage",
            "Model",
            "In",
            "Out",
            "Cached",
            "Cost",
            "Outcome",
            title=f"AI calls · {client}",
        )
        for row in ledger:
            table.add_row(
                f"{row['started_at']:%Y-%m-%d %H:%M}",
                row["reason_id"] or "-",
                row["stage"],
                row["model"],
                f"{row['input_tokens']:,}",
                f"{row['output_tokens']:,}",
                f"{row['cached_input_tokens']:,}",
                "?" if row["cost"] is None else f"{row['cost']:.4f}",
                row["outcome"],
            )
        console.print(table)
        known = [row["cost"] for row in ledger if row["cost"] is not None]
        console.print(
            f"{len(ledger)} calls · known cost {sum(known):.4f} "
            f"({len(ledger) - len(known)} without a configured rate)"
        )
