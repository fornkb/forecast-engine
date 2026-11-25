"""Operator commands for the saved forecast context."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from duckdb import Error as DatabaseError
from pydantic import ValidationError

from engine.config.schema import ClientProfile
from engine.reason.admin import propose_clarification, resolve_review_question
from engine.reason.collect import collect_evidence
from engine.reason.context import load_forecast_snapshot
from engine.reason.erp import require_reasoning
from engine.reason.evaluation import evaluate_reviews
from engine.reason.facts import sync_context
from engine.reason.questions import confirm_question, list_questions
from engine.reason.replay import replay_review
from engine.reason.report import load_report, print_summary, render_markdown
from engine.reason.review import load_review, run_review
from engine.reason.schemas import (
    AssumptionInput,
    ContextError,
    ContextSyncError,
    ForecastSnapshot,
    SyncResult,
)
from engine.store.duckdb_store import EngineStore

ClientOption = Annotated[str, typer.Option("--client", "-c", help="Client profile id.")]
PREVIEW_LIMIT = 8


def collect_command(
    profile: ClientProfile,
    store: EngineStore,
    run_id: int,
    snapshot_id: str | None,
    resume_id: str | None,
    as_of: datetime | None,
    as_json: bool,
) -> None:
    with _operator_errors():
        batch = collect_evidence(
            profile,
            store,
            run_id,
            snapshot_id=snapshot_id,
            resume_id=resume_id,
            as_of=as_of,
            on_start=lambda value: typer.echo(f"Collection ID {value}", err=True),
        )
    output = {
        "batch_id": batch.batch_id,
        "collection_id": batch.collection_id,
        "snapshot_id": batch.snapshot_id,
        "knowledge_cutoff": batch.knowledge_cutoff.isoformat(),
        "complete": batch.complete,
        "check_count": len(batch.check_ids),
        "evidence_count": len(batch.evidence_ids),
        "group_count": len(batch.groups),
        "candidate_count": len(batch.candidates),
        "gap_count": len(batch.gaps),
        "gaps": list(batch.gaps[:PREVIEW_LIMIT]),
        "status_counts": dict(batch.status_counts),
        "usage": [item.model_dump(mode="json") for item in batch.usage],
    }
    if as_json:
        typer.echo(json.dumps(output))
        return
    typer.echo(f"Evidence batch {batch.batch_id}")
    typer.echo(
        f"{output['check_count']} checks, {output['evidence_count']} evidence revisions, "
        f"{output['group_count']} exposure groups, {output['candidate_count']} news candidates"
    )
    for gap in output["gaps"]:
        typer.echo(f"  {gap}")
    if len(batch.gaps) > PREVIEW_LIMIT:
        typer.echo(f"  {len(batch.gaps) - PREVIEW_LIMIT} more gaps in the saved batch")


@contextmanager
def _operator_errors() -> Iterator[None]:
    try:
        yield
    except ContextSyncError as exc:
        _print_sync(SyncResult(invalid=exc.invalid))
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except (ContextError, DatabaseError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


def _print_sync(result: SyncResult) -> None:
    typer.echo(" ".join(f"{key}={value}" for key, value in result.model_dump().items()))


def _inspect(snapshot: ForecastSnapshot, series: str | None, as_json: bool) -> None:
    matching = sum(1 for _ in snapshot.iter_rows(series))
    gaps = tuple(gap for gap in snapshot.gaps if series is None or gap.series_id in (None, series))
    counts = dict(sorted(Counter(gap.code for gap in gaps).items()))
    units = sorted({row.unit for row in snapshot.iter_rows(series) if row.unit})
    preview = list(islice(snapshot.iter_rows(series), PREVIEW_LIMIT))
    if as_json:
        output = snapshot.model_dump(mode="json", exclude={"rows", "gaps"})
        output.update(
            row_count=len(snapshot.rows),
            matched_row_count=matching,
            units=units,
            gap_counts=counts,
            gaps=[gap.model_dump(mode="json") for gap in gaps[:PREVIEW_LIMIT]],
            preview=[row.model_dump(mode="json") for row in preview],
        )
        typer.echo(json.dumps(output))
        return

    typer.echo(f"Snapshot {snapshot.snapshot_id} · forecast run {snapshot.forecast_run_id}")
    target = (
        f"{snapshot.target.id} — {snapshot.target.definition}" if snapshot.target else "unknown"
    )
    typer.echo(
        f"Target {target} | Units {', '.join(units) or 'unknown'} "
        f"| Rows {matching}/{len(snapshot.rows)}"
    )
    typer.echo("Gaps " + (", ".join(f"{code}={count}" for code, count in counts.items()) or "none"))
    for gap in gaps[:PREVIEW_LIMIT]:
        scope = f"{gap.series_id} {gap.period} · " if gap.series_id else ""
        typer.echo(f"  {scope}{gap.message}")
    typer.echo(f"First {len(preview)} of {matching} matching rows")
    for row in preview:
        typer.echo(
            f"  {row.series_id}  {row.period}  h={row.horizon}  "
            f"{row.model}  {row.baseline:g} {row.unit or ''}"
        )


def create_app(load: Callable[[str], tuple[ClientProfile, EngineStore]]) -> typer.Typer:
    app = typer.Typer(no_args_is_help=True, help="Inspect saved forecast and company context.")

    @app.command()
    def sync(client: ClientOption) -> None:
        """Import configured company facts in one batch."""
        with _operator_errors():
            profile, store = load(client)
            _print_sync(sync_context(profile, store))

    @app.command()
    def inspect(
        client: ClientOption,
        run_id: Annotated[int, typer.Option("--run-id", min=1)],
        series: Annotated[
            str | None, typer.Option("--series", help="Filter the preview by series ID.")
        ] = None,
        as_json: Annotated[
            bool, typer.Option("--json", help="Print a machine-readable summary.")
        ] = False,
    ) -> None:
        """Pin and inspect an explicit saved monthly forecast run."""
        with _operator_errors():
            profile, store = load(client)
            _inspect(load_forecast_snapshot(profile, store, run_id), series, as_json)

    @app.command()
    def questions(client: ClientOption) -> None:
        """List pending onboarding and conflict questions."""
        with _operator_errors():
            profile, store = load(client)
            require_reasoning(profile)
            pending = list_questions(store)
            typer.echo(f"Pending questions {len(pending)}")
            for question in pending:
                spec = question.spec
                months = ", ".join(month.isoformat() for month in spec.affected_months)
                typer.echo(
                    f"  {question.question_id} · {spec.subject.type} {spec.subject.id} "
                    f"· {spec.message}"
                )
                typer.echo(
                    f"    Answer type {spec.answer_type} | Target months {months or 'unspecified'}"
                )
                if question.answer:
                    typer.echo(f"    Previous answer {question.answer.answer_id}")

    @app.command()
    def confirm(
        client: ClientOption,
        question_id: Annotated[str, typer.Option("--question-id")],
        value: Annotated[str, typer.Option("--value")],
        actor: Annotated[str, typer.Option("--actor", help="Local operator's audit label.")],
        supersedes: Annotated[str | None, typer.Option("--supersedes-answer-id")] = None,
    ) -> None:
        """Record a typed operator answer to a context question."""
        with _operator_errors():
            profile, store = load(client)
            answer = confirm_question(
                profile,
                store,
                question_id,
                value,
                actor,
                supersedes_answer_id=supersedes,
            )
            typer.echo(
                f"Confirmed {question_id} · answer {answer.answer_id} · revision {answer.revision}"
            )

    return app


class AdminAction(StrEnum):
    LIST = "list"
    CONFIRM = "confirm"
    EDIT = "edit"
    CANCEL = "cancel"
    PROPOSE = "propose"


class AssumptionMethod(StrEnum):
    INCREMENT = "incremental_quantity"
    PERCENT = "proportional_change"


def register_review_commands(
    app: typer.Typer, load: Callable[[str], tuple[ClientProfile, EngineStore]]
):
    @app.command("reason-replay")
    def reason_replay(
        client: ClientOption,
        review_id: Annotated[str, typer.Option("--review-id")],
        as_json: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Verify and reproduce a finalized review offline."""
        with _operator_errors():
            _, store = load(client)
            result = replay_review(store, review_id)
            if as_json:
                typer.echo(result.model_dump_json(indent=2))
            else:
                outcome = "Reproduced" if result.reproduced else "Replay incomplete"
                typer.echo(
                    f"{outcome} · {review_id} · {len(result.stored_assessments)} saved findings"
                )
                typer.echo(f"Knowledge cutoff {result.manifest.knowledge_cutoff.isoformat()}")
                typer.echo(
                    "Integrity seal " + ("verified" if result.seal_verified else "unverified")
                )
                for value in (*result.issues, *result.limitations)[:PREVIEW_LIMIT]:
                    typer.echo(f"  {value}")
            if not result.reproduced:
                raise typer.Exit(1)

    @app.command("reason-evaluate")
    def reason_evaluate(
        client: ClientOption,
        review_ids: Annotated[list[str], typer.Option("--review-id")],
        outcomes: Annotated[Path, typer.Option("--outcomes", dir_okay=False)],
        group_labels: Annotated[Path | None, typer.Option("--group-labels", dir_okay=False)] = None,
        triage_labels: Annotated[
            Path | None, typer.Option("--triage-labels", dir_okay=False)
        ] = None,
        as_of: Annotated[str | None, typer.Option("--as-of")] = None,
        as_json: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Compare frozen review decisions against dated outcomes."""
        with _operator_errors():
            profile, store = load(client)
            try:
                cutoff = datetime.fromisoformat(as_of) if as_of is not None else None
            except ValueError as exc:
                raise ContextError("Provide an ISO evaluation cutoff with a UTC offset") from exc
            result = evaluate_reviews(
                profile,
                store,
                tuple(review_ids),
                outcomes,
                as_of=cutoff,
                group_labels_path=group_labels,
                triage_labels_path=triage_labels,
            )
            if as_json:
                typer.echo(result.model_dump_json(indent=2))
                return
            typer.echo(
                f"{len(result.review_ids)} reviews · {result.aggregate_unique_rows} unique monthly "
                f"decisions by window · {result.repeated_rows_removed} repeated decisions omitted"
            )
            for comparison in result.point_comparisons:
                if comparison.review_id is not None:
                    continue
                errors = ", ".join(
                    f"{score.comparator} MAE {score.mae:.4g}"
                    + (f" WAPE {score.wape:.2%}" if score.wape is not None else " WAPE unknown")
                    if score.mae is not None
                    else f"{score.comparator} unavailable"
                    for score in comparison.errors
                )
                comparison_label = comparison.comparison
                if comparison.provenance:
                    comparison_label += f" · {comparison.provenance}"
                typer.echo(
                    f"{comparison.phase} · {comparison.unit or 'unit unknown'} "
                    f"· {comparison.target} · {comparison_label} "
                    f"· {comparison.n}/{comparison.candidate_rows} rows · {errors}"
                )
                if comparison.exclusions:
                    typer.echo(
                        "  "
                        + ", ".join(
                            f"{name}={count}" for name, count in comparison.exclusions.items()
                        )
                    )
            if result.conditional_ranges:
                resolved = sum(item.resolved for item in result.conditional_ranges)
                unresolved = sum(item.unresolved for item in result.conditional_ranges)
                typer.echo(f"{resolved} realized conditions · {unresolved} unresolved ranges")
            for usage in result.aggregate_usage:
                typer.echo(
                    f"{usage.stage} · {usage.provider} {usage.transport} · "
                    f"{usage.attempts} attempts · {usage.retries} retries "
                    f"· {usage.cache_hits} cache hits"
                )

    @app.command("reason")
    def reason(
        client: ClientOption,
        run_id: Annotated[int, typer.Option("--run-id", min=1)],
        batch_id: Annotated[str | None, typer.Option("--batch-id")] = None,
        followup_limit: Annotated[
            int | None, typer.Option("--follow-up-limit", min=0, max=3)
        ] = None,
        resume_id: Annotated[str | None, typer.Option("--resume-id")] = None,
        new_review: Annotated[
            bool, typer.Option("--new-review", help="Start a fresh review budget.")
        ] = False,
        as_json: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Review one stored monthly forecast and its evidence."""
        with _operator_errors():
            profile, store = load(client)
            review = run_review(
                profile,
                store,
                run_id,
                batch_id=batch_id,
                followup_limit=followup_limit,
                resume_id=resume_id,
                new_review=new_review,
                on_start=lambda identifier: typer.echo(f"Review ID {identifier}", err=True),
                on_collection_start=lambda identifier: typer.echo(
                    f"Collection ID {identifier}", err=True
                ),
            )
            if as_json:
                typer.echo(review.model_dump_json())
            else:
                print_summary(load_report(store, review.review_id))

    @app.command("reason-report")
    def reason_report(
        client: ClientOption,
        review_id: Annotated[str, typer.Option("--review-id")],
        as_json: Annotated[bool, typer.Option("--json")] = False,
        markdown: Annotated[bool, typer.Option("--markdown")] = False,
        output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    ) -> None:
        """Read a saved review offline or export its full JSON or Markdown."""
        if (as_json and markdown) or (output is not None and not (as_json or markdown)):
            raise typer.BadParameter("Select one export format with --json or --markdown")
        with _operator_errors():
            _, store = load(client)
            report = load_report(store, review_id)
            if not (as_json or markdown):
                print_summary(report)
                return
            document = (
                report.model_dump_json(indent=2) + "\n" if as_json else render_markdown(report)
            )
            if output is None:
                typer.echo(document, nl=False)
            else:
                try:
                    output.write_text(document, encoding="utf-8")
                except OSError as exc:
                    raise ContextError(
                        f"Could not write report to {output}: {exc.strerror}"
                    ) from exc

    @app.command("reason-admin")
    def reason_admin(
        client: ClientOption,
        review_id: Annotated[str, typer.Option("--review-id")],
        action: Annotated[AdminAction, typer.Option("--action")] = AdminAction.LIST,
        question_id: Annotated[str | None, typer.Option("--question-id")] = None,
        actor: Annotated[str | None, typer.Option("--actor")] = None,
        value: Annotated[str | None, typer.Option("--value")] = None,
        text: Annotated[str | None, typer.Option("--text")] = None,
        method: Annotated[AssumptionMethod | None, typer.Option("--method")] = None,
        low: Annotated[float | None, typer.Option("--low")] = None,
        high: Annotated[float | None, typer.Option("--high")] = None,
        low_condition: Annotated[str | None, typer.Option("--low-condition")] = None,
        high_condition: Annotated[str | None, typer.Option("--high-condition")] = None,
        rationale: Annotated[str | None, typer.Option("--rationale")] = None,
        unit: Annotated[str | None, typer.Option("--unit")] = None,
        series: Annotated[list[str] | None, typer.Option("--series")] = None,
        as_json: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """List, confirm, edit, or cancel review questions."""
        with _operator_errors():
            profile, store = load(client)
            if action == AdminAction.LIST:
                review = load_review(store, review_id)
                if as_json:
                    typer.echo(json.dumps([q.model_dump(mode="json") for q in review.questions]))
                else:
                    typer.echo(f"Review {review.review_id}")
                    for question in review.questions:
                        typer.echo(
                            f"{question.status} · {question.question_id} · {question.message}"
                        )
                        if question.scope:
                            scope = question.scope
                            typer.echo(f"  {scope.period:%Y-%m} · {scope.target} · {scope.unit}")
                            for name, baseline in question.baselines:
                                typer.echo(f"  {name}: {baseline:g} {scope.unit}")
                return
            if not question_id:
                raise ContextError("Provide --question-id")
            if action == AdminAction.PROPOSE:
                if not text:
                    raise ContextError("Provide the operator answer with --text")
                identifier, proposal = propose_clarification(
                    profile, store, review_id, question_id, text
                )
                typer.echo(json.dumps({"review_id": identifier, "proposed_value": proposal}))
                return
            if not actor:
                raise ContextError("Provide --actor")
            fields = dict(
                method_id=method.value if method else None,
                low=low,
                high=high,
                low_condition=low_condition,
                high_condition=high_condition,
                rationale=rationale,
                unit=unit,
                series_ids=series,
            )
            assumption = None
            if any(item is not None for item in fields.values()):
                try:
                    assumption = AssumptionInput.model_validate(fields)
                except ValidationError as exc:
                    raise ContextError(
                        "Provide --method, finite --low and --high, both conditions, "
                        "--rationale, --unit and one --series for each included series"
                    ) from exc
            review = resolve_review_question(
                profile,
                store,
                review_id,
                question_id,
                action.value,
                actor,
                value=value,
                assumption=assumption,
            )
            report = load_report(store, review.review_id)
            if as_json:
                typer.echo(report.model_dump_json())
            else:
                print_summary(report)
