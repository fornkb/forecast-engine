"""Offline planner reports from saved review and forecast records."""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.table import Table
from rich.text import Text

from engine.config.base import StrictModel
from engine.reason import store as records
from engine.reason.collect import load_evidence_batch
from engine.reason.context import get_forecast_snapshot
from engine.reason.review import load_review
from engine.reason.schemas import (
    Assessment,
    CompanyFact,
    ContextError,
    EvidenceBatch,
    EvidenceRevision,
    ForecastSnapshot,
    ReviewManifest,
    SourceCheck,
)
from engine.store.duckdb_store import EngineStore


class ReviewReport(StrictModel):
    schema_version: str = "forecast-review-report/1"
    manifest: ReviewManifest
    base_batch: EvidenceBatch
    forecast: ForecastSnapshot
    assessments: tuple[Assessment, ...]
    source_checks: tuple[SourceCheck, ...]
    evidence: tuple[EvidenceRevision, ...]
    facts: tuple[CompanyFact, ...]


def load_report(store: EngineStore, review_id: str) -> ReviewReport:
    manifest = load_review(store, review_id)
    base = load_evidence_batch(store, manifest.base_batch_id)
    forecast = get_forecast_snapshot(store, manifest.snapshot_id)
    evidence_ids = tuple(dict.fromkeys(base.evidence_ids + manifest.appended_evidence_ids))
    check_ids = tuple(dict.fromkeys(base.check_ids + manifest.appended_check_ids))
    fact_ids = sorted(
        {key for row in forecast.rows for key in row.fact_version_ids}
        | set(manifest.appended_fact_version_ids)
    )
    with store.connect() as con:
        assessments = records.read_assessments(con, store.client_id, review_id)
        checks = records.read_check_ids(con, store.client_id, check_ids)
        evidence = records.read_evidence(con, store.client_id, evidence_ids)
        facts = tuple(records.get_fact(con, store.client_id, key) for key in fact_ids)
    missing = (
        *(f"check:{key}" for key in sorted(set(check_ids) - {item.check_id for item in checks})),
        *(
            f"evidence:{key}"
            for key in sorted(set(evidence_ids) - {item.evidence_id for item in evidence})
        ),
        *(f"fact:{key}" for key, fact in zip(fact_ids, facts, strict=True) if fact is None),
    )
    if missing:
        raise ContextError("Missing pinned revisions " + ", ".join(missing))
    for row in assessments:
        if row.manifest_revision != manifest.revision:
            raise ContextError(
                f"Assessment {row.scenario_id}/{row.series_id} has a mismatched revision"
            )
    return ReviewReport(
        manifest=manifest,
        base_batch=base,
        forecast=forecast,
        assessments=assessments,
        source_checks=checks,
        evidence=evidence,
        facts=facts,
    )


def _amount(value, unit, *, signed=False) -> str:
    if value is None:
        return "unknown"
    amount = f"{value:+g}" if signed else f"{value:g}"
    return f"{amount} {unit or 'unit unknown'}"


def print_summary(report: ReviewReport, console: Console | None = None) -> None:
    console = console or Console()
    manifest = report.manifest
    console.print(
        Text(
            f"Review {manifest.review_id} · {manifest.state} · forecast {manifest.forecast_run_id}"
        )
    )
    console.print(
        Text(
            f"Knowledge cutoff {manifest.knowledge_cutoff.isoformat()} · "
            f"{len(manifest.lookup_attempts)}/{manifest.followup_limit} follow-up attempts · "
            f"{len(manifest.model_calls)}/{manifest.model_request_limit} model requests"
        )
    )
    counts = Counter(item.finding for item in report.assessments)
    console.print(
        Text(
            f"{len(report.assessments)} findings · "
            + (
                ", ".join(
                    f"{key.replace('_', ' ')} {value}" for key, value in sorted(counts.items())
                )
                or "no verified row findings"
            )
        )
    )
    pending = sum(item.status == "pending" for item in manifest.questions)
    console.print(
        Text(f"{pending} pending questions · {len(manifest.gaps)} coverage or review gaps")
    )
    if report.assessments:
        table = Table("Series", "Month", "Target", "Original forecast", "Finding", "Conditions")
        for row in report.assessments[:8]:
            conditions = (
                "\n".join(
                    f"{item.description[:80]} · {_amount(item.endpoint, row.unit)} "
                    f"({_amount(item.delta, row.unit, signed=True)})"
                    for item in row.conditions
                )
                or "unknown"
            )
            table.add_row(
                Text(row.series_id),
                row.period.strftime("%Y-%m"),
                Text(row.target),
                _amount(row.forecast_baseline, row.forecast_unit),
                row.finding.replace("_", " "),
                Text(conditions),
            )
        console.print(table)
    for gap in manifest.gaps[:5]:
        console.print(Text(f"  {gap}"))


def _plain(value) -> str:
    clean = " ".join(str(value).split())
    return (
        clean.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def render_markdown(report: ReviewReport) -> str:
    manifest = report.manifest
    calls = manifest.model_calls
    lines = [
        "# Monthly forecast review",
        "",
        f"Client **{_plain(manifest.client_id)}**, forecast run **{manifest.forecast_run_id}**",
        f"Review `{manifest.review_id}` · revision {manifest.revision} · {manifest.state}",
        f"Base batch `{manifest.base_batch_id}`",
        f"Collection cutoff {manifest.collection_cutoff.isoformat()}",
        f"Review knowledge cutoff {manifest.knowledge_cutoff.isoformat()}",
        f"Follow-up attempts {len(manifest.lookup_attempts)}/{manifest.followup_limit} · "
        f"model requests {len(calls)}/{manifest.model_request_limit} · "
        f"input tokens {sum(call.input_tokens for call in calls)} · "
        f"output tokens {sum(call.output_tokens for call in calls)}",
    ]
    if manifest.successor_of:
        lines.append(f"Successor of `{manifest.successor_of}`")
    lines.extend(["", "## Monthly findings", ""])
    assumptions = {item.assumption_id: item for item in manifest.assumptions}
    for row in report.assessments:
        product = _plain(row.item_id or row.series_id)
        customer = _plain(row.customer_id or "unknown customer")
        lines.extend(
            [
                f"### {product} · {customer} · {row.period:%Y-%m}",
                "",
                f"Series {_plain(row.series_id)} · target {_plain(row.target)} · "
                f"horizon {row.horizon} · model {_plain(row.model)} · "
                f"original forecast {_amount(row.forecast_baseline, row.forecast_unit)}",
                f"**{row.finding.replace('_', ' ')}** · direction {row.direction}",
                _plain(row.rationale),
            ]
        )
        if row.supply_at or row.supply_shortfall is not None:
            lines.append(
                f"Supply shortfall {_amount(row.supply_shortfall, row.supply_unit)} "
                f"at {row.supply_at.isoformat() if row.supply_at else 'unknown date'}"
            )
        if row.supply_balance is not None:
            lines.append(
                f"Projected material balance at the site "
                f"{_amount(row.supply_balance, row.supply_unit)}"
            )
        if row.method_id:
            lines.append(f"Method `{row.method_id}` · `{row.method_version}`")
        if row.assumption_id:
            source = assumptions[row.assumption_id]
            lines.append(
                f"Conditional planner assumption by **{_plain(source.actor)}**, "
                f"confirmed {source.confirmed_at.isoformat()}. {_plain(source.values.rationale)}"
            )
        if not row.conditions:
            lines.append("Scenario quantities unknown.")
        for condition in row.conditions:
            lines.extend(
                [
                    "",
                    f"- {_plain(condition.description)} · "
                    f"**{_amount(condition.endpoint, row.unit)}** "
                    f"({_amount(condition.delta, row.unit, signed=True)}) "
                    f"· condition `{_plain(condition.condition_id)}`",
                ]
            )
            for operand in condition.operands:
                lines.append(
                    f"  - {_plain(operand.name)} {_amount(operand.value, operand.unit)} · "
                    f"`{operand.record_version_id}` · {_plain(operand.basis)}"
                )
            if condition.basis_record_ids:
                lines.append(
                    "  - Condition references "
                    + ", ".join(f"`{key}`" for key in condition.basis_record_ids)
                )
        citations = (*row.evidence_ids, *row.fact_version_ids)
        if citations:
            lines.extend(["", "Citations " + ", ".join(f"`{key}`" for key in citations)])
        if row.contrary_ids:
            lines.append("Contrary records " + ", ".join(f"`{key}`" for key in row.contrary_ids))
        if row.required_check_ids:
            lines.append(
                "Required checks " + ", ".join(f"`{key}`" for key in row.required_check_ids)
            )
        if row.gaps:
            lines.extend(["", *[f"- {_plain(gap)}" for gap in row.gaps]])
        lines.append("")
    covered = {(row.series_id, row.period) for row in report.assessments}
    missing = [row for row in report.forecast.rows if (row.series_id, row.period) not in covered]
    if missing:
        lines.extend(["### Rows without a verified finding", ""])
        for row in missing:
            lines.append(
                f"- {_plain(row.series_id)} · {row.period:%Y-%m} · "
                f"original forecast {_amount(row.baseline, row.unit)}"
            )
    lines.extend(["", "## News mapping decisions", ""])
    for mapping in manifest.mappings:
        lines.append(f"- **{mapping.status}** · `{mapping.event_id}` · {_plain(mapping.basis)}")
        lines.append("  - Articles " + ", ".join(f"`{key}`" for key in mapping.article_ids))
        if mapping.entities:
            lines.append(
                "  - Entities "
                + ", ".join(f"{_plain(item.type)} {_plain(item.id)}" for item in mapping.entities)
            )
        if mapping.months:
            lines.append(
                "  - Months " + ", ".join(month.strftime("%Y-%m") for month in mapping.months)
            )
        references = (*mapping.fact_version_ids, *mapping.evidence_ids, *mapping.question_ids)
        if references:
            lines.append("  - References " + ", ".join(f"`{key}`" for key in references))
    lines.extend(["", "## Operator questions", ""])
    for question in manifest.questions:
        lines.append(
            f"- **{question.status}** · `{question.question_id}` · {_plain(question.message)}"
        )
        if question.scope:
            for series, baseline in question.baselines:
                lines.append(
                    f"  - {_plain(series)} · {question.scope.period:%Y-%m} · "
                    f"{_amount(baseline, question.scope.unit)}"
                )
    lines.extend(["", "## Coverage and source checks", ""])
    lines.extend(f"- {_plain(gap)}" for gap in manifest.gaps)
    for check in report.source_checks:
        lines.append(
            f"- `{check.check_id}` · {_plain(check.scope.source_id)} · "
            f"{check.status} · {check.coverage} · {check.ended_at.isoformat()}"
            + (f" · {_plain(check.reason)}" if check.reason else "")
        )
        for month in check.months:
            lines.append(
                f"  - {month.period:%Y-%m} · {month.status}"
                + (f" · {_plain(month.reason)}" if month.reason else "")
            )
    lines.extend(["", "## Evidence and company facts", ""])
    for item in report.evidence:
        title = item.news.title if item.news else item.source_id
        lines.append(
            f"- `{item.evidence_id}` · {_plain(title)} · "
            f"available {item.first_known_at.isoformat()} · {_plain(item.source_ref)}"
        )
    for fact in report.facts:
        lines.append(
            f"- `{fact.fact_version_id}` · {_plain(fact.subject.id)} "
            f"{_plain(fact.relation)} · {_plain(fact.source_ref)}"
        )
    return "\n".join(lines).rstrip() + "\n"
