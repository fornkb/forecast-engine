"""Recorded provider usage and configured-rate estimates for frozen reviews."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal

from engine.config.base import StrictModel
from engine.reason.report import ReviewReport
from engine.reason.review_inputs import latest_checks
from engine.reason.schemas import ContextError


class UsageTotal(StrictModel):
    stage: str
    provider: str
    transport: str
    logical_scopes: int
    attempts: int = 0
    retries: int = 0
    retry_kind_unknown: int = 0
    continuations: int = 0
    cache_hits: int = 0
    successful_results: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    token_usage_unreported: int = 0
    estimated_cost: float | None = None
    currency: str | None = None
    rate_ids: tuple[str, ...] = ()


class SourceCoverage(StrictModel):
    provider: str
    source_id: str
    statuses: dict[str, int]
    coverage: dict[str, int]
    gaps: dict[str, int]


class ReviewUsage(StrictModel):
    review_id: str
    elapsed_seconds: float
    providers: tuple[UsageTotal, ...]
    coverage: tuple[SourceCoverage, ...]
    findings: dict[str, int]
    checked_groups: int
    relevant_groups: int
    quantified_groups: int
    reproducible_findings: int
    reproducible_calculation_share: float | None
    planner_assumption_ranges: int
    distinct_assumptions_used: int
    assumptions_per_quantified_finding: float | None
    evidence_bound_ranges: int
    directional_alerts: int
    unquantified_findings: int
    questions: dict[str, int]
    followup_limit: int
    followup_attempts: int
    admin_followups: int
    confirmed_mappings: int
    cancelled_mappings: int


def _transport(report, check) -> str:
    if check.scope.source_kind != "erp":
        return "http"
    source = check.scope.source_id
    if source in report.manifest.source_access:
        return report.manifest.source_access[source]
    observed = {
        item.source_ref.split(":", 1)[0]
        for item in report.evidence
        if item.source_kind == "erp" and item.source_id == source
    }
    return next(iter(observed)) if len(observed) == 1 else "unknown_erp"


def _cost(rate, attempts, *, call=None):
    if not attempts:
        return Decimal(0), None, None
    if rate is None:
        return None, None, None
    total = Decimal(str(rate.per_request or 0)) * attempts
    if call is None:
        if rate.per_request is None:
            return None, rate.currency, rate.rate_id
    else:
        counts = (
            (call.input_tokens, rate.per_million_input),
            (call.output_tokens, rate.per_million_output),
            (call.cache_read_input_tokens, rate.per_million_cache_read),
            (call.cache_creation_input_tokens, rate.per_million_cache_creation),
        )
        if any(count and amount is None for count, amount in counts) and any(
            amount is not None for _, amount in counts
        ):
            return None, rate.currency, rate.rate_id
        if not any(amount is not None for _, amount in counts) and rate.per_request is None:
            return None, rate.currency, rate.rate_id
        reported = call.token_usage_reported is True or (
            call.token_usage_reported is None and any(count for count, _ in counts)
        )
        if any(amount is not None for _, amount in counts) and not reported:
            return None, rate.currency, rate.rate_id
        total += sum(
            (
                Decimal(str(amount)) * count / 1_000_000
                for count, amount in counts
                if amount is not None
            ),
            Decimal(0),
        )
    return total, rate.currency, rate.rate_id


def provider_usage(reports: tuple[ReviewReport, ...]) -> tuple[UsageTotal, ...]:
    if len({report.manifest.client_id for report in reports}) > 1:
        raise ContextError("Aggregate usage requires reviews from one client")
    buckets = defaultdict(list)
    seen = set()
    for report in sorted(
        reports, key=lambda item: (item.manifest.started_at, item.manifest.review_id)
    ):
        m = report.manifest
        base_ids = set(report.base_batch.check_ids)
        for check in report.source_checks:
            if check.check_id in seen:
                continue
            seen.add(check.check_id)
            base = check.check_id in base_ids
            stage = "collection" if base else "followup"
            owner = report.base_batch.collection_id if base else m.budget_owner_id
            transport = _transport(report, check)
            rate = m.provider_rates.get(check.provider) if transport != "csv" else None
            counts = Counter(
                attempts=check.attempts, retries=check.retries, cache_hits=check.cache_hits
            )
            if check.status in {"found", "no_match"} and (check.attempts or check.cache_hits):
                counts["successful_results"] = 1
            buckets[stage, check.provider, transport].append(
                ((owner, check.scope.query_hash), counts, _cost(rate, check.attempts))
            )
        for call in m.model_calls:
            if call.call_id in seen:
                continue
            seen.add(call.call_id)
            counts = Counter(
                attempts=1,
                retries=int(call.request_kind == "retry"),
                retry_kind_unknown=int(call.request_kind is None),
                continuations=int(call.request_kind == "continuation"),
                successful_results=int(call.outcome in {"accepted", "tool_result"}),
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                cache_read_input_tokens=call.cache_read_input_tokens,
                cache_creation_input_tokens=call.cache_creation_input_tokens,
                token_usage_unreported=int(call.token_usage_reported is not True),
            )
            cost = _cost(m.provider_rates.get("anthropic"), 1, call=call)
            buckets["assistant", "anthropic", "http"].append(
                ((m.budget_owner_id, call.work_id), counts, cost)
            )
    totals = []
    for (stage, provider, transport), group in sorted(buckets.items()):
        counts = sum((item[1] for item in group), Counter())
        amounts = [item[2] for item in group]
        currencies = {currency for _, currency, _ in amounts if currency}
        rates = {rate for _, _, rate in amounts if rate}
        total = None
        if len(currencies) == 1 and all(value is not None for value, _, _ in amounts):
            total = float(sum(value for value, _, _ in amounts))
        totals.append(
            UsageTotal(
                stage=stage,
                provider=provider,
                transport=transport,
                logical_scopes=len({item[0] for item in group}),
                estimated_cost=total,
                currency=next(iter(currencies)) if len(currencies) == 1 else None,
                rate_ids=tuple(sorted(rates)),
                **counts,
            )
        )
    return tuple(totals)


def review_usage(report: ReviewReport, *, reproduced: bool) -> ReviewUsage:
    m = report.manifest
    checks_by_source = defaultdict(list)
    for check in latest_checks(report.source_checks):
        checks_by_source[check.provider, check.scope.source_id].append(check)
    coverage = tuple(
        SourceCoverage(
            provider=provider,
            source_id=source,
            statuses=dict(Counter(c.status for c in checks)),
            coverage=dict(Counter(c.coverage for c in checks)),
            gaps=dict(
                Counter(
                    c.reason or c.status
                    for c in checks
                    if c.coverage != "complete" and c.status != "not_applicable"
                )
            ),
        )
        for (provider, source), checks in sorted(checks_by_source.items())
    )
    ranges = [row for row in report.assessments if row.conditions]
    assumptions = {row.assumption_id for row in ranges if row.assumption_id}
    relevant = [
        row for row in report.assessments if row.finding in {"possible_impact", "what_if_range"}
    ]
    return ReviewUsage(
        review_id=m.review_id,
        elapsed_seconds=(m.knowledge_cutoff - m.started_at).total_seconds(),
        providers=provider_usage((report,)),
        coverage=coverage,
        findings=dict(Counter(row.finding for row in report.assessments)),
        checked_groups=len(m.groups),
        relevant_groups=len({row.group_id for row in relevant}),
        quantified_groups=len({row.group_id for row in ranges}),
        reproducible_findings=len(ranges) if reproduced else 0,
        reproducible_calculation_share=(
            (len(ranges) if reproduced else 0) / len(report.assessments)
        )
        if report.assessments
        else None,
        planner_assumption_ranges=sum(row.assumption_id is not None for row in ranges),
        distinct_assumptions_used=len(assumptions),
        assumptions_per_quantified_finding=len(assumptions) / len(ranges) if ranges else None,
        evidence_bound_ranges=sum(row.assumption_id is None for row in ranges),
        directional_alerts=sum(
            row.direction in {"increase", "decrease", "mixed"} for row in relevant
        ),
        unquantified_findings=sum(not row.conditions for row in relevant),
        questions=dict(Counter(f"{q.purpose}:{q.status}" for q in m.questions)),
        admin_followups=len(m.questions),
        followup_limit=m.followup_limit,
        followup_attempts=len(m.lookup_attempts),
        confirmed_mappings=sum(
            q.purpose == "mapping_confirmation" and q.status == "confirmed" for q in m.questions
        ),
        cancelled_mappings=sum(
            q.purpose == "mapping_confirmation" and q.status == "cancelled" for q in m.questions
        ),
    )
