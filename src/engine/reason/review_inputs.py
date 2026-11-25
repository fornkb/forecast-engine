"""Build scoped group packets from a review's immutable inputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from engine.reason import store as records
from engine.reason.collect import _context_facts
from engine.reason.config import TargetConfig
from engine.reason.facts import overlaps
from engine.reason.matching import (
    candidate_articles,
    demand_groups,
    operating_links,
    row_facts,
    weather_groups,
    weather_links,
)
from engine.reason.schemas import (
    CompanyFact,
    ContextError,
    EvidenceBatch,
    EvidenceRevision,
    ExposureGroup,
    ForecastSnapshot,
    ReviewManifest,
    SourceCheck,
    content_digest,
)
from engine.reason.supply import active_operational, supply_groups

_SUPPLY_DATASETS = {
    "bom",
    "vendor_link",
    "usable_stock",
    "consumption",
    "purchase_order",
    "receipt",
    "allocation",
    "alternate",
}


def latest_checks(checks: tuple[SourceCheck, ...]) -> tuple[SourceCheck, ...]:
    # Cache hits and repeated denials preserve history without changing identical model inputs.
    selected = {}
    for check in checks:
        scope = check.scope
        key = scope.source_kind, scope.source_id, scope.location_role, scope.months, scope.page
        old = selected.get(key)
        if old is None or _check_content(old) != _check_content(check):
            selected[key] = check
    return tuple(selected.values())


def _check_content(check: SourceCheck) -> tuple:
    return (
        check.status,
        check.coverage,
        check.reason,
        check.evidence_ids,
        check.months,
        check.more_pages,
        check.scope.filters,
    )


def _effective_snapshot(
    snapshot: ForecastSnapshot,
    additions: tuple[CompanyFact, ...],
) -> ForecastSnapshot:
    target = snapshot.target
    for fact in additions:
        if fact.relation == "target" and isinstance(fact.value, TargetConfig):
            target = fact.value
    rows = []
    for row in snapshot.rows:
        applicable = tuple(
            fact for fact in additions if overlaps(fact, row.target_start, row.target_end)
        )
        overrides = {
            identifier
            for fact in applicable
            for identifier in fact.resolves_fact_ids
            if fact.effective_from <= row.target_start
            and (fact.effective_to is None or fact.effective_to >= row.target_end)
        }
        unit = row.unit
        for fact in applicable:
            if (
                fact.relation == "unit"
                and fact.subject.type == "series"
                and fact.subject.id == row.series_id
                and isinstance(fact.value, str)
            ):
                unit = fact.value
        rows.append(
            row.model_copy(
                update={
                    "unit": unit,
                    "fact_version_ids": tuple(
                        dict.fromkeys(
                            (*row.fact_version_ids, *(fact.fact_version_id for fact in applicable))
                        )
                    ),
                    "superseded_fact_version_ids": tuple(
                        sorted(set(row.superseded_fact_version_ids) | overrides)
                    ),
                }
            )
        )
    return snapshot.model_copy(update={"rows": tuple(rows), "target": target})


@dataclass(frozen=True)
class ReviewInputs:
    snapshot: ForecastSnapshot
    evidence: dict[str, EvidenceRevision]
    facts: dict[str, CompanyFact]
    active_fact_ids: frozenset[str]
    checks: tuple[SourceCheck, ...]
    links: dict
    operating: dict
    groups: tuple[ExposureGroup, ...]

    @property
    def candidates(self):
        return candidate_articles(tuple(self.evidence.values()), self.checks)


def _order_events(groups, evidence, active):
    combined = defaultdict(list)
    for group in groups:
        item = evidence[group.evidence_ids[0]]
        op = item.operational
        event = item.source_event_id or item.record_id
        combined[event, op.customer_id, op.item_id, group.period].append(group)
    result = []
    for (event, customer, item_id, period), matching in sorted(combined.items()):
        ids = tuple(sorted({key for group in matching for key in group.evidence_ids}))
        flags = {flag for group in matching for flag in group.flags}
        items = [evidence[key] for key in ids]
        roles = [(revision.operational.dataset, revision.operational.status) for revision in items]
        competing = {
            earlier.evidence_id
            for revision in items
            for earlier in evidence.values()
            if earlier.record_id == revision.record_id
            and earlier.source_version
            and earlier.source_version == revision.source_version
            and earlier.evidence_id not in active
            and (earlier.quantity, earlier.operational, earlier.valid_from, earlier.valid_to)
            != (revision.quantity, revision.operational, revision.valid_from, revision.valid_to)
        }
        memberships = {frozenset(group.members) for group in matching}
        if competing or len(roles) != len(set(roles)) or len(memberships) > 1:
            flags.add("conflicting_order_evidence")
        group = matching[0]
        result.append(
            group.model_copy(
                update={
                    "group_id": "group_"
                    + content_digest(
                        [items[0].client_id, "order", event, customer, item_id, period]
                    ),
                    "kind": "order_event",
                    "event_ids": ("order_event_" + content_digest([items[0].client_id, event]),),
                    "evidence_ids": (*ids, *sorted(competing)),
                    "members": tuple(
                        sorted(
                            {member for part in matching for member in part.members},
                            key=lambda member: member.series_id,
                        )
                    ),
                    "exceptions": tuple(
                        sorted(
                            {member for part in matching for member in part.exceptions},
                            key=lambda member: member.series_id,
                        )
                    ),
                    "flags": tuple(sorted(flags)),
                }
            )
        )
    return tuple(result)


def order_groups(snapshot, evidence):
    operational = active_operational(tuple(evidence.values()))
    return _order_events(
        demand_groups(snapshot, operational),
        evidence,
        {item.evidence_id for item in operational},
    )


def pinned_inputs(snapshot, manifest, revisions, facts, checks) -> ReviewInputs:
    """Prepare effective rows from only the exact facts and evidence in this review."""
    cutoff = manifest.knowledge_cutoff
    for item in revisions:
        if item.first_known_at > cutoff or item.captured_at > cutoff:
            raise ContextError(f"Evidence {item.evidence_id} was recorded after the review cutoff")
    additions = tuple(facts[identifier] for identifier in manifest.appended_fact_version_ids)
    for fact in additions:
        if fact.known_at > cutoff or fact.captured_at > cutoff:
            raise ContextError(
                f"Company fact {fact.fact_version_id} was recorded after the review cutoff"
            )
    current = {fact.record_id: fact for fact in additions}
    active_fact_ids = frozenset(facts) - {fact.fact_version_id for fact in additions} | {
        fact.fact_version_id for fact in current.values() if fact.status != "withdrawn"
    }
    return ReviewInputs(
        _effective_snapshot(snapshot, tuple(current.values())),
        {revision.evidence_id: revision for revision in revisions},
        facts,
        active_fact_ids,
        checks,
        {},
        {},
        (),
    )


def load_inputs(
    store,
    profile,
    snapshot: ForecastSnapshot,
    batch: EvidenceBatch,
    manifest: ReviewManifest,
) -> ReviewInputs:
    facts = _context_facts(store, snapshot)
    evidence_ids = tuple(dict.fromkeys(batch.evidence_ids + manifest.appended_evidence_ids))
    check_ids = tuple(dict.fromkeys(batch.check_ids + manifest.appended_check_ids))
    with store.connect() as con:
        additions = tuple(
            records.get_fact(con, store.client_id, identifier)
            for identifier in manifest.appended_fact_version_ids
        )
        checks = records.read_check_ids(con, store.client_id, check_ids)
        revisions = records.read_evidence(con, store.client_id, evidence_ids)
    if (
        len(facts) != len({key for row in snapshot.rows for key in row.fact_version_ids})
        or len(revisions) != len(evidence_ids)
        or len(checks) != len(check_ids)
        or any(fact is None for fact in additions)
    ):
        raise ContextError("Review refers to missing pinned inputs")
    facts.update({fact.fact_version_id: fact for fact in additions})
    pinned = pinned_inputs(snapshot, manifest, revisions, facts, checks)
    adjusted, evidence = pinned.snapshot, pinned.evidence
    operational = active_operational(revisions)
    links = {(row.series_id, row.period): row_facts(adjusted, facts, row) for row in adjusted.rows}
    locations = {key: weather_links(value, facts) for key, value in links.items()}
    operating = operating_links(adjusted, operational, locations)
    for key, entities in operating.items():
        for entity in entities:
            if entity.type in {"site", "route"}:
                locations[key].setdefault((entity.id, entity.type), ())
    terminal = latest_checks(checks)
    live_sources = {
        check.scope.source_id
        for check in terminal
        if check.scope.source_kind == "erp" and check.coverage == "complete"
    }
    sources = [
        source for source in profile.reasoning.evidence.erp if source.dataset in _SUPPLY_DATASETS
    ]
    datasets = {source.dataset for source in sources if source.id in live_sources}
    sufficient = (
        {"usable_stock", "consumption"} <= datasets
        and bool({"bom", "vendor_link"} & datasets)
        and bool({"receipt", "purchase_order"} & datasets)
        and all(source.id in live_sources for source in sources)
    )
    supply_weather = (
        *supply_groups(adjusted, operational, complete_inputs=sufficient, locations=locations),
        *weather_groups(adjusted, terminal, evidence, locations, operating),
    )
    by_id = {}
    for group in supply_weather:
        old = by_id.get(group.group_id)
        if old:
            group = group.model_copy(
                update={
                    "evidence_ids": tuple(sorted(set(old.evidence_ids + group.evidence_ids))),
                    "flags": tuple(sorted(set(old.flags + group.flags))),
                }
            )
        by_id[group.group_id] = group.model_copy(update={"event_ids": (group.group_id,)})
    orders = _order_events(
        demand_groups(adjusted, operational),
        evidence,
        {item.evidence_id for item in operational},
    )
    return replace(pinned, links=links, operating=operating, groups=(*by_id.values(), *orders))


def compact_evidence(item: EvidenceRevision) -> dict:
    data = {
        "id": item.evidence_id,
        "source_id": item.source_id,
        "source_ref": item.source_ref[:400],
        "source_version": item.source_version,
        "available_at": item.first_known_at.isoformat(),
        "valid_from": item.valid_from.isoformat() if item.valid_from else None,
        "valid_to": item.valid_to.isoformat() if item.valid_to else None,
        "quantity": item.quantity,
        "unit": item.unit,
    }
    if item.operational:
        data["operational"] = item.operational.model_dump(mode="json", exclude_none=True)
    if item.news:
        data.update(
            title=item.news.title,
            publisher=item.news.publisher,
            excerpt=(item.excerpt or "")[:650],
            location_id=item.location_id,
        )
    if item.weather:
        data["weather"] = item.weather.model_dump(mode="json", exclude={"members"})
        data["weather"]["returned_members"] = len(item.weather.members)
    if item.enso:
        data["enso"] = item.enso.model_dump(mode="json")
    return {key: value for key, value in data.items() if value is not None}


def compact_fact(fact: CompanyFact) -> dict:
    return fact.model_dump(
        mode="json",
        exclude_none=True,
        exclude={
            "content_digest",
            "client_id",
            "record_id",
            "captured_at",
            "source_version",
            "source_ref",
        },
    )


def group_packet(inputs: ReviewInputs, group: ExposureGroup):
    rows = [
        row
        for row in inputs.snapshot.rows
        if row.period == group.period
        and row.series_id in {member.series_id for member in group.members}
    ]
    fact_ids = set(group.fact_version_ids)
    for row in rows:
        fact_ids.update(row.fact_version_ids)
    facts = {key: inputs.facts[key] for key in sorted(fact_ids) if key in inputs.active_fact_ids}
    evidence = {key: inputs.evidence[key] for key in group.evidence_ids}
    for item in active_operational(tuple(inputs.evidence.values())):
        op = item.operational
        if op.dataset != "method_parameter" or not rows:
            continue
        if any(
            (
                op.series_id == row.series_id
                or (
                    op.series_id is None
                    and (op.item_id, op.customer_id) == (row.item_id, row.customer_id)
                    and row.customer_id
                    and row.item_id
                )
            )
            for row in rows
        ):
            evidence[item.evidence_id] = item
    if group.kind == "weather":
        for item in inputs.evidence.values():
            if item.enso and any(
                season.start <= group.period <= season.end for season in item.enso.seasons
            ):
                evidence[item.evidence_id] = item
    source_ids = {item.source_id for item in evidence.values()}
    checks = {
        check.check_id: check
        for check in latest_checks(inputs.checks)
        if group.period in check.scope.months
        and (
            bool(set(check.evidence_ids) & evidence.keys())
            or check.scope.source_id in source_ids
            or (group.kind == "news" and check.scope.source_kind == "news")
            or (group.kind == "weather" and check.scope.source_id == group.location_id)
        )
    }
    packet = {
        "group": group.model_dump(mode="json"),
        "rows": [
            row.model_dump(mode="json", exclude={"fact_version_ids", "superseded_fact_version_ids"})
            for row in rows
        ],
        "target": inputs.snapshot.target.model_dump(mode="json")
        if inputs.snapshot.target
        else None,
        "evidence": [compact_evidence(item) for item in evidence.values()],
        "facts": [compact_fact(fact) for fact in facts.values()],
        "checks": [
            dict(
                id=check.check_id,
                kind=check.scope.source_kind,
                source=check.scope.source_id,
                status=check.status,
                coverage=check.coverage,
                reason=check.reason,
                months=[
                    m.model_dump(mode="json") for m in check.months if m.period == group.period
                ],
            )
            for check in checks.values()
        ],
    }
    return packet, evidence, facts, checks
