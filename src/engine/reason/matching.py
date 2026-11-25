"""Associate eligible forecast rows with confirmed ERP and regional evidence."""

from __future__ import annotations

from collections import defaultdict, deque

from engine.reason.facts import overlaps
from engine.reason.schemas import (
    CandidateAssociation,
    CompanyFact,
    EntityRef,
    EvidenceRevision,
    ExposureGroup,
    ForecastRow,
    ForecastSnapshot,
    SourceCheck,
    content_digest,
)
from engine.reason.supply import member


def row_facts(
    snapshot: ForecastSnapshot,
    facts: dict[str, CompanyFact],
    row: ForecastRow,
) -> dict[EntityRef, tuple[str, ...]]:
    """Use a continuous effective path from the row to each linked entity."""
    blocked = set(row.superseded_fact_version_ids)
    for gap in snapshot.gaps:
        if gap.code == "conflicting_fact" and (gap.series_id, gap.period) == (
            row.series_id,
            row.period,
        ):
            blocked.update(gap.fact_version_ids)
    links = defaultdict(list)
    for version in row.fact_version_ids:
        fact = facts.get(version)
        if version not in blocked and fact and fact.object:
            links[fact.subject].append(fact)
    roots = [
        EntityRef(type="series", id=row.series_id),
        EntityRef(type="company", id=row.client_id),
    ]
    roots.extend(
        EntityRef(type=kind, id=value)
        for kind, value in (("item", row.item_id), ("customer", row.customer_id))
        if value
    )
    queue = deque((root, row.target_start, row.target_end, ()) for root in roots)
    visited = set()
    result = defaultdict(set)
    while queue and len(visited) < 256:
        subject, start, end, path = queue.popleft()
        key = (subject, start, end)
        if key in visited:
            continue
        visited.add(key)
        for fact in links[subject]:
            if not overlaps(fact, start, end):
                continue
            valid_from = max(start, fact.effective_from)
            valid_to = min(end, fact.effective_to or end)
            versions = (*path, fact.fact_version_id)
            result[fact.object].update(versions)
            queue.append((fact.object, valid_from, valid_to, versions))
    return {
        **{root: () for root in roots},
        **{key: tuple(sorted(ids)) for key, ids in result.items()},
    }


def weather_links(
    linked: dict[EntityRef, tuple[str, ...]],
    facts: dict[str, CompanyFact],
) -> dict[tuple[str, str], tuple[str, ...]]:
    result = defaultdict(set)
    for entity, ids in linked.items():
        if entity.type != "location":
            continue
        for version in ids:
            fact = facts[version]
            if fact.object == entity and fact.location_role:
                result[entity.id, fact.location_role].update(ids)
    return {key: tuple(sorted(ids)) for key, ids in result.items()}


def operating_links(
    snapshot: ForecastSnapshot, evidence: tuple[EvidenceRevision, ...], locations: dict
):
    """Index mapped item dependencies and their effective vendor/route links."""
    by_item, vendors = defaultdict(list), defaultdict(list)
    for revision in evidence:
        op = revision.operational
        if op.dataset in {"bom", "vendor_link"} and op.item_id:
            by_item[op.item_id].append(revision)
        if op.dataset == "vendor_link":
            vendors[op.material_id, op.site_id].append(revision)
    by_row = {}
    for row in snapshot.rows:
        key = (row.series_id, row.period)
        sites = {location for location, role in locations[key] if role == "site"}
        matched = defaultdict(set)
        for revision in by_item[row.item_id]:
            op = revision.operational
            if (
                op.customer_id not in (None, row.customer_id)
                or op.series_id not in (None, row.series_id)
                or (sites and op.site_id not in sites)
                or (op.dataset == "bom" and row.unit and row.unit != op.finished_unit)
            ):
                continue
            start = max(row.target_start, revision.valid_from)
            end = min(row.target_end, revision.valid_to or row.target_end)
            if start >= end:
                continue
            matched[EntityRef(type="material", id=op.material_id)].add(revision.evidence_id)
            matched[EntityRef(type="site", id=op.site_id)].add(revision.evidence_id)
            for kind, identifier in (("supplier", op.supplier_id), ("route", op.route_id)):
                if identifier:
                    matched[EntityRef(type=kind, id=identifier)].add(revision.evidence_id)
            for vendor in vendors[op.material_id, op.site_id]:
                details = vendor.operational
                if (
                    details.item_id not in (None, row.item_id)
                    or details.customer_id not in (None, row.customer_id)
                    or details.series_id not in (None, row.series_id)
                    or vendor.valid_from >= end
                    or (vendor.valid_to and vendor.valid_to <= start)
                ):
                    continue
                for kind, identifier in (
                    ("supplier", details.supplier_id),
                    ("route", details.route_id),
                ):
                    if identifier:
                        matched[EntityRef(type=kind, id=identifier)].update(
                            (revision.evidence_id, vendor.evidence_id)
                        )
        by_row[key] = matched
    return by_row


def candidate_articles(
    evidence: tuple[EvidenceRevision, ...],
    checks: tuple[SourceCheck, ...],
) -> tuple[CandidateAssociation, ...]:
    articles = [item for item in evidence if item.news]
    by_candidate = defaultdict(list)
    for check in checks:
        if check.scope.source_kind == "news":
            for evidence_id in check.evidence_ids:
                by_candidate[evidence_id].append(check)
    by_record = defaultdict(list)
    for article in articles:
        by_record[article.record_id].append(article)
    by_content = defaultdict(list)
    for versions in by_record.values():
        latest = max(versions, key=lambda item: item.revision)
        by_content[latest.news.duplicate_key].append(latest)
    canonical, collapsed = [], {}
    for matching in by_content.values():
        selected = min(matching, key=lambda item: (not item.news.primary, item.source_ref))
        canonical.append(selected)
        collapsed[selected.evidence_id] = tuple(
            sorted(
                version.evidence_id
                for item in matching
                for version in by_record[item.record_id]
                if version.evidence_id != selected.evidence_id
            )
        )
    events, titles = defaultdict(list), defaultdict(list)
    for article in canonical:
        if article.source_event_id:
            events[article.source_event_id].append(article)
        if article.news.provider_page_date:
            titles[article.news.title.casefold(), article.news.provider_page_date].append(article)
    candidates = []
    for article in sorted(canonical, key=lambda item: (not item.news.primary, item.source_ref)):
        duplicates = collapsed[article.evidence_id]
        retrieved = [
            check for key in (article.evidence_id, *duplicates) for check in by_candidate[key]
        ]
        entities = {entity for check in retrieved for entity in check.scope.entity_ids}
        corroborating = {
            other.evidence_id
            for other in (
                *events.get(article.source_event_id, ()),
                *titles.get((article.news.title.casefold(), article.news.provider_page_date), ()),
            )
            if other.news.publisher != article.news.publisher
        }
        candidates.append(
            CandidateAssociation(
                evidence_id=article.evidence_id,
                check_ids=tuple(sorted({check.check_id for check in retrieved})),
                entities=tuple(sorted(entities, key=lambda entity: (entity.type, entity.id))),
                basis="public_query_alias" if entities else "broad_search_unresolved",
                corroborating_ids=tuple(sorted(corroborating)),
                duplicate_ids=duplicates,
            )
        )
    return tuple(candidates)


def weather_groups(
    snapshot: ForecastSnapshot,
    checks: tuple[SourceCheck, ...],
    evidence: dict[str, EvidenceRevision],
    locations: dict,
    operating: dict,
) -> list[ExposureGroup]:
    groups = []
    for check in checks:
        scope = check.scope
        if scope.source_kind != "weather" or check.status != "found":
            continue
        location = scope.entity_ids[0]
        for month in check.months:
            period = month.period
            ids = tuple(
                key
                for key in check.evidence_ids
                if key in evidence
                and evidence[key].weather
                and evidence[key].weather.period == period
            )
            if not ids:
                continue
            selected = [
                row
                for row in snapshot.rows
                if row.period == period
                and (location.id, scope.location_role) in locations[row.series_id, period]
            ]
            mapping = EntityRef(type=scope.location_role, id=location.id)
            operating_ids = {
                key for row in selected for key in operating[row.series_id, period].get(mapping, ())
            }
            groups.append(
                ExposureGroup(
                    group_id="group_"
                    + content_digest(
                        [snapshot.client_id, "weather", location.id, scope.location_role, period]
                    ),
                    kind="weather",
                    dependency=location,
                    period=period,
                    location_id=location.id,
                    location_role=scope.location_role,
                    evidence_ids=tuple(sorted(set(ids) | operating_ids)),
                    fact_version_ids=tuple(
                        sorted(
                            {
                                version
                                for row in selected
                                for version in locations[row.series_id, period][
                                    location.id, scope.location_role
                                ]
                            }
                        )
                    ),
                    members=tuple(member(row) for row in selected),
                    flags=("partial_regional_outlook",) if month.status != "complete" else (),
                )
            )
    return groups


def demand_groups(
    snapshot: ForecastSnapshot,
    evidence: tuple[EvidenceRevision, ...],
) -> list[ExposureGroup]:
    groups = []
    for revision in evidence:
        record = revision.operational
        if record.dataset not in {"order_amendment", "contract_option"}:
            continue
        by_period = defaultdict(list)
        for row in snapshot.rows:
            if row.item_id != record.item_id or row.customer_id != record.customer_id:
                continue
            if not row.target_start <= revision.valid_from < row.target_end:
                continue
            if record.series_id is None or row.series_id == record.series_id:
                by_period[row.period].append(row)
        for period, rows in by_period.items():
            compatible = [row for row in rows if row.unit == revision.unit]
            flags = []
            if len(compatible) != len(rows):
                flags.append("incompatible_unit")
            if len(compatible) > 1:
                flags.append("ambiguous_customer_item_mapping")
            groups.append(
                ExposureGroup(
                    group_id="group_" + content_digest([revision.record_id, period]),
                    kind="customer_order"
                    if record.dataset == "order_amendment"
                    else "contract_option",
                    dependency=EntityRef(type="customer", id=record.customer_id),
                    period=period,
                    evidence_ids=(revision.evidence_id,),
                    members=tuple(member(row) for row in compatible),
                    exceptions=tuple(member(row) for row in rows if row not in compatible),
                    flags=tuple(flags),
                    event_at=revision.valid_from,
                    unit=revision.unit,
                )
            )
    return groups
