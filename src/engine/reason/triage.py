"""Validate shared news mappings and resolve dated exposure memberships."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from engine.reason.facts import overlaps
from engine.reason.schemas import (
    CandidateAssociation,
    CompanyFact,
    ContextError,
    EntityRef,
    EvidenceRevision,
    ExposureGroup,
    ForecastRow,
    MappingDecision,
    MappingProposal,
    QuestionSpec,
    ReviewQuestion,
    content_digest,
)
from engine.reason.supply import member

_ENTITY_TYPES = {
    "company",
    "series",
    "item",
    "customer",
    "material",
    "supplier",
    "route",
    "location",
    "site",
    "market",
    "dependency",
}


@dataclass(frozen=True)
class NewsEvent:
    event_id: str
    candidates: tuple[CandidateAssociation, ...]

    @property
    def article_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.candidates)


def news_events(candidates, evidence: dict[str, EvidenceRevision]) -> tuple[NewsEvent, ...]:
    by_event = defaultdict(list)
    for candidate in candidates:
        revision = evidence[candidate.evidence_id]
        event_key = revision.source_event_id or revision.news.duplicate_key
        key = "news_event_" + content_digest([revision.client_id, revision.source_id, event_key])
        by_event[key].append(candidate)
    return tuple(NewsEvent(key, tuple(items)) for key, items in sorted(by_event.items()))


def applies(revision: EvidenceRevision, row: ForecastRow) -> bool:
    return (revision.valid_from is None or revision.valid_from < row.target_end) and (
        revision.valid_to is None or revision.valid_to > row.target_start
    )


def _event_overlaps_path(event, row, evidence, windows) -> bool:
    start = max((row.target_start, *(left for left, _ in windows)))
    end = min((row.target_end, *(right or row.target_end for _, right in windows)))
    return start < end and any(
        (evidence[key].valid_from is None or evidence[key].valid_from < end)
        and (evidence[key].valid_to is None or evidence[key].valid_to > start)
        for key in event.article_ids
    )


def mapping_spec(event: NewsEvent, entity: EntityRef, month: date, basis: str) -> QuestionSpec:
    return QuestionSpec(
        subject=entity,
        field=f"applies_to:{event.event_id}:{month.isoformat()}",
        purpose="mapping_confirmation",
        message=(
            f"Confirm a forecast series exposed to {entity.type} {entity.id} in {month}. {basis}"
        ),
        answer_type="object_id",
        object_type="series",
        affected_months=(month,),
        evidence_ids=event.article_ids,
        event_ids=(event.event_id,),
    )


def validate_triage(
    proposals: tuple[MappingProposal, ...],
    events: tuple[NewsEvent, ...],
    rows: tuple[ForecastRow, ...],
    evidence_ids: Iterable[str],
    fact_ids: Iterable[str],
) -> None:
    indexed = {event.event_id: event for event in events}
    if len(proposals) != len(indexed) or {item.event_id for item in proposals} != indexed.keys():
        raise ContextError("triage_requires_one_decision_for_each_event")
    months = {row.period for row in rows}
    evidence_ids, fact_ids = set(evidence_ids), set(fact_ids)
    for item in proposals:
        expected = indexed[item.event_id]
        if (
            len(item.article_ids) != len(expected.article_ids)
            or set(item.article_ids) != set(expected.article_ids)
            or set(item.evidence_ids) - evidence_ids
            or set(item.fact_version_ids) - fact_ids
        ):
            raise ContextError("triage_unknown_reference")
        if len(set(item.months)) != len(item.months) or set(item.months) - months:
            raise ContextError("triage_month_outside_forecast")
        if any(
            entity.type not in _ENTITY_TYPES or not entity.id.strip() or len(entity.id) > 160
            for entity in item.entities
        ):
            raise ContextError("triage_invalid_entity")
        if len(set(item.entities)) != len(item.entities):
            raise ContextError("triage_duplicate_entity")
        if item.status == "established" and (not item.entities or not item.months):
            raise ContextError("triage_missing_entity_or_month")


def map_event(
    proposal: MappingProposal,
    event: NewsEvent,
    rows: tuple[ForecastRow, ...],
    evidence: dict[str, EvidenceRevision],
    facts: dict[str, CompanyFact],
    links: dict,
    operating: dict,
    questions: tuple[ReviewQuestion, ...],
    active_fact_ids: frozenset[str],
) -> tuple[MappingDecision, tuple[ExposureGroup, ...], tuple[QuestionSpec, ...]]:
    """A search hit and a proposed entity never serve as a company relationship."""
    groups, unresolved = [], []
    used_facts, used_evidence = set(), set()
    requested = getattr(proposal, "requested_status", proposal.status)
    if requested != "irrelevant":
        for month in proposal.months:
            for entity in proposal.entities:
                spec = mapping_spec(event, entity, month, proposal.basis)
                confirmed = {
                    item.fact_version_id
                    for item in questions
                    if item.purpose == "mapping_confirmation"
                    and item.status == "confirmed"
                    and item.subject == entity
                    and item.months == (month,)
                    and item.evidence_ids == event.article_ids
                    and item.fact_version_id in active_fact_ids
                }
                selected, exceptions, fact_ids, evidence_ids = [], [], set(), set()
                partial = False
                for row in rows:
                    key = row.series_id, row.period
                    if row.period != month or not any(
                        applies(evidence[identifier], row) for identifier in event.article_ids
                    ):
                        continue
                    allowed_facts = set(proposal.fact_version_ids)
                    allowed_evidence = set(proposal.evidence_ids)
                    fact_path = set(links[key].get(entity, ()))
                    op_path = set(operating[key].get(entity, ()))
                    fact_path = fact_path if fact_path <= allowed_facts else set()
                    op_path = op_path if op_path <= allowed_evidence else set()
                    anchors = {
                        identifier
                        for identifier in confirmed
                        if facts[identifier].object == EntityRef(type="series", id=row.series_id)
                        and overlaps(facts[identifier], row.target_start, row.target_end)
                    }
                    if requested != "established" or not _event_overlaps_path(
                        event,
                        row,
                        evidence,
                        tuple(
                            (facts[key].effective_from, facts[key].effective_to)
                            for key in fact_path
                        ),
                    ):
                        fact_path = set()
                    if requested != "established" or not _event_overlaps_path(
                        event,
                        row,
                        evidence,
                        tuple(
                            (evidence[key].valid_from, evidence[key].valid_to) for key in op_path
                        ),
                    ):
                        op_path = set()
                    if not (anchors or fact_path or op_path):
                        exceptions.append(member(row))
                        continue
                    selected.append(member(row))
                    fact_ids.update(fact_path | anchors)
                    evidence_ids.update(op_path)
                    partial |= (
                        any(
                            facts[key].effective_from > row.target_start
                            or (
                                facts[key].effective_to and facts[key].effective_to < row.target_end
                            )
                            for key in fact_path | anchors
                        )
                        or any(
                            evidence[key].valid_from > row.target_start
                            or (evidence[key].valid_to and evidence[key].valid_to < row.target_end)
                            for key in op_path
                        )
                        or not any(
                            evidence[key].valid_from is not None
                            and evidence[key].valid_from <= row.target_start
                            and (
                                evidence[key].valid_to is None
                                or evidence[key].valid_to >= row.target_end
                            )
                            for key in event.article_ids
                        )
                    )
                if not selected:
                    if any(
                        row.period == month
                        and any(applies(evidence[key], row) for key in event.article_ids)
                        for row in rows
                    ):
                        unresolved.append(spec)
                    continue
                used_facts.update(fact_ids)
                used_evidence.update(evidence_ids)
                # One event/entity/month assessment serves all verified products/customers.
                groups.append(
                    ExposureGroup(
                        group_id="group_"
                        + content_digest([event.event_id, entity.model_dump(), month]),
                        kind="news",
                        dependency=entity,
                        period=month,
                        event_ids=(event.event_id,),
                        evidence_ids=tuple(sorted(set(event.article_ids) | evidence_ids)),
                        fact_version_ids=tuple(sorted(fact_ids)),
                        members=tuple(selected),
                        exceptions=tuple(exceptions),
                        flags=("partial_event_validity",) if partial else (),
                    )
                )
    status = (
        "irrelevant"
        if requested == "irrelevant"
        else ("established" if groups and not unresolved else "uncertain")
    )
    payload = proposal.model_dump(include=set(MappingProposal.model_fields))
    payload.update(
        status=status,
        fact_version_ids=tuple(sorted(set(proposal.fact_version_ids) | used_facts)),
        evidence_ids=tuple(sorted(set(proposal.evidence_ids) | used_evidence)),
    )
    decision = MappingDecision(
        **payload,
        requested_status=requested,
        decision_id="mapping_" + content_digest(payload),
    )
    return decision, tuple(groups), tuple(unresolved)
