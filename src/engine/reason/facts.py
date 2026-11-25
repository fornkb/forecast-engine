"""Import and query dated company facts."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from itertools import combinations
from math import isfinite
from zoneinfo import ZoneInfo

from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.config import ContextSource, OperationalSource
from engine.reason.erp import (
    PostgresFetch,
    SourceRow,
    fetch_postgres_context,
    read_context_source,
    require_reasoning,
)
from engine.reason.schemas import (
    CompanyFact,
    ContextError,
    ContextSyncError,
    EntityRef,
    FactConflict,
    SyncResult,
    content_digest,
)
from engine.store.duckdb_store import EngineStore

OPERATOR_SOURCE = "operator_confirmation"


def require_client(profile: ClientProfile, store: EngineStore) -> None:
    if store.client_id != profile.client_id:
        raise ContextError("The client profile and context store must belong to the same client")


def utc_instant(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ContextError("Decision and validity timestamps must include a timezone")
    return value.astimezone(UTC)


def local_midnight(value: date, zone: ZoneInfo) -> datetime:
    return datetime.combine(value, time.min, tzinfo=zone).astimezone(UTC)


def source_time(
    value: object, zone: ZoneInfo, source: ContextSource | OperationalSource, *, known=False
) -> datetime:
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value.strip())
        except ValueError:
            value = datetime.fromisoformat(value.strip())
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            if source.naive_timestamps != "client_local":
                raise ValueError("datetime requires an explicit UTC offset")
            value = value.replace(tzinfo=zone, fold=0)
            if value.utcoffset() != value.replace(fold=1).utcoffset():
                raise ValueError("ambiguous or nonexistent local datetime requires a UTC offset")
        return value.astimezone(UTC)
    if isinstance(value, date):
        # A date with no release time proves availability by the end of that day.
        return local_midnight(value + timedelta(days=1) if known else value, zone)
    raise ValueError("expected an ISO date or datetime")


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def required_text(value: object, field: str) -> str:
    if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
        return str(value).strip()
    raise ValueError(f"{field} requires non-empty text or an integer ID")


def scalar_value(value: object, value_type: str) -> str | float | bool:
    if value_type == "number":
        if isinstance(value, bool):
            raise ValueError("value must be a finite number")
        try:
            number = float(str(value))
        except ValueError as exc:
            raise ValueError("value must be a finite number") from exc
        if not isfinite(number):
            raise ValueError("value must be a finite number")
        return number
    if value_type == "boolean":
        normalized = str(value).strip().lower()
        if normalized not in ("true", "false"):
            raise ValueError("value must be true or false")
        return normalized == "true"
    return required_text(value, "value")


def _normalize_fact(
    client_id: str, source: ContextSource, row: SourceRow, zone: ZoneInfo, captured_at: datetime
) -> CompanyFact:
    record_id = required_text(row["record_id"], "record_id")
    subject = EntityRef(type=source.subject_type, id=required_text(row["subject_id"], "subject_id"))
    source_version = (
        None if _blank(row.get("version")) else required_text(row["version"], "version")
    )
    effective_from = source_time(row["effective_from"], zone, source)
    effective_to = None
    if not _blank(row.get("effective_to")):
        effective_to = source_time(row["effective_to"], zone, source)
        if effective_to <= effective_from:
            raise ValueError("effective_to must come after effective_from")
    known_at = (
        None
        if _blank(row.get("known_at"))
        else source_time(row["known_at"], zone, source, known=True)
    )
    status = (
        "active" if _blank(row.get("status")) else required_text(row["status"], "status").lower()
    )
    if status not in ("active", "withdrawn"):
        raise ValueError("status must be active or withdrawn")
    object_ref, value = None, None
    if status == "active":
        if source.object_type:
            object_ref = EntityRef(
                type=source.object_type, id=required_text(row["object_id"], "object_id")
            )
        else:
            value = scalar_value(row.get("value"), source.value_type)
    normalized = {
        "subject": subject,
        "relation": source.relation_type,
        "object": object_ref,
        "value": value,
        "source_ref": f"{source.kind}:{source.path or source.relation}",
        "location_role": source.location_role,
        "cardinality": source.cardinality,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "status": status,
    }
    digest = content_digest({**normalized, "known_at": known_at})
    return CompanyFact(
        client_id=client_id,
        source_id=source.id,
        record_id=record_id,
        source_version=source_version,
        fact_version_id="fact_"
        + content_digest([client_id, source.id, record_id, source_version, digest]),
        content_digest=digest,
        captured_at=captured_at,
        known_at=known_at or captured_at,
        **normalized,
    )


def sync_context(
    profile: ClientProfile, store: EngineStore, *, fetch: PostgresFetch = fetch_postgres_context
) -> SyncResult:
    require_client(profile, store)
    config = require_reasoning(profile)
    if not config.sources:
        raise ContextSyncError("Configure reasoning.sources before syncing context facts")
    # Finish external reads before opening a transaction. Unknown availability uses
    # the clock after fetching, never the effective date or the start of the sync.
    batches = [
        (source, read_context_source(profile, source, fetch=fetch)) for source in config.sources
    ]
    captured_at = datetime.now(UTC)
    zone = ZoneInfo(config.timezone)
    facts = []
    for source, rows in batches:
        errors = []
        for index, row in enumerate(rows, start=1):
            try:
                facts.append(_normalize_fact(profile.client_id, source, row, zone, captured_at))
            except (ValueError, OverflowError) as exc:
                errors.append(f"record {index}: {exc}")
        if errors:
            raise ContextSyncError(
                f"Context source {source.id!r}: {errors[0]}", invalid=len(errors)
            )
    with records.write_transaction(store) as con:
        con.execute(records.REASON_DDL)
        new = sum(records.save_fact(con, fact) for fact in facts)
        active = records.eligible_facts(con, profile.client_id, captured_at)
        conflicted = {
            version for conflict in find_conflicts(active) for version in conflict.fact_version_ids
        }
        return SyncResult(new=new, reused=len(facts) - new, conflicted=len(conflicted))


def list_facts(
    store: EngineStore, *, known_at: datetime, valid_from: datetime, valid_to: datetime
) -> tuple[CompanyFact, ...]:
    """Resolve facts as of a decision instant and for a half-open validity interval."""
    known_at, valid_from, valid_to = map(utc_instant, (known_at, valid_from, valid_to))
    if valid_to <= valid_from:
        raise ContextError("Fact query validity end must come after its start")
    if not store.path.exists():
        return ()
    with store.connect() as con:
        if not records.table_exists(con, "reason_company_facts"):
            return ()
        return records.eligible_facts(con, store.client_id, known_at, valid_from, valid_to)


def get_fact_version(store: EngineStore, fact_version_id: str) -> CompanyFact:
    if store.path.exists():
        with store.connect() as con:
            if records.table_exists(con, "reason_company_facts"):
                fact = records.get_fact(con, store.client_id, fact_version_id)
                if fact:
                    return fact
    raise ContextError(f"Fact version {fact_version_id!r} was not found for {store.client_id!r}")


def overlaps(fact: CompanyFact, start: datetime, end: datetime) -> bool:
    return fact.effective_from < end and (fact.effective_to is None or fact.effective_to > start)


def _conflicting_pair(
    left: CompanyFact,
    right: CompanyFact,
    confirmations: list[CompanyFact],
    valid_from: datetime | None,
    valid_to: datetime | None,
) -> bool:
    if left.cardinality == right.cardinality == "many" or (left.object, left.value) == (
        right.object,
        right.value,
    ):
        return False
    start = max(
        value
        for value in (left.effective_from, right.effective_from, valid_from)
        if value is not None
    )
    end = min(
        (value for value in (left.effective_to, right.effective_to, valid_to) if value is not None),
        default=None,
    )
    if end is not None and start >= end:
        return False
    pair = {left.fact_version_id, right.fact_version_id}
    return not any(
        (pair - {confirmed.fact_version_id}).issubset(confirmed.resolves_fact_ids)
        and confirmed.effective_from <= start
        and (confirmed.effective_to is None or (end is not None and confirmed.effective_to >= end))
        for confirmed in confirmations
    )


def find_conflicts(
    facts: tuple[CompanyFact, ...],
    *,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> tuple[FactConflict, ...]:
    """Report unresolved overlapping values for fields configured as singular."""
    groups = defaultdict(list)
    for fact in facts:
        groups[fact.field_key].append(fact)
    result = []
    for (subject, relation, location_role), group in groups.items():
        confirmations = [fact for fact in group if fact.status == "confirmed"]
        versions = {
            fact.fact_version_id
            for left, right in combinations(group, 2)
            if _conflicting_pair(left, right, confirmations, valid_from, valid_to)
            for fact in (left, right)
        }
        if versions:
            result.append(
                FactConflict(
                    subject=subject,
                    relation=relation,
                    location_role=location_role,
                    fact_version_ids=tuple(sorted(versions)),
                )
            )
    return tuple(result)
