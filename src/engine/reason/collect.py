"""Frozen monthly evidence collection and the shared, budgeted single-scope lookup."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from engine.config.base import StrictModel
from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.config import EvidenceConfig, OperationalSource, ProviderPolicy
from engine.reason.context import get_forecast_snapshot, load_forecast_snapshot
from engine.reason.erp import (
    OperationalReadError,
    PostgresFetch,
    fetch_postgres_context,
    read_operational_source,
    require_reasoning,
)
from engine.reason.facts import local_midnight, require_client, utc_instant
from engine.reason.matching import (
    candidate_articles,
    demand_groups,
    operating_links,
    row_facts,
    weather_groups,
    weather_links,
)
from engine.reason.news import news_request, parse_news
from engine.reason.providers import (
    HttpRequest,
    HttpTransport,
    ParsedEvidence,
    ProviderError,
    http_get,
    next_month,
)
from engine.reason.schemas import (
    CompanyFact,
    ContextError,
    EntityRef,
    EvidenceBatch,
    EvidenceRevision,
    ForecastSnapshot,
    LookupResult,
    MonthCoverage,
    ProviderUsage,
    QueryRequestContext,
    QueryScope,
    SourceCheck,
    SourceKind,
    content_digest,
)
from engine.reason.supply import active_operational, parse_operational, supply_groups
from engine.reason.weather import enso_request, parse_enso, parse_seasonal, seasonal_request
from engine.store.duckdb_store import EngineStore

QUERY_VERSION = "phase2-v1"
_LEASE = timedelta(seconds=90)
_PROVIDERS = {"erp": "erp", "news": "brave", "weather": "open_meteo", "enso": "noaa_cpc"}


class CollectionRun(StrictModel):
    forecast_run_id: int
    snapshot_id: str
    config_digest: str
    started_at: datetime
    as_of: datetime | None = None


@dataclass(frozen=True)
class _Lookup:
    profile: ClientProfile
    store: EngineStore
    owner: tuple[str, str, str]
    config: EvidenceConfig
    total_limit: int
    as_of: datetime | None
    http: HttpTransport
    fetch: PostgresFetch


@dataclass(frozen=True)
class _Attempt:
    identifier: str
    started: datetime
    retry: bool
    request: HttpRequest | None


@dataclass(frozen=True)
class _Plan:
    scope: QueryScope
    skip: str | None = None


def _config_digest(profile: ClientProfile) -> str:
    config = require_reasoning(profile)
    return content_digest(
        [
            QUERY_VERSION,
            config.timezone,
            config.evidence.model_dump(mode="json"),
            profile.profile_dir,
        ]
    )


def _months(periods: tuple[date, ...]) -> tuple[date, ...]:
    ordered = tuple(sorted(set(periods)))
    if not ordered or any(period.day != 1 for period in ordered):
        raise ContextError("Evidence scopes require explicit calendar months")
    return ordered


def _scope(
    profile: ClientProfile,
    kind: SourceKind,
    source_id: str,
    months: tuple[date, ...],
    *,
    page: int = 0,
    entities: tuple[EntityRef, ...] = (),
    aliases: tuple[str, ...] = (),
    coordinates=None,
    role=None,
    filters: tuple[tuple[str, str], ...] = (),
) -> QueryScope:
    months = _months(months)
    zone = ZoneInfo(require_reasoning(profile).timezone)
    scope = QueryScope(
        client_id=profile.client_id,
        source_kind=kind,
        source_id=source_id,
        query_hash="",
        entity_ids=entities,
        public_aliases=aliases,
        coordinates=coordinates,
        location_role=role,
        months=months,
        valid_from=local_midnight(months[0], zone),
        valid_to=local_midnight(next_month(months[-1]), zone),
        filters=(("timezone", zone.key), *filters),
        page=page,
    )
    return scope.model_copy(
        update={
            "query_hash": content_digest(
                [
                    QUERY_VERSION,
                    scope.model_dump(mode="json", exclude={"query_hash"}),
                ]
            )
        }
    )


def make_query_scope(
    profile: ClientProfile,
    source_kind: SourceKind,
    source_id: str,
    months: tuple[date, ...],
    *,
    page: int = 0,
    as_of: datetime | None = None,
    location_role=None,
) -> QueryScope:
    """Build an approved scope from identifiers in the deployment profile."""
    config = require_reasoning(profile).evidence
    if page < 0 or (page and source_kind != "news"):
        raise ContextError("Only configured news scopes support pagination")
    if source_kind == "erp" and any(source.id == source_id for source in config.erp):
        return _scope(profile, source_kind, source_id, months)
    if source_kind == "news" and config.news and page < config.news.max_pages:
        search = next((search for search in config.news.searches if search.id == source_id), None)
        if search:
            today = utc_instant(as_of or datetime.now(UTC)).date()
            since = today - timedelta(days=config.news.lookback_days - 1)
            return _scope(
                profile,
                source_kind,
                source_id,
                months,
                page=page,
                aliases=tuple(
                    " ".join(value.split())
                    for value in (search.public_alias, search.geography)
                    if value
                ),
                entities=(EntityRef(type=search.entity_type, id=search.entity_id),)
                if search.entity_id
                else (),
                filters=(("freshness", f"{since}to{today}"),),
            )
    if source_kind == "weather" and config.weather:
        location = next(
            (
                value
                for value in config.weather.locations
                if (value.location_id, value.role) == (source_id, location_role)
                and value.role in config.weather.applicable_roles
            ),
            None,
        )
        if location:
            return _scope(
                profile,
                source_kind,
                source_id,
                months,
                role=location.role,
                entities=(EntityRef(type="location", id=location.location_id),),
                coordinates=(location.latitude, location.longitude),
            )
    if source_kind == "enso" and config.enso and source_id == "noaa_cpc":
        return _scope(profile, source_kind, source_id, months)
    raise ContextError("Requested provider, search, location, page, or ERP view is not configured")


def _source(config: EvidenceConfig, scope: QueryScope) -> OperationalSource | ProviderPolicy:
    if scope.source_kind == "erp":
        return next(source for source in config.erp if source.id == scope.source_id)
    return getattr(config, scope.source_kind)


def _cache_key(service: _Lookup, scope: QueryScope) -> str:
    source = _source(service.config, scope)
    options = source.model_dump(mode="json")
    if scope.source_kind == "news":
        options.pop("searches")
        options["primary_domains"] = sorted(
            {domain for search in source.searches for domain in search.primary_domains}
        )
    elif scope.source_kind == "weather":
        options.pop("locations")
        options["climatology"] = [
            ref.model_dump(mode="json")
            for ref in source.climatology
            if ref.location_id == scope.source_id
        ]
    return content_digest(
        [
            QUERY_VERSION,
            scope.client_id,
            scope.source_kind,
            options,
            scope.months,
            scope.filters,
            scope.public_aliases,
            scope.coordinates,
            scope.page,
            service.profile.profile_dir if scope.source_kind == "erp" else None,
        ]
    )


def _check(
    scope: QueryScope,
    started: datetime,
    ended: datetime,
    **fields,
) -> SourceCheck:
    if scope.source_kind in {"weather", "enso"} and "months" not in fields:
        fields["months"] = tuple(
            MonthCoverage(period=month, status="none", reason=fields.get("reason"))
            for month in scope.months
        )
    return SourceCheck(
        check_id="check_" + str(uuid4()),
        scope=scope,
        provider=_PROVIDERS[scope.source_kind],
        started_at=started,
        ended_at=ended,
        **fields,
    )


def _counts(con, owner: tuple[str, str, str], scope: QueryScope) -> tuple[int, int, int]:
    return con.execute(
        """SELECT count(*), count(*) FILTER (WHERE provider = ?),
        count(*) FILTER (WHERE query_hash = ?) FROM reason_evidence_attempts
        WHERE client_id = ? AND owner_kind = ? AND owner_id = ?""",
        [_PROVIDERS[scope.source_kind], scope.query_hash, *owner],
    ).fetchone()


def _begin(
    service: _Lookup,
    scope: QueryScope,
    cache_key: str,
    *,
    skip: str | None,
    retry_limit: int,
) -> SourceCheck | _Attempt:
    with records.write_transaction(service.store) as con:
        prior = records.read_checks(con, service.owner, scope.query_hash)
        if service.owner[1] == "collection" and prior:
            last = prior[-1]
            retryable = last.retryable and last.attempts
            if not retryable or sum(check.attempts for check in prior) >= retry_limit + 1:
                return last
        now = datetime.now(UTC)
        if skip:
            status = "not_applicable" if skip == "not_applicable" else "unavailable"
            check = _check(scope, now, now, status=status, coverage="none", reason=skip)
            records.save_check(con, service.owner, cache_key, check)
            return check
        source = _source(service.config, scope)
        cutoff = service.as_of or now
        ttl = source.cache_seconds
        earliest = cutoff - timedelta(seconds=ttl) if ttl else None
        cached = records.cached_check(
            con,
            scope.client_id,
            cache_key,
            cutoff,
            None,
            allow_partial=True,
            owner=service.owner[1:],
        )
        if cached is None and (service.as_of is not None or ttl):
            cached = records.cached_check(
                con,
                scope.client_id,
                cache_key,
                cutoff,
                earliest,
                allow_partial=service.as_of is not None,
            )
        if cached:
            check = cached.model_copy(
                update={
                    "scope": scope,
                    "check_id": "check_" + str(uuid4()),
                    "started_at": now,
                    "ended_at": now,
                    "requests": 0,
                    "attempts": 0,
                    "retries": 0,
                    "cache_hits": 1,
                    "reused_check_id": cached.check_id,
                }
            )
            records.save_check(con, service.owner, cache_key, check)
            return check
        gap = "no_saved_eligible_check" if service.as_of is not None else None
        pending = con.execute(
            """SELECT attempt_id, started_at, retry FROM reason_evidence_attempts
            WHERE client_id = ? AND owner_kind = ? AND owner_id = ?
            AND query_hash = ? AND check_id IS NULL""",
            [*service.owner, scope.query_hash],
        ).fetchone()
        if pending:
            attempt_id, started, repeat = pending
            if now - started < _LEASE:
                raise ContextError(
                    f"Lookup is in progress. Resume {service.owner[2]} after it ends."
                )
            check = _check(
                scope,
                started,
                now,
                status="unavailable",
                coverage="none",
                reason="interrupted_attempt",
                provider_status="interrupted",
                retryable=True,
                attempts=1,
                requests=int(not repeat),
                retries=int(repeat),
            )
            records.save_check(con, service.owner, cache_key, check)
            con.execute(
                "UPDATE reason_evidence_attempts SET check_id = ? "
                "WHERE client_id = ? AND attempt_id = ?",
                [check.check_id, scope.client_id, attempt_id],
            )
            return check
        total, provider, per_scope = _counts(con, service.owner, scope)
        provider_limit = (
            service.config.max_erp_attempts if scope.source_kind == "erp" else source.max_attempts
        )
        if total >= service.total_limit or provider >= provider_limit:
            gap = gap or "request_budget_exhausted"
        request = None
        if gap is None:
            try:
                request = _prepare_request(scope, source, now)
            except ProviderError as exc:
                gap = str(exc)
        if gap:
            check = _check(scope, now, now, status="unavailable", coverage="none", reason=gap)
            records.save_check(con, service.owner, cache_key, check)
            return check
        attempt = _Attempt(str(uuid4()), now, bool(per_scope), request)
        con.execute(
            "INSERT INTO reason_evidence_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            [
                scope.client_id,
                attempt.identifier,
                *service.owner[1:],
                scope.query_hash,
                _PROVIDERS[scope.source_kind],
                now,
                attempt.retry,
            ],
        )
        return attempt


def _prepare_request(scope: QueryScope, source, now: datetime) -> HttpRequest | None:
    if scope.source_kind == "news":
        return news_request(source, scope)
    if scope.source_kind == "weather":
        return seasonal_request(source, scope, now)
    if scope.source_kind == "enso":
        return enso_request(source)
    if source.kind == "postgres" and not os.environ.get(source.dsn_env):
        raise ProviderError("missing_credentials")
    return None


def _perform(
    service: _Lookup,
    scope: QueryScope,
    source: OperationalSource | ProviderPolicy,
    attempt: _Attempt,
) -> tuple[ParsedEvidence, str]:
    if scope.source_kind == "erp":
        try:
            rows, truncated = read_operational_source(service.profile, source, fetch=service.fetch)
        except OperationalReadError as exc:
            raise ProviderError(
                "erp_source_unavailable",
                status=exc.status,
                transient=exc.transient,
            ) from None
        except ContextError:
            raise ProviderError("erp_source_unavailable") from None
        return parse_operational(source, scope, rows, truncated, datetime.now(UTC)), "ok"
    response = service.http(attempt.request, source.timeout_seconds)
    if not 200 <= response.status < 300:
        transient = response.status in {408, 425, 429, 500, 502, 503, 504}
        raise ProviderError(
            f"http_{response.status}", status=str(response.status), transient=transient
        )
    captured = datetime.now(UTC)
    try:
        if scope.source_kind == "news":
            parsed = parse_news(response.json(), source, scope, captured)
        elif scope.source_kind == "weather":
            parsed = parse_seasonal(response.json(), source, scope, captured)
        else:
            parsed = parse_enso(response.body, source, scope, captured)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise ProviderError("invalid_provider_payload", status=str(response.status)) from None
    return parsed, str(response.status)


def _execute(
    service: _Lookup,
    scope: QueryScope,
    *,
    skip: str | None = None,
) -> SourceCheck:
    cache_key = _cache_key(service, scope) if skip is None else ""
    source = _source(service.config, scope) if skip is None else None
    retry_limit = 0
    if service.owner[1] == "collection":
        if isinstance(source, ProviderPolicy):
            retry_limit = source.retries
        elif isinstance(source, OperationalSource) and source.kind == "postgres":
            retry_limit = service.config.erp_retries
    while True:
        iteration = datetime.now(UTC)
        action = _begin(service, scope, cache_key, skip=skip, retry_limit=retry_limit)
        if isinstance(action, SourceCheck):
            if (
                action.provider_status == "interrupted"
                and retry_limit > 0
                and action.ended_at >= iteration
            ):
                # A reclaimed attempt was just persisted. The ledger still enforces the cap.
                continue
            return action
        error = None
        try:
            parsed, status = _perform(service, scope, source, action)
        except (TimeoutError, OSError):
            error = ProviderError("network_error", transient=True)
        except ProviderError as exc:
            error = exc
        ended = datetime.now(UTC)
        with records.write_transaction(service.store) as con:
            reserved = con.execute(
                "SELECT check_id FROM reason_evidence_attempts "
                "WHERE client_id = ? AND attempt_id = ?",
                [scope.client_id, action.identifier],
            ).fetchone()
            if reserved[0] is not None:
                raise ContextError(f"Lookup lease ended. Resume {service.owner[2]}.")
            if error is not None:
                check = _check(
                    scope,
                    action.started,
                    ended,
                    status="unavailable",
                    coverage="none",
                    reason=str(error),
                    provider_status=error.status,
                    retryable=error.transient,
                    attempts=1,
                    requests=int(not action.retry),
                    retries=int(action.retry),
                )
                cacheable = False
            else:
                saved = records.put_evidence(con, scope.client_id, parsed.revisions)
                ids = tuple(
                    sorted(
                        {
                            revision.evidence_id
                            for revision in saved
                            if revision.first_known_at <= ended
                        }
                    )
                )
                eligible_records = {
                    revision.record_id for revision in saved if revision.first_known_at <= ended
                }
                ineligible = any(
                    revision.first_known_at > ended and revision.record_id not in eligible_records
                    for revision in saved
                )
                coverage = parsed.coverage
                if ineligible and coverage == "complete":
                    coverage = "partial"
                month_status = parsed.months
                if ineligible:
                    month_status = tuple(
                        MonthCoverage(
                            period=month.period, status="partial", reason="source_not_yet_known"
                        )
                        if month.status == "complete"
                        else month
                        for month in month_status
                    )
                check = _check(
                    scope,
                    action.started,
                    ended,
                    status="found"
                    if ids
                    else "no_match"
                    if coverage == "complete"
                    else "unavailable",
                    coverage=coverage,
                    reason="source_not_yet_known" if ineligible else parsed.reason,
                    covered_from=parsed.covered_from,
                    covered_to=parsed.covered_to,
                    months=month_status,
                    evidence_ids=ids,
                    provider_status=status,
                    more_pages=parsed.more_pages,
                    attempts=1,
                    requests=int(not action.retry),
                    retries=int(action.retry),
                )
                cacheable = True
            records.save_check(con, service.owner, cache_key, check, cacheable=cacheable)
            con.execute(
                "UPDATE reason_evidence_attempts SET check_id = ? "
                "WHERE client_id = ? AND attempt_id = ?",
                [check.check_id, scope.client_id, action.identifier],
            )
        if error is None or not error.transient or retry_limit == 0:
            return check
        # The failed check is durable before another attempt is reserved.


def _check_snapshot(
    profile: ClientProfile,
    snapshot: ForecastSnapshot,
    forecast_run_id: int,
) -> None:
    if (
        type(forecast_run_id) is not int
        or forecast_run_id < 1
        or snapshot.client_id != profile.client_id
        or snapshot.forecast_run_id != forecast_run_id
        or snapshot.granularity != "monthly"
        or profile.calendar.granularity != "monthly"
        or snapshot.timezone != require_reasoning(profile).timezone
    ):
        raise ContextError(
            "Snapshot must belong to the requested client, monthly forecast run and timezone"
        )


def _bind(con, owner: tuple[str, str, str], data: str) -> None:
    binding = con.execute(
        "SELECT binding FROM reason_evidence_owners "
        "WHERE client_id = ? AND owner_kind = ? AND owner_id = ?",
        list(owner),
    ).fetchone()
    if binding and json.loads(binding[0]) != json.loads(data):
        raise ContextError("Evidence request does not match the saved collection or review binding")
    if not binding:
        con.execute("INSERT INTO reason_evidence_owners VALUES (?, ?, ?, ?)", [*owner, data])


def lookup_scope(
    profile: ClientProfile,
    store: EngineStore,
    scope: QueryScope,
    request_context: QueryRequestContext,
    *,
    http: HttpTransport = http_get,
    fetch: PostgresFetch = fetch_postgres_context,
) -> LookupResult:
    """Look up one bounded page. Each explicit retry may reserve one more review slot."""
    require_client(profile, store)
    config = require_reasoning(profile).evidence
    if request_context.client_id != store.client_id or scope.client_id != store.client_id:
        raise ContextError("Lookup context and scope must belong to the configured client")
    cutoff = (
        utc_instant(request_context.knowledge_cutoff) if request_context.knowledge_cutoff else None
    )
    if cutoff is not None and cutoff > datetime.now(UTC):
        raise ContextError("Knowledge cutoff cannot be in the future")
    snapshot = get_forecast_snapshot(store, request_context.snapshot_id)
    _check_snapshot(profile, snapshot, request_context.forecast_run_id)
    months = _months(request_context.months)
    if set(months) - {row.period for row in snapshot.rows} or set(scope.months) - set(months):
        raise ContextError("Lookup months must belong to the pinned review and forecast")
    if cutoff is not None and snapshot.captured_at > cutoff:
        raise ContextError("Snapshot was captured after the knowledge cutoff")
    approved = make_query_scope(
        profile,
        scope.source_kind,
        scope.source_id,
        scope.months,
        page=scope.page,
        as_of=cutoff,
        location_role=scope.location_role,
    )
    if approved != scope:
        raise ContextError(
            "Lookup scope does not match the configured source and public parameters"
        )
    limit = min(config.followup_limit, request_context.request_budget)
    owner = (profile.client_id, "review", request_context.review_id)
    with records.write_transaction(store) as con:
        con.execute(records.REASON_DDL)
        _bind(
            con,
            owner,
            json.dumps(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "months": [value.isoformat() for value in months],
                    "limit": limit,
                    "config": _config_digest(profile),
                }
            ),
        )
    service = _Lookup(profile, store, owner, config, limit, cutoff, http, fetch)
    check = _execute(service, scope)
    checks = (check,)
    if check.more_pages and scope.source_kind == "news":
        reason = (
            "page_cap_reached"
            if scope.page + 1 >= config.news.max_pages
            else "more_pages_available"
        )
        checks += (_pagination_gap(service, scope, check, reason=reason),)
    with store.connect() as con:
        count = _counts(con, owner, scope)[0]
    return LookupResult(checks=checks, evidence_ids=check.evidence_ids, attempts_used=count)


def recover_review_attempts(con, client_id: str, review_id: str, scopes: tuple[QueryScope, ...]):
    """Close expired reservations without dispatching another provider request."""
    owner = (client_id, "review", review_id)
    indexed = {scope.query_hash: scope for scope in scopes}
    now = datetime.now(UTC)
    for attempt in records.read_lookup_attempts(con, client_id, review_id):
        if attempt.check_id is not None:
            continue
        if now - attempt.started_at < _LEASE:
            raise ContextError(
                f"Lookup is in progress. Resume {review_id} after its lease expires."
            )
        scope = indexed.get(attempt.query_hash)
        if scope is None:
            raise ContextError("Interrupted lookup has no pinned request scope")
        check = _check(
            scope,
            attempt.started_at,
            now,
            status="unavailable",
            coverage="none",
            reason="interrupted_attempt",
            provider_status="interrupted",
            retryable=True,
            attempts=1,
            requests=int(not attempt.retry),
            retries=int(attempt.retry),
        )
        records.save_check(con, owner, "", check)
        con.execute(
            "UPDATE reason_evidence_attempts SET check_id = ? "
            "WHERE client_id = ? AND attempt_id = ?",
            [check.check_id, client_id, attempt.attempt_id],
        )


def _pagination_gap(
    service: _Lookup,
    scope: QueryScope,
    check: SourceCheck,
    *,
    reason: str = "page_cap_reached",
) -> SourceCheck:
    if check.reason == reason:
        return check
    now = datetime.now(UTC)
    gap = _check(
        scope,
        now,
        now,
        status=check.status,
        coverage="partial",
        reason=reason,
        evidence_ids=check.evidence_ids,
        reused_check_id=check.check_id,
        more_pages=True,
    )
    with records.write_transaction(service.store) as con:
        records.save_check(con, service.owner, "", gap)
    return gap


def _load_collection(store: EngineStore, collection_id: str) -> CollectionRun:
    if store.path.exists():
        with store.connect() as con:
            if records.table_exists(con, "reason_evidence_owners"):
                found = con.execute(
                    "SELECT binding FROM reason_evidence_owners "
                    "WHERE client_id = ? AND owner_kind = 'collection' AND owner_id = ?",
                    [store.client_id, collection_id],
                ).fetchone()
                if found:
                    return CollectionRun.model_validate_json(found[0])
    raise ContextError(f"Collection {collection_id!r} was not found for this client")


def _snapshot_at(
    store: EngineStore,
    run_id: int,
    cutoff: datetime,
) -> ForecastSnapshot:
    if store.path.exists():
        with store.connect() as con:
            if records.table_exists(con, "reason_forecast_snapshots"):
                row = con.execute(
                    "SELECT snapshot_id FROM reason_forecast_snapshots "
                    "WHERE client_id = ? AND forecast_run_id = ? AND captured_at <= ? "
                    "ORDER BY captured_at DESC, snapshot_id LIMIT 1",
                    [store.client_id, run_id, cutoff],
                ).fetchone()
                if row:
                    return records.read_snapshot(con, store.client_id, row[0])
    raise ContextError(
        "Historical collection requires a context snapshot saved by its knowledge cutoff"
    )


def _context_facts(store: EngineStore, snapshot: ForecastSnapshot) -> dict[str, CompanyFact]:
    ids = sorted({identifier for row in snapshot.rows for identifier in row.fact_version_ids})
    if not ids:
        return {}
    with store.connect() as con:
        rows = con.execute(
            "SELECT fact_data FROM reason_company_facts "
            "WHERE client_id = ? AND fact_version_id IN (SELECT unnest(?::VARCHAR[])) "
            "AND captured_at <= ? AND known_at <= ?",
            [store.client_id, ids, snapshot.captured_at, snapshot.captured_at],
        ).fetchall()
    facts = [CompanyFact.model_validate_json(item[0]) for item in rows]
    return {fact.fact_version_id: fact for fact in facts}


def _external_plan(
    profile: ClientProfile,
    snapshot: ForecastSnapshot,
    at: datetime,
    links: dict,
    locations: dict,
    operating: dict,
) -> list[_Plan]:
    config = require_reasoning(profile).evidence
    months = _months(tuple(row.period for row in snapshot.rows))
    plan = []
    if config.news:
        by_entity = defaultdict(set)
        for (series, period), entities in links.items():
            for entity in entities.keys() | operating[series, period].keys():
                by_entity[entity].add(period)
        for (_, period), mapped in locations.items():
            for location, _role in mapped:
                by_entity[EntityRef(type="location", id=location)].add(period)
        if not config.news.searches:
            plan.append(
                _Plan(_scope(profile, "news", "public_search", months), "missing_public_alias")
            )
        for search in config.news.searches:
            entity = (
                EntityRef(type=search.entity_type, id=search.entity_id)
                if search.entity_id
                else None
            )
            relevant = tuple(sorted(by_entity.get(entity, ()))) if entity else months
            query = make_query_scope(profile, "news", search.id, relevant or months, as_of=at)
            plan.append(_Plan(query, "missing_dependency_mapping" if not relevant else None))
    if config.weather:
        mapped = defaultdict(set)
        for (_, period), row_locations in locations.items():
            for key in row_locations:
                mapped[key].add(period)
        role_gaps = defaultdict(set)
        for row in snapshot.rows:
            row_locations = locations[row.series_id, row.period]
            for role in config.weather.applicable_roles:
                if not any(location_role == role for _, location_role in row_locations):
                    role_gaps[role].add(row.period)
        configured = {(value.location_id, value.role) for value in config.weather.locations}
        for key, periods in sorted(mapped.items()):
            location, role = key
            relevant = tuple(sorted(periods))
            if role not in config.weather.applicable_roles:
                query = _scope(profile, "weather", location, relevant, role=role)
                plan.append(_Plan(query, "not_applicable"))
            elif key not in configured:
                query = _scope(profile, "weather", location, relevant, role=role)
                plan.append(_Plan(query, "missing_coordinates"))
            else:
                query = make_query_scope(profile, "weather", location, relevant, location_role=role)
                plan.append(_Plan(query))
        for role, periods in sorted(role_gaps.items()):
            query = _scope(profile, "weather", "unmapped", tuple(sorted(periods)), role=role)
            plan.append(_Plan(query, "missing_location_mapping"))
        if not config.weather.applicable_roles and not mapped:
            plan.append(_Plan(_scope(profile, "weather", "unlinked", months), "not_applicable"))
        for location, role in sorted(configured - mapped.keys()):
            query = _scope(profile, "weather", location, months, role=role)
            reason = (
                "missing_location_mapping"
                if role in config.weather.applicable_roles
                else "not_applicable"
            )
            plan.append(_Plan(query, reason))
    if config.enso:
        plan.append(_Plan(make_query_scope(profile, "enso", "noaa_cpc", months)))
    return plan


def _collect_checks(
    service: _Lookup,
    plan: list[_Plan],
    at: datetime,
) -> list[SourceCheck]:
    results = []
    for item in plan:
        scope = item.scope
        check = _execute(service, scope, skip=item.skip)
        results.append(check)
        if item.skip or scope.source_kind != "news":
            continue
        cap = service.config.news.max_pages
        while check.more_pages and scope.page + 1 < cap:
            scope = make_query_scope(
                service.profile,
                "news",
                scope.source_id,
                scope.months,
                page=scope.page + 1,
                as_of=at,
            )
            check = _execute(service, scope)
            results.append(check)
        if check.more_pages:
            results[-1] = _pagination_gap(service, scope, check)
    return results


def _usage(checks: tuple[SourceCheck, ...]) -> tuple[ProviderUsage, ...]:
    by_provider = defaultdict(list)
    for check in checks:
        by_provider[check.provider].append(check)
    return tuple(
        ProviderUsage(
            provider=provider,
            requests=sum(check.requests for check in group),
            attempts=sum(check.attempts for check in group),
            retries=sum(check.retries for check in group),
            cache_hits=sum(check.cache_hits for check in group),
        )
        for provider, group in sorted(by_provider.items())
    )


def collect_evidence(
    profile: ClientProfile,
    store: EngineStore,
    forecast_run_id: int,
    *,
    snapshot_id: str | None = None,
    resume_id: str | None = None,
    as_of: datetime | None = None,
    http: HttpTransport = http_get,
    fetch: PostgresFetch = fetch_postgres_context,
    on_start: Callable[[str], None] | None = None,
) -> EvidenceBatch:
    """Run or resume collection, then commit its immutable evidence and membership manifest."""
    require_client(profile, store)
    config = require_reasoning(profile).evidence
    digest = _config_digest(profile)
    now = datetime.now(UTC)
    cutoff = utc_instant(as_of) if as_of is not None else None
    if cutoff and cutoff > now:
        raise ContextError("Knowledge cutoff cannot be in the future")
    collection_id = resume_id or "collection_" + str(uuid4())
    owner = (store.client_id, "collection", collection_id)
    if resume_id:
        job = _load_collection(store, resume_id)
        if (
            job.forecast_run_id != forecast_run_id
            or (snapshot_id and snapshot_id != job.snapshot_id)
            or (cutoff and cutoff != job.as_of)
        ):
            raise ContextError("Resume ID belongs to a different forecast, snapshot or cutoff")
        with store.connect() as con:
            batch = records.read_batch(con, store.client_id, collection_id, by_collection=True)
        if batch:
            return batch
        if job.config_digest != digest:
            raise ContextError("Restore the original collection configuration before resuming")
        snapshot = get_forecast_snapshot(store, job.snapshot_id)
    else:
        if snapshot_id:
            snapshot = get_forecast_snapshot(store, snapshot_id)
        elif cutoff:
            snapshot = _snapshot_at(store, forecast_run_id, cutoff)
        else:
            snapshot = load_forecast_snapshot(profile, store, forecast_run_id)
        job = CollectionRun(
            forecast_run_id=forecast_run_id,
            snapshot_id=snapshot.snapshot_id,
            config_digest=digest,
            started_at=now,
            as_of=cutoff,
        )
    _check_snapshot(profile, snapshot, forecast_run_id)
    cutoff = job.as_of
    if cutoff and snapshot.captured_at > cutoff:
        raise ContextError("Snapshot was captured after the knowledge cutoff")
    with records.write_transaction(store) as con:
        con.execute(records.REASON_DDL)
        _bind(con, owner, job.model_dump_json())
    if on_start:
        on_start(collection_id)
    service = _Lookup(profile, store, owner, config, config.max_attempts, cutoff, http, fetch)
    at = cutoff or job.started_at
    months = _months(tuple(row.period for row in snapshot.rows))
    erp_checks = _collect_checks(
        service,
        [_Plan(make_query_scope(profile, "erp", source.id, months)) for source in config.erp],
        at,
    )
    with store.connect() as con:
        erp = records.read_evidence(
            con,
            store.client_id,
            tuple(key for check in erp_checks for key in check.evidence_ids),
        )
    operational = active_operational(erp)
    facts = _context_facts(store, snapshot)
    links = {(row.series_id, row.period): row_facts(snapshot, facts, row) for row in snapshot.rows}
    locations = {
        (row.series_id, row.period): weather_links(links[row.series_id, row.period], facts)
        for row in snapshot.rows
    }
    operating = operating_links(snapshot, operational, locations)
    for key, entities in operating.items():
        for entity in entities:
            if entity.type in {"site", "route"}:
                locations[key].setdefault((entity.id, entity.type), ())
    external_checks = _collect_checks(
        service,
        _external_plan(profile, snapshot, at, links, locations, operating),
        at,
    )
    return _freeze(
        service,
        snapshot,
        job,
        (*erp_checks, *external_checks),
        operational,
        locations,
        operating,
    )


def _freeze(
    service: _Lookup,
    snapshot: ForecastSnapshot,
    job: CollectionRun,
    checks: tuple[SourceCheck, ...],
    operational: tuple[EvidenceRevision, ...],
    locations: dict,
    operating: dict,
) -> EvidenceBatch:
    client, _, collection_id = service.owner
    required = {"dependency", "usable_stock", "consumption", "replenishment"}
    complete_sources = {
        check.scope.source_id
        for check in checks
        if check.scope.source_kind == "erp" and check.coverage == "complete"
    }
    mapped = {
        "replenishment"
        if source.dataset in {"receipt", "purchase_order"}
        else "dependency"
        if source.dataset in {"bom", "vendor_link"}
        else source.dataset
        for source in service.config.erp
        if source.id in complete_sources
    }
    gaps = {f"missing_supply_dataset:{dataset}" for dataset in required - mapped}
    supply_feeds = {
        source.id
        for source in service.config.erp
        if source.dataset
        in {
            "bom",
            "vendor_link",
            "usable_stock",
            "consumption",
            "purchase_order",
            "receipt",
            "allocation",
            "alternate",
        }
    }
    gaps.update(
        f"context:{gap.code}"
        for gap in snapshot.gaps
        if gap.code not in {"legacy_provenance", "catalog_unverified", "horizon_unverified"}
    )
    with records.write_transaction(service.store) as con:
        existing = records.read_batch(con, client, collection_id, by_collection=True)
        if existing:
            return existing
        frozen = datetime.now(UTC)
        cutoff = job.as_of or frozen
        all_checks = records.read_checks(con, service.owner)
        ids = tuple(sorted({key for check in checks for key in check.evidence_ids}))
        evidence = tuple(
            revision
            for revision in records.read_evidence(con, client, ids)
            if revision.first_known_at <= cutoff and revision.captured_at <= cutoff
        )
        groups = (
            *supply_groups(
                snapshot,
                operational,
                complete_inputs=required <= mapped and supply_feeds <= complete_sources,
                locations=locations,
            ),
            *weather_groups(
                snapshot,
                checks,
                {item.evidence_id: item for item in evidence},
                locations,
                operating,
            ),
            *demand_groups(snapshot, operational),
        )
        coverage_flags = {
            "missing_usable_stock",
            "no_opening_stock_snapshot",
            "same_day_stock_movement_order_unknown",
            "ambiguous_stock_baseline",
            "ambiguous_receipt_identity",
            "unverified_warehouse_transfers",
            "incompatible_unit",
            "unresolved_multilevel_bom",
            "ambiguous_bom",
            "bom_timing_or_finished_unit_mismatch",
            "substitute_availability_unknown",
            "ambiguous_customer_item_mapping",
        }
        gaps.update(
            f"{group.kind}:{flag}"
            for group in groups
            for flag in group.flags
            if flag in coverage_flags
        )
        gaps.update(
            f"{check.scope.source_kind}:{check.scope.source_id}:{check.reason or check.status}"
            for check in checks
            if check.status != "not_applicable" and check.coverage != "complete"
        )
        batch = EvidenceBatch(
            batch_id="batch_" + content_digest([client, collection_id, cutoff]),
            collection_id=collection_id,
            client_id=client,
            forecast_run_id=snapshot.forecast_run_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_captured_at=snapshot.captured_at,
            query_version=QUERY_VERSION,
            config_digest=job.config_digest,
            started_at=job.started_at,
            knowledge_cutoff=cutoff,
            frozen_at=frozen,
            check_ids=tuple(check.check_id for check in all_checks),
            evidence_ids=tuple(item.evidence_id for item in evidence),
            groups=tuple(
                sorted(groups, key=lambda group: (group.period, group.kind, group.group_id))
            ),
            candidates=candidate_articles(evidence, checks),
            gaps=tuple(sorted(gaps)),
            status_counts=tuple(sorted(Counter(check.status for check in checks).items())),
            complete=not gaps,
            usage=_usage(all_checks),
        )
        records.save_batch(con, batch)
    return batch


def load_evidence_batch(
    client: EngineStore | str,
    batch_id: str,
) -> EvidenceBatch:
    if isinstance(client, str) and (
        not client or client in {".", ".."} or any(value in client for value in ("/", "\\", ":"))
    ):
        raise ContextError("Expected a client profile ID")
    store = client if isinstance(client, EngineStore) else EngineStore(client)
    if store.path.exists():
        with store.connect() as con:
            if records.table_exists(con, "reason_evidence_batches"):
                batch = records.read_batch(con, store.client_id, batch_id)
                if batch:
                    return batch
    raise ContextError(f"Evidence batch {batch_id!r} was not found for {store.client_id!r}")


def load_evidence(
    store: EngineStore, evidence_ids: tuple[str, ...]
) -> tuple[EvidenceRevision, ...]:
    if not evidence_ids:
        return ()
    if store.path.exists():
        with store.connect() as con:
            if records.table_exists(con, "reason_evidence_revisions"):
                evidence = records.read_evidence(con, store.client_id, evidence_ids)
                if len(evidence) == len(set(evidence_ids)):
                    return evidence
    raise ContextError(f"Evidence IDs were not found for {store.client_id!r}")


def load_source_check(store: EngineStore, check_id: str) -> SourceCheck:
    if store.path.exists():
        with store.connect() as con:
            if records.table_exists(con, "reason_evidence_checks"):
                row = con.execute(
                    "SELECT check_data FROM reason_evidence_checks "
                    "WHERE client_id = ? AND check_id = ?",
                    [store.client_id, check_id],
                ).fetchone()
                if row:
                    return SourceCheck.model_validate_json(row[0])
    raise ContextError(f"Source check {check_id!r} was not found for {store.client_id!r}")
