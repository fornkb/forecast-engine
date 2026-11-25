"""Brave News Search. Article URLs are stored only as evidence citations."""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime, timedelta
from html import unescape
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from engine.reason.config import BraveNewsConfig
from engine.reason.providers import (
    HttpRequest,
    ParsedEvidence,
    ProviderError,
    evidence_revision,
    optional_instant,
)
from engine.reason.schemas import NewsArticle, QueryScope, content_digest

BRAVE_URL = "https://api.search.brave.com/res/v1/news/search"
_TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid", "msclkid"}


def news_request(config: BraveNewsConfig, scope: QueryScope) -> HttpRequest:
    token = os.environ.get(config.api_key_env)
    if not token:
        raise ProviderError("missing_credentials")
    filters = dict(scope.filters)
    return HttpRequest(
        BRAVE_URL,
        {
            "q": " ".join(scope.public_aliases),
            "freshness": filters["freshness"],
            "count": str(config.page_size),
            "offset": str(scope.page),
            "spellcheck": "false",
        },
        {"X-Subscription-Token": token, "Accept": "application/json"},
    )


def canonical_url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Invalid article URL")
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None:
        raise ValueError("Invalid article URL")
    host = parts.hostname.lower().rstrip(".")
    if ":" in host:
        host = f"[{host}]"
    if parts.port and (parts.scheme, parts.port) not in {("http", 80), ("https", 443)}:
        host += f":{parts.port}"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING
        )
    )
    return urlunsplit((parts.scheme, host, parts.path or "/", query, ""))


def _text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unescape(re.sub(r"<[^>]*>", "", value[: limit * 2])).split())[:limit]


def parse_news(
    payload: dict, config: BraveNewsConfig, scope: QueryScope, captured_at: datetime
) -> ParsedEvidence:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Missing Brave results")
    primary_domains = {
        domain.lower() for search in config.searches for domain in search.primary_domains
    }
    revisions = {}
    skipped = 0
    for raw in results[: config.page_size]:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        try:
            url = canonical_url(raw.get("url"))
        except ValueError:
            skipped += 1
            continue
        host = urlsplit(url).hostname or ""
        title = _text(raw.get("title"), 500)
        excerpt = _text(raw.get("description"), 2000)
        if not title:
            skipped += 1
            continue
        profile = raw.get("profile")
        publisher = _text(profile.get("name"), 160) if isinstance(profile, dict) else ""
        page_date = _text(raw.get("page_age"), 80) or None
        # Brave's page_age can also be a modification date. It is not publication time.
        published = optional_instant(raw.get("published_at"))
        event = _text(raw.get("event_id"), 160) or None
        event_start = optional_instant(raw.get("event_start"))
        event_end = optional_instant(raw.get("event_end"))
        location = _text(raw.get("location"), 160) or None
        if event_start and event_end and event_end <= event_start:
            event_start = event_end = None
        duplicate_key = content_digest([title.casefold(), excerpt.casefold()] if excerpt else [url])
        article = NewsArticle(
            title=title,
            publisher=publisher or host,
            primary=any(
                host == domain or host.endswith("." + domain) for domain in primary_domains
            ),
            provider_page_date=page_date,
            provider_fetched_at=optional_instant(raw.get("page_fetched")),
            duplicate_key=duplicate_key,
        )
        available = published or optional_instant(page_date)
        if available is None and page_date:
            try:
                day = date.fromisoformat(page_date)
                if day > captured_at.date():
                    available = datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC)
            except (ValueError, OverflowError):
                pass
        item = evidence_revision(
            scope.client_id,
            "news",
            "brave",
            url,
            captured_at,
            source_ref=url,
            source_event_id=event,
            issued_at=published,
            valid_from=event_start,
            valid_to=event_end,
            location_id=location,
            released_at=available,
            excerpt=excerpt or None,
            news=article,
        )
        # Index fetch timestamps change without an article revision.
        revision_key = article.model_dump(exclude={"provider_fetched_at"})
        item = item.model_copy(
            update={
                "evidence_id": "evidence_"
                + content_digest(
                    [
                        item.record_id,
                        revision_key,
                        excerpt,
                        published,
                        event,
                        event_start,
                        event_end,
                        location,
                    ]
                )
            }
        )
        revisions.setdefault(item.record_id, item)
    query = payload.get("query")
    explicit_more = query.get("more_results_available") if isinstance(query, dict) else None
    more = explicit_more if isinstance(explicit_more, bool) else len(results) >= config.page_size
    start, end = dict(scope.filters)["freshness"].split("to")
    return ParsedEvidence(
        tuple(revisions.values()),
        coverage="partial" if skipped or len(results) > config.page_size else "complete",
        reason="invalid_or_truncated_results"
        if skipped or len(results) > config.page_size
        else None,
        covered_from=datetime.fromisoformat(start).replace(tzinfo=UTC),
        covered_to=datetime.fromisoformat(end).replace(tzinfo=UTC) + timedelta(days=1),
        more_pages=more,
    )
