"""News search: GDELT (free, no key, titles only) or Brave (paid key, with snippets).

Queries are built only from public words in the driver template and profile
(material names, regions, topics). No customer names or ERP numbers are sent.
Article URLs are citations; they are never fetched.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from html import unescape
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from engine.reason.connectors.http import HttpRequest, HttpResponse, ProviderError, get_ok, http_get
from engine.reason.schemas import Article, Signal

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
BRAVE_URL = "https://api.search.brave.com/res/v1/news/search"
_TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid", "msclkid"}
# GDELT allows about one request every 5 seconds per client and answers bursts with HTTP 429.
GDELT_GAP_SECONDS = 6.0
# After GDELT keeps refusing, stop asking for a while instead of retrying every driver.
GDELT_PAUSE_SECONDS = 600.0
GDELT_LIMITED = (
    "GDELT news is limiting requests from this network. Wait about 10 minutes and try again, "
    "or use the Brave news provider (sources.news.provider: brave)."
)
_gdelt_lock = threading.Lock()
_gdelt_last = 0.0
_gdelt_paused_until = 0.0


def canonical_url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Invalid article URL")
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Invalid article URL")
    query = urlencode(
        sorted(
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING
        )
    )
    return urlunsplit((parts.scheme, parts.hostname.lower(), parts.path or "/", query, ""))


def clean(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unescape(re.sub(r"<[^>]*>", "", value[: limit * 2])).split())[:limit]


def title_key(title: str) -> str:
    """Syndicated copies of one story share a title; keep one."""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def dedupe(articles: list[Article]) -> list[Article]:
    seen, kept = set(), []
    for article in articles:
        key = title_key(article.title)
        if len(key) < 12 or key in seen:
            continue
        seen.add(key)
        kept.append(article)
    return kept


def _gdelt_get(ctx, request: HttpRequest, timeout: float) -> HttpResponse:
    """GDELT politely: space real requests out, retry twice after a 429, then pause."""
    global _gdelt_last, _gdelt_paused_until
    if ctx.http is not http_get:  # injected transports (tests, replays) need no pacing
        return get_ok(ctx.http, request, timeout)
    if time.monotonic() < _gdelt_paused_until:
        raise ProviderError(GDELT_LIMITED)
    response = None
    for attempt in range(3):
        with _gdelt_lock:
            wait = _gdelt_last + GDELT_GAP_SECONDS * (attempt + 1) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            response = ctx.http(request, timeout)
            _gdelt_last = time.monotonic()
        if response.status != 429:
            break
    if response.status == 429:
        _gdelt_paused_until = time.monotonic() + GDELT_PAUSE_SECONDS
        raise ProviderError(GDELT_LIMITED)
    if response.status != 200:
        raise ProviderError(f"HTTP {response.status} from {request.url}")
    return response


def _gdelt(query: str, source, ctx) -> list[Article]:
    terms = [query]
    if source.language:
        terms.append(f"sourcelang:{source.language}")
    if source.source_country:
        terms.append(f"sourcecountry:{source.source_country}")
    request = HttpRequest(
        GDELT_URL,
        {
            "query": " ".join(terms),
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(75, source.max_articles_per_driver * 4)),
            "timespan": f"{source.lookback_days}d",
            "sort": "hybridrel",
        },
    )
    body = _gdelt_get(ctx, request, source.timeout_seconds).text().strip()
    if not body:
        return []
    if not body.startswith("{"):
        # GDELT answers a bad query with a plain-text message.
        raise ProviderError(f"GDELT: {body[:150]}")
    articles = []
    for raw in json.loads(body).get("articles", []):
        try:
            url = canonical_url(raw.get("url"))
        except ValueError:
            continue
        published = None
        with suppress(ValueError):
            published = datetime.strptime(raw.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(
                tzinfo=UTC
            )
        title = clean(raw.get("title"), 300)
        if title:
            articles.append(
                Article(
                    url=url,
                    title=title,
                    source=clean(raw.get("domain"), 120) or (urlsplit(url).hostname or ""),
                    published=published,
                    country=clean(raw.get("sourcecountry"), 60),
                )
            )
    return articles


def _brave(query: str, source, ctx) -> list[Article]:
    token = os.environ.get(source.api_key_env)
    if not token:
        raise ProviderError(f"missing Brave key: set {source.api_key_env}")
    request = HttpRequest(
        BRAVE_URL,
        {
            "q": query,
            "count": str(min(50, source.max_articles_per_driver * 3)),
            "freshness": "pm" if source.lookback_days > 7 else "pw",
            "spellcheck": "false",
        },
        {"X-Subscription-Token": token, "Accept": "application/json"},
    )
    articles = []
    for raw in get_ok(ctx.http, request, source.timeout_seconds).json().get("results", []):
        try:
            url = canonical_url(raw.get("url"))
        except ValueError:
            continue
        title = clean(raw.get("title"), 300)
        profile = raw.get("profile") if isinstance(raw.get("profile"), dict) else {}
        published = None
        for key in ("page_age", "published_at"):
            try:
                value = datetime.fromisoformat(str(raw.get(key)))
                published = value if value.tzinfo else value.replace(tzinfo=UTC)
                break
            except ValueError:
                continue
        if title:
            articles.append(
                Article(
                    url=url,
                    title=title,
                    source=clean(profile.get("name"), 120) or (urlsplit(url).hostname or ""),
                    published=published,
                    snippet=clean(raw.get("description"), 400),
                )
            )
    return articles


def fetch(driver, ctx) -> list[Signal]:
    source = ctx.sources.news
    if source.provider == "none" or not driver.query:
        return []
    found = (
        _gdelt(driver.query, source, ctx)
        if source.provider == "gdelt"
        else _brave(driver.query, source, ctx)
    )
    articles = dedupe(found)[: source.max_articles_per_driver]
    return [
        Signal(
            driver_id=driver.id,
            kind="news",
            summary=f"{len(articles)} recent articles for '{driver.query}'",
            data={"query": driver.query, "provider": source.provider},
            articles=articles,
            source_ref=source.provider,
        )
    ]


def scan(driver, ctx) -> list[Signal]:
    """The open scan: one news search per topic, so one failed search does not hide the rest."""
    source = ctx.sources.news
    if source.provider == "none":
        return []
    search = _gdelt if source.provider == "gdelt" else _brave
    signals, seen = [], set()
    for query in driver.params.get("queries", []):
        try:
            found = dedupe(search(query, source, ctx))
        except (ProviderError, ValueError) as exc:
            signals.append(
                Signal(
                    driver_id=driver.id,
                    kind="news",
                    summary=f"open search '{query}' failed",
                    data={"query": query, "provider": source.provider},
                    ok=False,
                    error=str(exc)[:200],
                    source_ref=source.provider,
                )
            )
            continue
        articles = [a for a in found if title_key(a.title) not in seen]
        articles = articles[: source.max_articles_per_driver]
        seen.update(title_key(a.title) for a in articles)
        signals.append(
            Signal(
                driver_id=driver.id,
                kind="news",
                summary=f"{len(articles)} recent articles for open search '{query}'",
                data={"query": query, "provider": source.provider},
                articles=articles,
                source_ref=source.provider,
            )
        )
    return signals
