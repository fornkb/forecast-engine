"""Shared protocol boundaries and normalization for evidence adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from engine.reason.config import OPERATIONAL_SCENARIO_FIELDS
from engine.reason.schemas import CoverageStatus, EvidenceRevision, MonthCoverage, content_digest


@dataclass(frozen=True)
class HttpRequest:
    url: str
    params: dict[str, str] = field(default_factory=dict, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes

    def json(self) -> dict:
        document = json.loads(self.body)
        if not isinstance(document, dict) or document.get("error"):
            raise ValueError("Expected a provider result object")
        return document


HttpTransport = Callable[[HttpRequest, float], HttpResponse]


class ProviderError(Exception):
    def __init__(self, reason: str, *, status: str | None = None, transient: bool = False):
        super().__init__(reason)
        self.status = status
        self.transient = transient


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_get(request: HttpRequest, timeout: float) -> HttpResponse:
    url = request.url
    if request.params:
        url = f"{url}?{urlencode(request.params)}"
    try:
        with build_opener(_NoRedirect()).open(
            Request(url, headers=request.headers), timeout=timeout
        ) as response:
            limit = 8 * 1024 * 1024
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ProviderError("response_size_limit")
            return HttpResponse(response.status, body)
    except HTTPError as exc:
        exc.close()
        return HttpResponse(exc.code, b"")
    except (TimeoutError, URLError, OSError):
        raise ProviderError("network_error", transient=True) from None


@dataclass(frozen=True)
class ParsedEvidence:
    revisions: tuple[EvidenceRevision, ...] = ()
    coverage: CoverageStatus = "complete"
    reason: str | None = None
    covered_from: datetime | None = None
    covered_to: datetime | None = None
    months: tuple[MonthCoverage, ...] = ()
    more_pages: bool = False


def optional_instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        instant = datetime.fromisoformat(value)
    except ValueError:
        return None
    return instant.astimezone(UTC) if instant.utcoffset() is not None else None


def next_month(period: date) -> date:
    return date(period.year + (period.month == 12), period.month % 12 + 1, 1)


def evidence_revision(
    client_id: str,
    source_kind: str,
    source_id: str,
    natural_id: str,
    captured_at: datetime,
    *,
    released_at: datetime | None = None,
    **fields,
) -> EvidenceRevision:
    record_id = "record_" + content_digest([client_id, source_kind, source_id, natural_id])
    revision = EvidenceRevision(
        client_id=client_id,
        evidence_id="",
        record_id=record_id,
        revision=0,
        source_kind=source_kind,
        source_id=source_id,
        captured_at=captured_at,
        first_known_at=max(captured_at, released_at or captured_at),
        **fields,
    )
    payload = revision.model_dump(
        mode="json", exclude={"captured_at", "first_known_at", "evidence_id"}
    )
    if operational := payload["operational"]:
        # Phase 2 records with no scenario metadata retain their existing identities.
        for name in OPERATIONAL_SCENARIO_FIELDS:
            if operational[name] is None:
                del operational[name]
    return revision.model_copy(
        update={"evidence_id": "evidence_" + content_digest([payload, released_at])}
    )
