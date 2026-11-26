"""Tiny HTTP GET boundary shared by connectors. Injectable, so tests never touch the network."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

USER_AGENT = "forecast-engine-reasoning/2"


@dataclass(frozen=True)
class HttpRequest:
    url: str
    params: dict[str, str] = field(default_factory=dict, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes

    def text(self) -> str:
        try:
            return self.body.decode("utf-8")
        except UnicodeDecodeError:
            return self.body.decode("cp1252", errors="replace")

    def json(self) -> dict:
        document = json.loads(self.body)
        if not isinstance(document, dict):
            raise ProviderError("expected a JSON object")
        return document


HttpTransport = Callable[[HttpRequest, float], HttpResponse]


class ProviderError(Exception):
    """A source failed. The run continues and shows the gap."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_get(request: HttpRequest, timeout: float) -> HttpResponse:
    url = f"{request.url}?{urlencode(request.params)}" if request.params else request.url
    headers = {"User-Agent": USER_AGENT, **request.headers}
    try:
        with build_opener(_NoRedirect()).open(Request(url, headers=headers), timeout=timeout) as r:
            limit = 8 * 1024 * 1024
            body = r.read(limit + 1)
            if len(body) > limit:
                raise ProviderError("response too large")
            return HttpResponse(r.status, body)
    except HTTPError as exc:
        exc.close()
        return HttpResponse(exc.code, b"")
    except (TimeoutError, URLError, OSError) as exc:
        raise ProviderError(f"network error: {type(exc).__name__}") from None


def get_ok(http: HttpTransport, request: HttpRequest, timeout: float) -> HttpResponse:
    response = http(request, timeout)
    if response.status != 200:
        raise ProviderError(f"HTTP {response.status} from {request.url}")
    return response
