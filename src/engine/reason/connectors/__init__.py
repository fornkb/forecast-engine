"""Data connectors. Adding a new kind of source = one module + one line in CONNECTORS.

A connector is `fetch(driver: ResolvedDriver, ctx: Context) -> list[Signal]`.
It raises ProviderError on failure; the run records the gap and continues.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

from engine.reason.config import SourcesConfig
from engine.reason.connectors import calendar, enso, news, weather
from engine.reason.connectors.http import HttpTransport, ProviderError, http_get
from engine.reason.schemas import Signal


@dataclass(frozen=True)
class Context:
    as_of: datetime
    timezone: str
    months: list[date]
    sources: SourcesConfig
    http: HttpTransport = http_get
    history: dict[str, dict[date, float]] = field(default_factory=dict)


Connector = Callable[..., list[Signal]]

CONNECTORS: dict[str, Connector] = {
    "news": news.fetch,
    "weather": weather.fetch,
    "enso": enso.fetch,
    "calendar": calendar.fetch,
    "scan": news.scan,
}
# Connectors that call the internet; a scenario pack replaces exactly these.
NETWORK = {"news", "weather", "enso", "scan"}

__all__ = ["CONNECTORS", "NETWORK", "Context", "ProviderError", "http_get"]
