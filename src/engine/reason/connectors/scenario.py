"""Scenario packs: prepared signals that replace live network sources.

Used for demos and tests, so a run is reproducible and does not depend on
today's news. A pack is JSON:

    {"name": "...", "as_of": "2026-11-26", "note": "...",
     "drivers": {"<driver_id>": [{"kind": "news", "summary": "...", "data": {},
                                  "articles": [{"url": ..., "title": ..., "source": ...,
                                                "published": "2026-11-20T00:00:00Z",
                                                "snippet": "..."}]}]}}

Offline connectors (festival calendar) and measured effects still run normally.
Every pack article should be clearly marked as illustrative, never passed off as real news.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from engine.reason.schemas import ReasoningError, Signal


class ScenarioPack:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise ReasoningError(f"Scenario pack not found: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        self.path = path
        self.name = str(document.get("name") or path.stem)
        self.note = str(document.get("note") or "")
        self.as_of = date.fromisoformat(document["as_of"]) if document.get("as_of") else None
        self._drivers = document.get("drivers") or {}

    def signals(self, driver_id: str, kind: str) -> list[Signal]:
        return [
            Signal.model_validate({"driver_id": driver_id, "source_ref": "scenario", **item})
            for item in self._drivers.get(driver_id, [])
            if item.get("kind") == kind
        ]

    def has(self, driver_id: str) -> bool:
        return bool(self._drivers.get(driver_id))
