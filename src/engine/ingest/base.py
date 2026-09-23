"""Source reader interface and registry.

A new source format (real ERP extract, CSV export, database view) is added by
implementing `SourceReader` and registering it — no other module changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from engine.config.schema import ClientProfile
from engine.ingest.schema import IngestResult

_READERS: dict[str, type[SourceReader]] = {}


class SourceReader(ABC):
    """Maps one source into the canonical dim_series / fact_demand frames."""

    name: ClassVar[str]
    # A database reader is located by its client profile, not by a path. The CLI
    # reads this to decide whether --file is required, so neither a missing path
    # nor a pointless one reaches a reader.
    requires_path: ClassVar[bool] = True

    @abstractmethod
    def read(self, path: Path | None, client: ClientProfile) -> IngestResult: ...


def register_reader(cls: type[SourceReader]) -> type[SourceReader]:
    _READERS[cls.name] = cls
    return cls


def get_reader(name: str) -> SourceReader:
    if name not in _READERS:
        raise KeyError(f"Unknown reader {name!r}; registered readers: {sorted(_READERS)}")
    return _READERS[name]()
