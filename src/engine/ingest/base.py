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
    """Maps one source file into the canonical dim_series / fact_demand frames."""

    name: ClassVar[str]

    @abstractmethod
    def read(self, path: Path, client: ClientProfile) -> IngestResult: ...


def register_reader(cls: type[SourceReader]) -> type[SourceReader]:
    _READERS[cls.name] = cls
    return cls


def get_reader(name: str) -> SourceReader:
    if name not in _READERS:
        raise KeyError(f"Unknown reader {name!r}; registered readers: {sorted(_READERS)}")
    return _READERS[name]()
