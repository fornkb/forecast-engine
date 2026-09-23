"""Long-format panel reader.

For sources that already arrive as one row per series-period — a database
extract, a prepared research dataset, an ERP view materialised to Parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from engine.config.schema import ClientProfile
from engine.ingest.base import SourceReader, register_reader
from engine.ingest.mapping import to_canonical
from engine.ingest.schema import IngestError, IngestResult


@register_reader
class ParquetPanelReader(SourceReader):
    name = "parquet_panel"

    def read(self, path: Path | None, client: ClientProfile) -> IngestResult:
        if path is None or not path.exists():
            raise IngestError(f"Source file not found: {path}")
        return to_canonical(pl.read_parquet(path), client, path.name)


@register_reader
class CsvPanelReader(SourceReader):
    name = "csv_panel"

    def read(self, path: Path | None, client: ClientProfile) -> IngestResult:
        if path is None or not path.exists():
            raise IngestError(f"Source file not found: {path}")
        return to_canonical(pl.read_csv(path, try_parse_dates=True), client, path.name)
