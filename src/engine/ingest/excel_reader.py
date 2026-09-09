"""Excel demand-history reader."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from engine.config.schema import ClientProfile
from engine.ingest.base import SourceReader, register_reader
from engine.ingest.mapping import to_canonical
from engine.ingest.schema import IngestError, IngestResult


@register_reader
class ExcelDemandReader(SourceReader):
    """Reads a wide demand-history workbook into the canonical model.

    One row per series-period is expected; series attributes repeat on every row
    and are collapsed into `dim_series`.
    """

    name = "excel_demand"

    def read(self, path: Path, client: ClientProfile) -> IngestResult:
        if not path.exists():
            raise IngestError(f"Source file not found: {path}")

        sheet = client.source.sheet
        raw = pl.read_excel(path, sheet_name=sheet) if sheet else pl.read_excel(path)
        return to_canonical(raw, client, path.name)
