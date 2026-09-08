"""Canonical data model.

Every source file is mapped into these two frames. Nothing downstream of ingest
ever reads a source file directly, so a new source format means a new reader —
not changes to features, models, or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

DIM_SERIES_SCHEMA: dict[str, pl.DataType] = {
    "client_id": pl.Utf8,
    "series_id": pl.Utf8,
    "customer": pl.Utf8,
    "item_code": pl.Utf8,
    "description": pl.Utf8,
    "item_group": pl.Utf8,
    "source_sheet": pl.Utf8,
    "uom": pl.Utf8,
    # Present only in labelled synthetic data; used to validate the pattern
    # classifier and deliberately excluded from every feature set.
    "ground_truth_pattern": pl.Utf8,
    "source_file": pl.Utf8,
    "ingested_at": pl.Datetime("us"),
}

FACT_DEMAND_SCHEMA: dict[str, pl.DataType] = {
    "client_id": pl.Utf8,
    "series_id": pl.Utf8,
    "period": pl.Date,
    "quantity": pl.Float64,
    "source_file": pl.Utf8,
    "ingested_at": pl.Datetime("us"),
}

# Series attributes that must not vary within one series_id.
SERIES_INVARIANT_COLUMNS = ("customer", "item_code", "item_group", "source_sheet", "uom")


class IngestError(RuntimeError):
    """Raised when a source file violates the canonical contract."""


@dataclass(frozen=True)
class IngestReport:
    """Reconciliation figures, compared before and after canonical mapping."""

    source_file: str
    source_rows: int
    canonical_rows: int
    source_quantity_sum: float
    canonical_quantity_sum: float
    series_count: int
    period_min: str
    period_max: str

    @property
    def rows_reconcile(self) -> bool:
        return self.source_rows == self.canonical_rows

    @property
    def quantity_reconciles(self) -> bool:
        return abs(self.source_quantity_sum - self.canonical_quantity_sum) < 1e-6

    def raise_if_lossy(self) -> None:
        if not self.rows_reconcile:
            raise IngestError(
                f"Row-count mismatch for {self.source_file}: "
                f"{self.source_rows} read, {self.canonical_rows} mapped"
            )
        if not self.quantity_reconciles:
            raise IngestError(
                f"Quantity-sum mismatch for {self.source_file}: "
                f"{self.source_quantity_sum} read, {self.canonical_quantity_sum} mapped"
            )


@dataclass(frozen=True)
class IngestResult:
    dim_series: pl.DataFrame
    fact_demand: pl.DataFrame
    report: IngestReport


def validate_canonical(dim: pl.DataFrame, fact: pl.DataFrame) -> None:
    """Fail loudly on any violation of the canonical contract."""
    if fact.is_empty():
        raise IngestError("No demand rows found in source file")

    null_keys = fact.filter(
        pl.col("series_id").is_null() | pl.col("period").is_null() | pl.col("quantity").is_null()
    )
    if not null_keys.is_empty():
        raise IngestError(
            f"{null_keys.height} demand rows have a null series_id, period, or quantity; "
            f"first offending row: {null_keys.head(1).to_dicts()[0]}"
        )

    negatives = fact.filter(pl.col("quantity") < 0)
    if not negatives.is_empty():
        raise IngestError(
            f"{negatives.height} demand rows have negative quantity; "
            f"first offending row: {negatives.head(1).to_dicts()[0]}"
        )

    duplicates = (
        fact.group_by("series_id", "period").len().filter(pl.col("len") > 1).sort("series_id")
    )
    if not duplicates.is_empty():
        raise IngestError(
            f"{duplicates.height} duplicate (series_id, period) pairs; "
            f"first offending pair: {duplicates.head(1).to_dicts()[0]}"
        )

    dim_duplicates = dim.group_by("series_id").len().filter(pl.col("len") > 1)
    if not dim_duplicates.is_empty():
        raise IngestError(
            f"{dim_duplicates.height} series_id values map to more than one attribute set — "
            "series attributes must be constant across the series' history"
        )

    orphans = fact.join(dim.select("series_id"), on="series_id", how="anti")
    if not orphans.is_empty():
        raise IngestError(f"{orphans.height} demand rows reference an unknown series_id")
