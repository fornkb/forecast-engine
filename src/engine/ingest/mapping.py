"""Shared source-to-canonical mapping.

Readers differ only in how they get a table off disk. Everything after that —
renaming, casting, collapsing series attributes, validating, reconciling — is
identical, and lives here so a new file format cannot quietly acquire different
semantics.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from engine.config.schema import ClientProfile
from engine.ingest.schema import (
    DIM_SERIES_SCHEMA,
    FACT_DEMAND_SCHEMA,
    SERIES_INVARIANT_COLUMNS,
    IngestError,
    IngestReport,
    IngestResult,
    validate_canonical,
)


def to_canonical(raw: pl.DataFrame, client: ClientProfile, source_file: str) -> IngestResult:
    """Map a raw source table onto dim_series + fact_demand, or fail loudly."""
    mapping = client.source.column_map.model_dump()

    missing = [
        source_col
        for field, source_col in mapping.items()
        if source_col is not None and source_col not in raw.columns
    ]
    if missing:
        raise IngestError(
            f"{source_file} is missing expected columns {missing}; "
            f"found columns: {raw.columns}. Fix source.column_map in the client profile."
        )

    source_rows = raw.height
    source_quantity_sum = float(raw[mapping["quantity"]].cast(pl.Float64).sum() or 0.0)
    ingested_at = datetime.now()

    renames = {
        source_col: field
        for field, source_col in mapping.items()
        if source_col is not None and field != "ground_truth_pattern" and source_col != field
    }
    frame = raw.rename(renames)

    pattern_col = mapping.get("ground_truth_pattern")
    if pattern_col and pattern_col in raw.columns:
        if pattern_col != "ground_truth_pattern":
            frame = frame.rename({pattern_col: "ground_truth_pattern"})
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias("ground_truth_pattern"))

    frame = frame.with_columns(
        pl.col("series_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("period").cast(pl.Date),
        pl.col("quantity").cast(pl.Float64),
        pl.lit(client.client_id).alias("client_id"),
        pl.lit(source_file).alias("source_file"),
        pl.lit(ingested_at).alias("ingested_at"),
    )

    fact_demand = frame.select(list(FACT_DEMAND_SCHEMA)).sort("series_id", "period")

    dim_series = (
        frame.group_by("series_id")
        .agg(
            pl.col("client_id").first(),
            *[pl.col(col).first() for col in SERIES_INVARIANT_COLUMNS],
            pl.col("description").first(),
            pl.col("ground_truth_pattern").first(),
            pl.col("source_file").first(),
            pl.col("ingested_at").first(),
            *[pl.col(col).n_unique().alias(f"_n_{col}") for col in SERIES_INVARIANT_COLUMNS],
        )
        .sort("series_id")
    )

    inconsistent = dim_series.filter(
        pl.any_horizontal([pl.col(f"_n_{col}") > 1 for col in SERIES_INVARIANT_COLUMNS])
    )
    if not inconsistent.is_empty():
        raise IngestError(
            f"{inconsistent.height} series have attributes that change over time "
            f"(e.g. {inconsistent.head(1)['series_id'].item()}); series attributes must be "
            "constant. Fix the source extract or split the series."
        )

    dim_series = dim_series.select(list(DIM_SERIES_SCHEMA))
    validate_canonical(dim_series, fact_demand)

    report = IngestReport(
        source_file=source_file,
        source_rows=source_rows,
        canonical_rows=fact_demand.height,
        source_quantity_sum=source_quantity_sum,
        canonical_quantity_sum=float(fact_demand["quantity"].sum()),
        series_count=dim_series.height,
        period_min=str(fact_demand["period"].min()),
        period_max=str(fact_demand["period"].max()),
    )
    report.raise_if_lossy()

    return IngestResult(dim_series=dim_series, fact_demand=fact_demand, report=report)
