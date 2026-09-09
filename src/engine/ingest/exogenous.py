"""Context-feature ingestion.

Exogenous signals arrive in wide form (one column per driver) and are stored
long (one row per series-period-feature), so adding a signal later is a data
change rather than a schema change.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from engine.config.schema import ClientProfile
from engine.ingest.schema import IngestError

REQUIRED_KEYS = ("series_id", "period")


def read_exogenous(path: Path, client: ClientProfile) -> pl.DataFrame:
    """Read a wide exogenous file into the store's long format."""
    if not path.exists():
        raise IngestError(f"Exogenous file not found: {path}")

    features = client.exogenous.features
    if not features:
        raise IngestError(
            f"Client {client.client_id} lists no exogenous.features, so there is "
            "nothing to ingest. Add them to the client profile first."
        )

    raw = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)

    missing = [c for c in (*REQUIRED_KEYS, *features) if c not in raw.columns]
    if missing:
        raise IngestError(f"{path.name} is missing columns {missing}; found: {raw.columns}")

    frame = raw.select(
        pl.col("series_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("period").cast(pl.Date),
        *[pl.col(name).cast(pl.Float64) for name in features],
    )

    null_keys = frame.filter(pl.col("series_id").is_null() | pl.col("period").is_null())
    if not null_keys.is_empty():
        raise IngestError(f"{null_keys.height} exogenous rows have a null series_id or period")

    duplicates = frame.group_by("series_id", "period").len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise IngestError(
            f"{duplicates.height} duplicate (series_id, period) pairs in {path.name}; "
            "exogenous values must be unique per series and period"
        )

    return (
        frame.unpivot(
            index=["series_id", "period"],
            on=features,
            variable_name="feature",
            value_name="value",
        )
        .with_columns(
            pl.lit(client.client_id).alias("client_id"),
            pl.lit(path.name).alias("source_file"),
            pl.lit(datetime.now()).alias("ingested_at"),
        )
        .select(
            "client_id", "series_id", "period", "feature", "value", "source_file", "ingested_at"
        )
    )
