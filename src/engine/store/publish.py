"""Forecast write-back to the ERP database.

A file is a snapshot handed to a person; this is how a forecast reaches a
system. The engine's own store stays the source of truth — publishing copies
one completed run into a schema the ERP reads, and copying it again replaces
that run rather than duplicating it.

Reading and writing use different roles and different DSNs on purpose: nothing
on this path has any route to the ERP's own sales history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import polars as pl

from engine.config.schema import ClientProfile
from engine.ingest.postgres_reader import redact, scrub
from engine.store.duckdb_store import EngineStore

FORECAST_TABLE = "demand_forecast"
QUANTILE_TABLE = "demand_forecast_quantile"
RUN_TABLE = "forecast_run"

FORECAST_COLUMNS = [
    "run_id",
    "client_id",
    "series_id",
    "cutoff_period",
    "period",
    "horizon",
    "model",
    "forecast",
]
QUANTILE_COLUMNS = [
    "run_id",
    "client_id",
    "series_id",
    "period",
    "horizon",
    "model",
    "level",
    "value",
]


class PublishError(RuntimeError):
    """Raised when a forecast cannot be written to the target database."""


@dataclass(frozen=True)
class PublishReport:
    run_id: int
    target: str
    forecast_rows: int
    quantile_rows: int
    series_count: int
    cutoff_period: str
    horizon: int


def _connect(dsn: str):
    try:
        import adbc_driver_postgresql.dbapi as pg
    except ModuleNotFoundError as exc:
        raise PublishError(
            'The PostgreSQL driver is not installed. Install it with: pip install -e ".[postgres]"'
        ) from exc
    try:
        return pg.connect(dsn)
    except Exception as exc:
        raise PublishError(f"Could not connect to {redact(dsn)}: {scrub(str(exc), dsn)}") from None


def _ingest(cursor, schema: str, table: str, frame: pl.DataFrame, columns: list[str]) -> int:
    if frame.is_empty():
        return 0
    cursor.adbc_ingest(
        table,
        frame.select(columns).to_arrow(),
        mode="append",
        db_schema_name=schema,
    )
    return frame.height


def publish_forecasts(store: EngineStore, client: ClientProfile, run_id: int) -> PublishReport:
    """Copy one forecast run into the ERP-readable output schema.

    The whole run — summary, point forecasts and quantiles — goes in a single
    transaction. A partial publish would leave the ERP reading a forecast that
    covers some items and not others, which is worse than no forecast at all.
    """
    config = client.output.postgres
    if config is None:
        raise PublishError(
            f"Client {client.client_id} has no output.postgres block, so there is "
            "nowhere to publish to. Add one to the client profile."
        )

    dsn = os.environ.get(config.dsn_env)
    if not dsn:
        raise PublishError(
            f"Environment variable {config.dsn_env} is not set, so there is no "
            "connection to the output database. Set it to a PostgreSQL DSN, e.g. "
            "postgresql://engine_writer:<password>@localhost:5432/forecast_erp"
        )

    forecasts = store.load_forecasts(run_id)
    if forecasts.is_empty():
        raise PublishError(
            f"Run {run_id} has no forecasts in the store. Run `engine forecast` first."
        )
    quantiles = store.load_forecast_quantiles(run_id)

    schema = config.target_schema
    cutoff = forecasts["cutoff_period"].max()
    summary = pl.DataFrame(
        {
            "run_id": [run_id],
            "client_id": [client.client_id],
            "cutoff_period": [cutoff],
            "horizon": [int(forecasts["horizon"].max())],
            "series_count": [forecasts["series_id"].n_unique()],
            "row_count": [forecasts.height],
        }
    ).with_columns(
        pl.col("run_id").cast(pl.Int64),
        pl.col("horizon").cast(pl.Int32),
        pl.col("series_count").cast(pl.Int32),
        pl.col("row_count").cast(pl.Int32),
    )

    try:
        with _connect(dsn) as conn, conn.cursor() as cursor:
            # Republishing a run replaces it. Deleting first makes the command
            # safe to re-run after a failure, rather than leaving duplicates
            # that the primary key would reject on the second attempt anyway.
            for table in (QUANTILE_TABLE, FORECAST_TABLE, RUN_TABLE):
                cursor.execute(
                    f"DELETE FROM {schema}.{table} WHERE client_id = $1 AND run_id = $2",
                    (client.client_id, run_id),
                )
            _ingest(cursor, schema, RUN_TABLE, summary, list(summary.columns))
            forecast_rows = _ingest(cursor, schema, FORECAST_TABLE, forecasts, FORECAST_COLUMNS)
            quantile_rows = _ingest(cursor, schema, QUANTILE_TABLE, quantiles, QUANTILE_COLUMNS)
            conn.commit()
    except PublishError:
        raise
    except Exception as exc:
        raise PublishError(
            f"Publishing run {run_id} to {redact(dsn)} failed: {scrub(str(exc), dsn)}"
        ) from None

    return PublishReport(
        run_id=run_id,
        target=f"{schema}.{FORECAST_TABLE}",
        forecast_rows=forecast_rows,
        quantile_rows=quantile_rows,
        series_count=forecasts["series_id"].n_unique(),
        cutoff_period=str(cutoff),
        horizon=int(forecasts["horizon"].max()),
    )
