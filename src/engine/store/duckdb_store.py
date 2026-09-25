"""Canonical store: one embedded DuckDB database per client.

Per-client database files (rather than a shared database with a tenant column)
keep tenant isolation a property of the filesystem layout, which is the boundary
the multi-tenancy design calls for later.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from engine.config.loader import project_root
from engine.config.schema import quantile_column

DDL = """
CREATE SEQUENCE IF NOT EXISTS run_id_seq START 1;

CREATE TABLE IF NOT EXISTS dim_series (
    client_id            VARCHAR NOT NULL,
    series_id            VARCHAR NOT NULL,
    customer             VARCHAR,
    item_code            VARCHAR,
    description          VARCHAR,
    item_group           VARCHAR,
    source_sheet         VARCHAR,
    uom                  VARCHAR,
    ground_truth_pattern VARCHAR,
    source_file          VARCHAR NOT NULL,
    ingested_at          TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_demand (
    client_id   VARCHAR NOT NULL,
    series_id   VARCHAR NOT NULL,
    period      DATE NOT NULL,
    quantity    DOUBLE NOT NULL,
    source_file VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL
);

-- Long format: one row per series-period-feature, so a client can add context
-- signals without a schema migration.
CREATE TABLE IF NOT EXISTS fact_exogenous (
    client_id   VARCHAR NOT NULL,
    series_id   VARCHAR NOT NULL,
    period      DATE NOT NULL,
    feature     VARCHAR NOT NULL,
    value       DOUBLE,
    source_file VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS series_profile (
    client_id         VARCHAR NOT NULL,
    series_id         VARCHAR NOT NULL,
    n_periods         BIGINT,
    n_zeros           BIGINT,
    zero_fraction     DOUBLE,
    mean_demand       DOUBLE,
    total_volume      DOUBLE,
    cv                DOUBLE,
    adi               DOUBLE,
    cv2               DOUBLE,
    trend_strength    DOUBLE,
    seasonal_strength DOUBLE,
    residual_cv       DOUBLE,
    slope             DOUBLE,
    pattern           VARCHAR,
    volume_segment    VARCHAR,
    computed_at       TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     BIGINT PRIMARY KEY,
    client_id  VARCHAR NOT NULL,
    run_type   VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    config     VARCHAR
);

CREATE TABLE IF NOT EXISTS backtest_predictions (
    run_id        BIGINT NOT NULL,
    model         VARCHAR NOT NULL,
    origin        INTEGER NOT NULL,
    cutoff_period DATE NOT NULL,
    series_id     VARCHAR NOT NULL,
    period        DATE NOT NULL,
    horizon       INTEGER NOT NULL,
    actual        DOUBLE,
    forecast      DOUBLE
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    run_id      BIGINT NOT NULL,
    model       VARCHAR NOT NULL,
    scope_type  VARCHAR NOT NULL,
    scope_value VARCHAR NOT NULL,
    metric      VARCHAR NOT NULL,
    value       DOUBLE,
    n_obs       BIGINT
);

CREATE TABLE IF NOT EXISTS model_selection (
    run_id    BIGINT NOT NULL,
    series_id VARCHAR NOT NULL,
    model     VARCHAR NOT NULL,
    score     DOUBLE,
    metric    VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id        BIGINT NOT NULL,
    client_id     VARCHAR NOT NULL,
    series_id     VARCHAR NOT NULL,
    cutoff_period DATE NOT NULL,
    period        DATE NOT NULL,
    horizon       INTEGER NOT NULL,
    model         VARCHAR NOT NULL,
    forecast      DOUBLE
);

-- Long format so the set of quantile levels can change per client without a
-- schema migration. Consumed by the decision layer, which needs a distribution
-- rather than a point to set safety stock against a service level.
CREATE TABLE IF NOT EXISTS forecast_quantiles (
    run_id    BIGINT NOT NULL,
    client_id VARCHAR NOT NULL,
    series_id VARCHAR NOT NULL,
    period    DATE NOT NULL,
    horizon   INTEGER NOT NULL,
    model     VARCHAR NOT NULL,
    level     DOUBLE NOT NULL,
    value     DOUBLE
);

-- The backtest computed these all along but only kept the point forecast, which
-- is enough to score accuracy and not enough to score a decision: choosing a
-- stock level needs the distribution at the origin the decision was made from.
CREATE TABLE IF NOT EXISTS backtest_quantiles (
    run_id        BIGINT NOT NULL,
    model         VARCHAR NOT NULL,
    origin        INTEGER NOT NULL,
    cutoff_period DATE NOT NULL,
    series_id     VARCHAR NOT NULL,
    period        DATE NOT NULL,
    horizon       INTEGER NOT NULL,
    level         DOUBLE NOT NULL,
    actual        DOUBLE,
    value         DOUBLE
);

-- Long, like backtest_metrics: one row per model per scope per metric, so a new
-- decision metric is a row rather than a column.
CREATE TABLE IF NOT EXISTS decision_metrics (
    run_id      BIGINT NOT NULL,
    model       VARCHAR NOT NULL,
    scope_type  VARCHAR NOT NULL,
    scope_value VARCHAR NOT NULL,
    metric      VARCHAR NOT NULL,
    value       DOUBLE,
    n_obs       BIGINT
);

-- What to actually do, per item: the numbers an ERP's reorder screen needs.
CREATE TABLE IF NOT EXISTS decision_policy (
    run_id          BIGINT NOT NULL,
    client_id       VARCHAR NOT NULL,
    series_id       VARCHAR NOT NULL,
    model           VARCHAR NOT NULL,
    service_level   DOUBLE NOT NULL,
    window_periods  INTEGER NOT NULL,
    expected_demand DOUBLE,
    target_stock    DOUBLE,
    safety_stock    DOUBLE,
    reorder_point   DOUBLE
);

CREATE TABLE IF NOT EXISTS pattern_validation (
    run_id        BIGINT NOT NULL,
    ground_truth  VARCHAR,
    predicted     VARCHAR,
    n             BIGINT
);
"""


# Applied on every open. `IF NOT EXISTS` makes each one idempotent, so a store
# created by an earlier version gains new columns instead of needing a rebuild.
MIGRATIONS = (
    "ALTER TABLE series_profile ADD COLUMN IF NOT EXISTS total_volume DOUBLE",
    "ALTER TABLE series_profile ADD COLUMN IF NOT EXISTS volume_segment VARCHAR",
)


class EngineStore:
    """Thin, explicit persistence layer. All SQL lives here."""

    def __init__(self, client_id: str, root: Path | None = None) -> None:
        self.client_id = client_id
        base = root or project_root() / "data" / "processed"
        self.dir = base / client_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "engine.duckdb"

    @contextmanager
    def connect(self):
        con = duckdb.connect(str(self.path))
        try:
            yield con
        finally:
            con.close()

    def init_schema(self) -> None:
        with self.connect() as con:
            con.execute(DDL)
            for statement in MIGRATIONS:
                con.execute(statement)

    # ------------------------------------------------------------------ ingest

    def replace_source(self, dim: pl.DataFrame, fact: pl.DataFrame) -> None:
        """Idempotent load: re-ingesting the same file replaces its rows exactly.

        The source file is read from the incoming frames rather than passed in,
        so the rows deleted and the rows inserted can never refer to different
        files.
        """
        self.init_schema()
        source_files = fact["source_file"].unique().to_list()

        with self.connect() as con:
            con.register("incoming_dim", dim)
            con.register("incoming_fact", fact)
            con.execute("BEGIN TRANSACTION")
            for source_file in source_files:
                con.execute(
                    "DELETE FROM dim_series WHERE client_id = ? AND source_file = ?",
                    [self.client_id, source_file],
                )
                con.execute(
                    "DELETE FROM fact_demand WHERE client_id = ? AND source_file = ?",
                    [self.client_id, source_file],
                )
            con.execute("INSERT INTO dim_series BY NAME SELECT * FROM incoming_dim")
            con.execute("INSERT INTO fact_demand BY NAME SELECT * FROM incoming_fact")
            con.execute("COMMIT")

        self._export_parquet(dim, fact)

    def _export_parquet(self, dim: pl.DataFrame, fact: pl.DataFrame) -> None:
        canonical = self.dir / "canonical"
        canonical.mkdir(parents=True, exist_ok=True)
        dim.write_parquet(canonical / "dim_series.parquet")
        fact.write_parquet(canonical / "fact_demand.parquet")

    def load_dim(self) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM dim_series WHERE client_id = ? ORDER BY series_id",
                [self.client_id],
            ).pl()

    def load_fact(self) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute(
                "SELECT client_id, series_id, period, quantity FROM fact_demand "
                "WHERE client_id = ? ORDER BY series_id, period",
                [self.client_id],
            ).pl()

    def table_exists(self, name: str) -> bool:
        if not self.path.exists():
            return False
        with self.connect() as con:
            rows = con.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
            ).fetchall()
        return bool(rows)

    # --------------------------------------------------------------- exogenous

    def replace_exogenous(self, frame: pl.DataFrame) -> None:
        """Idempotent load of context features, replacing the file's rows."""
        self.init_schema()
        source_files = frame["source_file"].unique().to_list()
        with self.connect() as con:
            con.register("incoming_exog", frame)
            con.execute("BEGIN TRANSACTION")
            for source_file in source_files:
                con.execute(
                    "DELETE FROM fact_exogenous WHERE client_id = ? AND source_file = ?",
                    [self.client_id, source_file],
                )
            con.execute("INSERT INTO fact_exogenous BY NAME SELECT * FROM incoming_exog")
            con.execute("COMMIT")

    def load_exogenous(self, features: list[str]) -> pl.DataFrame | None:
        """Wide frame of the requested features, or None if none are stored."""
        if not features or not self.table_exists("fact_exogenous"):
            return None

        with self.connect() as con:
            long = con.execute(
                "SELECT series_id, period, feature, value FROM fact_exogenous "
                "WHERE client_id = ? AND feature = ANY(?)",
                [self.client_id, features],
            ).pl()

        if long.is_empty():
            return None

        wide = long.pivot(values="value", index=["series_id", "period"], on="feature")
        missing = [name for name in features if name not in wide.columns]
        if missing:
            raise ValueError(
                f"Exogenous features {missing} are configured but absent from the store. "
                "Run `engine exogenous` with a file that contains them."
            )
        return wide.select("series_id", "period", *features)

    # ---------------------------------------------------------------- profiles

    def write_series_profiles(self, profiles: pl.DataFrame) -> None:
        self.init_schema()
        frame = profiles.with_columns(
            pl.lit(self.client_id).alias("client_id"),
            pl.lit(datetime.now()).alias("computed_at"),
        ).select(
            "client_id",
            "series_id",
            "n_periods",
            "n_zeros",
            "zero_fraction",
            "mean_demand",
            "total_volume",
            "cv",
            "adi",
            "cv2",
            "trend_strength",
            "seasonal_strength",
            "residual_cv",
            "slope",
            "pattern",
            "volume_segment",
            "computed_at",
        )
        with self.connect() as con:
            con.register("incoming_profiles", frame)
            con.execute("DELETE FROM series_profile WHERE client_id = ?", [self.client_id])
            con.execute("INSERT INTO series_profile BY NAME SELECT * FROM incoming_profiles")

    def load_series_profiles(self) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM series_profile WHERE client_id = ? ORDER BY series_id",
                [self.client_id],
            ).pl()

    # -------------------------------------------------------------------- runs

    def start_run(self, run_type: str, config: dict[str, Any] | None = None) -> int:
        self.init_schema()
        with self.connect() as con:
            run_id = con.execute("SELECT nextval('run_id_seq')").fetchone()[0]
            con.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                [
                    run_id,
                    self.client_id,
                    run_type,
                    datetime.now(),
                    json.dumps(config or {}, default=str),
                ],
            )
        return int(run_id)

    def latest_run(self, run_type: str) -> int | None:
        if not self.table_exists("runs"):
            return None
        with self.connect() as con:
            row = con.execute(
                "SELECT run_id FROM runs WHERE client_id = ? AND run_type = ? "
                "ORDER BY run_id DESC LIMIT 1",
                [self.client_id, run_type],
            ).fetchone()
        return int(row[0]) if row else None

    def run_info(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).pl()
        return row.to_dicts()[0] if not row.is_empty() else None

    # --------------------------------------------------------------- backtest

    def write_backtest_predictions(self, run_id: int, predictions: pl.DataFrame) -> None:
        frame = predictions.with_columns(pl.lit(run_id, dtype=pl.Int64).alias("run_id")).select(
            "run_id",
            "model",
            "origin",
            "cutoff_period",
            "series_id",
            "period",
            "horizon",
            "actual",
            "forecast",
        )
        with self.connect() as con:
            con.register("incoming_preds", frame)
            con.execute("INSERT INTO backtest_predictions BY NAME SELECT * FROM incoming_preds")

    def write_backtest_quantiles(
        self, run_id: int, predictions: pl.DataFrame, levels: list[float]
    ) -> int:
        """Persist each origin's quantiles beside the actual they are scored on.

        `actual` is carried here rather than joined back later: a decision is
        judged against what happened in the period it was made for, and keeping
        them in one row means that pairing cannot drift.
        """
        present = [(level, quantile_column(level)) for level in levels]
        present = [(level, name) for level, name in present if name in predictions.columns]
        if not present:
            return 0

        frame = (
            predictions.select(
                "model",
                "origin",
                "cutoff_period",
                "series_id",
                "period",
                "horizon",
                "actual",
                *[pl.col(name).alias(str(level)) for level, name in present],
            )
            .unpivot(
                index=[
                    "model",
                    "origin",
                    "cutoff_period",
                    "series_id",
                    "period",
                    "horizon",
                    "actual",
                ],
                on=[str(level) for level, _ in present],
                variable_name="level",
                value_name="value",
            )
            .with_columns(
                pl.lit(run_id, dtype=pl.Int64).alias("run_id"),
                pl.col("level").cast(pl.Float64),
            )
            .drop_nulls("value")
        )

        with self.connect() as con:
            con.register("incoming_bq", frame)
            con.execute("INSERT INTO backtest_quantiles BY NAME SELECT * FROM incoming_bq")
        return frame.height

    def load_backtest_quantiles(self, run_id: int, level: float | None = None) -> pl.DataFrame:
        # A store written before the decision layer existed has no such table.
        # Empty is the honest answer, and the caller's message says to re-run.
        if not self.table_exists("backtest_quantiles"):
            return pl.DataFrame()
        sql = "SELECT * FROM backtest_quantiles WHERE run_id = ?"
        params: list[object] = [run_id]
        if level is not None:
            sql += " AND level = ?"
            params.append(level)
        with self.connect() as con:
            return con.execute(sql + " ORDER BY series_id, period, level", params).pl()

    def load_backtest_predictions(self, run_id: int) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute("SELECT * FROM backtest_predictions WHERE run_id = ?", [run_id]).pl()

    def write_metrics(self, run_id: int, metrics: pl.DataFrame) -> None:
        frame = metrics.with_columns(pl.lit(run_id, dtype=pl.Int64).alias("run_id")).select(
            "run_id", "model", "scope_type", "scope_value", "metric", "value", "n_obs"
        )
        with self.connect() as con:
            con.register("incoming_metrics", frame)
            con.execute("INSERT INTO backtest_metrics BY NAME SELECT * FROM incoming_metrics")

    def load_metrics(self, run_id: int) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute("SELECT * FROM backtest_metrics WHERE run_id = ?", [run_id]).pl()

    def write_model_selection(self, run_id: int, selection: pl.DataFrame) -> None:
        frame = selection.with_columns(pl.lit(run_id, dtype=pl.Int64).alias("run_id")).select(
            "run_id", "series_id", "model", "score", "metric"
        )
        with self.connect() as con:
            con.register("incoming_selection", frame)
            con.execute("INSERT INTO model_selection BY NAME SELECT * FROM incoming_selection")

    def load_model_selection(self, run_id: int) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute("SELECT * FROM model_selection WHERE run_id = ?", [run_id]).pl()

    def write_pattern_validation(self, run_id: int, matrix: pl.DataFrame) -> None:
        frame = matrix.with_columns(pl.lit(run_id, dtype=pl.Int64).alias("run_id")).select(
            "run_id", "ground_truth", "predicted", "n"
        )
        with self.connect() as con:
            con.register("incoming_pattern_val", frame)
            con.execute("INSERT INTO pattern_validation BY NAME SELECT * FROM incoming_pattern_val")

    def load_pattern_validation(self, run_id: int) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute("SELECT * FROM pattern_validation WHERE run_id = ?", [run_id]).pl()

    # -------------------------------------------------------------- forecasts

    def write_forecasts(self, run_id: int, forecasts: pl.DataFrame) -> None:
        frame = forecasts.with_columns(
            pl.lit(run_id, dtype=pl.Int64).alias("run_id"),
            pl.lit(self.client_id).alias("client_id"),
        ).select(
            "run_id",
            "client_id",
            "series_id",
            "cutoff_period",
            "period",
            "horizon",
            "model",
            "forecast",
        )
        with self.connect() as con:
            con.register("incoming_forecasts", frame)
            con.execute("INSERT INTO forecasts BY NAME SELECT * FROM incoming_forecasts")

    def write_forecast_quantiles(
        self, run_id: int, forecasts: pl.DataFrame, levels: list[float]
    ) -> int:
        """Persist the quantile columns of a forward forecast in long form."""
        present = [(level, quantile_column(level)) for level in levels]
        present = [(level, name) for level, name in present if name in forecasts.columns]
        if not present:
            return 0

        frame = (
            forecasts.select(
                "series_id",
                "period",
                "horizon",
                "model",
                *[pl.col(name).alias(str(level)) for level, name in present],
            )
            .unpivot(
                index=["series_id", "period", "horizon", "model"],
                on=[str(level) for level, _ in present],
                variable_name="level",
                value_name="value",
            )
            .with_columns(
                pl.lit(run_id, dtype=pl.Int64).alias("run_id"),
                pl.lit(self.client_id).alias("client_id"),
                pl.col("level").cast(pl.Float64),
            )
            .drop_nulls("value")
        )

        with self.connect() as con:
            con.register("incoming_fq", frame)
            con.execute("INSERT INTO forecast_quantiles BY NAME SELECT * FROM incoming_fq")
        return frame.height

    def load_forecast_quantiles(self, run_id: int) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM forecast_quantiles WHERE run_id = ? "
                "ORDER BY series_id, period, level",
                [run_id],
            ).pl()

    def load_forecasts(self, run_id: int) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM forecasts WHERE run_id = ? ORDER BY series_id, period", [run_id]
            ).pl()

    # --------------------------------------------------------------- decisions

    def write_decision_metrics(self, run_id: int, metrics: pl.DataFrame) -> None:
        frame = metrics.with_columns(pl.lit(run_id, dtype=pl.Int64).alias("run_id")).select(
            "run_id", "model", "scope_type", "scope_value", "metric", "value", "n_obs"
        )
        with self.connect() as con:
            con.register("incoming_dm", frame)
            con.execute("INSERT INTO decision_metrics BY NAME SELECT * FROM incoming_dm")

    def load_decision_metrics(self, run_id: int) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute("SELECT * FROM decision_metrics WHERE run_id = ?", [run_id]).pl()

    def write_decision_policy(self, run_id: int, policy: pl.DataFrame) -> None:
        frame = policy.with_columns(
            pl.lit(run_id, dtype=pl.Int64).alias("run_id"),
            pl.lit(self.client_id).alias("client_id"),
        ).select(
            "run_id",
            "client_id",
            "series_id",
            "model",
            "service_level",
            "window_periods",
            "expected_demand",
            "target_stock",
            "safety_stock",
            "reorder_point",
        )
        with self.connect() as con:
            con.register("incoming_dp", frame)
            con.execute("INSERT INTO decision_policy BY NAME SELECT * FROM incoming_dp")

    def load_decision_policy(self, run_id: int) -> pl.DataFrame:
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM decision_policy WHERE run_id = ? ORDER BY series_id", [run_id]
            ).pl()
