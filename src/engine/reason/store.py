"""Additive DuckDB tables for the reasoning layer, in the client's existing store.

The forecast tables are never written here. A reasoning run is one JSON
document (`reason_runs`), plus flat rows the ERP portal can query directly:
the per-SKU bands and the model-call ledger (tokens and cost).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from engine.reason.schemas import LLMCall, Override, ReasonRun, Signal
from engine.store.duckdb_store import EngineStore

DDL = """
CREATE TABLE IF NOT EXISTS reason_runs (
    client_id VARCHAR NOT NULL, reason_id VARCHAR NOT NULL, forecast_run_id BIGINT NOT NULL,
    inputs_hash VARCHAR NOT NULL, status VARCHAR NOT NULL, created_at TIMESTAMPTZ NOT NULL,
    document VARCHAR NOT NULL, PRIMARY KEY (client_id, reason_id));
CREATE TABLE IF NOT EXISTS reason_bands (
    client_id VARCHAR NOT NULL, reason_id VARCHAR NOT NULL, series_id VARCHAR NOT NULL,
    family_id VARCHAR NOT NULL, period DATE NOT NULL, baseline DOUBLE NOT NULL,
    low DOUBLE NOT NULL, high DOUBLE NOT NULL, findings VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS reason_llm_calls (
    client_id VARCHAR NOT NULL, reason_id VARCHAR, call_id VARCHAR NOT NULL, stage VARCHAR NOT NULL,
    provider VARCHAR NOT NULL, model VARCHAR NOT NULL, started_at TIMESTAMPTZ NOT NULL,
    seconds DOUBLE NOT NULL, input_tokens BIGINT NOT NULL, output_tokens BIGINT NOT NULL,
    cached_input_tokens BIGINT NOT NULL, reasoning_tokens BIGINT NOT NULL, cost DOUBLE,
    currency VARCHAR NOT NULL, outcome VARCHAR NOT NULL, request_hash VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS reason_overrides (
    client_id VARCHAR NOT NULL, reason_id VARCHAR NOT NULL, finding_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL, low_pct DOUBLE, high_pct DOUBLE, actor VARCHAR NOT NULL,
    note VARCHAR NOT NULL, decided_at TIMESTAMPTZ NOT NULL);
CREATE TABLE IF NOT EXISTS reason_signal_cache (
    client_id VARCHAR NOT NULL, cache_key VARCHAR NOT NULL, fetched_at TIMESTAMPTZ NOT NULL,
    document VARCHAR NOT NULL, PRIMARY KEY (client_id, cache_key));
CREATE TABLE IF NOT EXISTS reason_profile_versions (
    client_id VARCHAR NOT NULL, profile_hash VARCHAR NOT NULL, saved_at TIMESTAMPTZ NOT NULL,
    actor VARCHAR NOT NULL, document VARCHAR NOT NULL);
"""


@contextmanager
def transaction(store: EngineStore) -> Iterator:
    with store.connect() as con:
        con.execute(DDL)
        con.execute("BEGIN TRANSACTION")
        try:
            yield con
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise


@contextmanager
def reader(store: EngineStore) -> Iterator:
    with store.connect() as con:
        con.execute(DDL)
        yield con


def save_run(store: EngineStore, run: ReasonRun) -> None:
    with transaction(store) as con:
        key = [store.client_id, run.reason_id]
        con.execute("DELETE FROM reason_runs WHERE client_id = ? AND reason_id = ?", key)
        con.execute("DELETE FROM reason_bands WHERE client_id = ? AND reason_id = ?", key)
        con.execute(
            "INSERT INTO reason_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                *key,
                run.forecast_run_id,
                run.inputs_hash,
                run.status,
                run.created_at,
                run.model_dump_json(),
            ],
        )
        if run.bands:
            con.executemany(
                "INSERT INTO reason_bands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    [
                        *key,
                        b.series_id,
                        b.family_id,
                        b.period,
                        b.baseline,
                        b.low,
                        b.high,
                        ",".join(b.finding_ids),
                    ]
                    for b in run.bands
                ],
            )


def _one(store: EngineStore, where: str, params: list) -> ReasonRun | None:
    if not store.path.exists():
        return None
    with reader(store) as con:
        row = con.execute(
            f"SELECT document FROM reason_runs WHERE client_id = ? AND {where} "
            "ORDER BY created_at DESC LIMIT 1",
            [store.client_id, *params],
        ).fetchone()
    return ReasonRun.model_validate_json(row[0]) if row else None


def load_run(store: EngineStore, reason_id: str) -> ReasonRun | None:
    return _one(store, "reason_id = ?", [reason_id])


def run_by_inputs(store: EngineStore, inputs_hash: str) -> ReasonRun | None:
    return _one(store, "inputs_hash = ? AND status = 'complete'", [inputs_hash])


def latest_run(store: EngineStore, exclude: str | None = None) -> ReasonRun | None:
    return _one(store, "status <> 'dry_run' AND reason_id <> ?", [exclude or ""])


def list_runs(store: EngineStore) -> list[tuple]:
    if not store.path.exists():
        return []
    with reader(store) as con:
        return con.execute(
            "SELECT reason_id, forecast_run_id, status, created_at FROM reason_runs "
            "WHERE client_id = ? ORDER BY created_at DESC",
            [store.client_id],
        ).fetchall()


def save_call(store: EngineStore, reason_id: str | None, call: LLMCall) -> None:
    u = call.usage
    with transaction(store) as con:
        con.execute(
            "INSERT INTO reason_llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                store.client_id,
                reason_id,
                call.call_id,
                call.stage,
                call.provider,
                call.model,
                call.started_at,
                call.seconds,
                u.input_tokens,
                u.output_tokens,
                u.cached_input_tokens,
                u.reasoning_tokens,
                call.cost,
                call.currency,
                call.outcome,
                call.request_hash,
            ],
        )


def call_ledger(store: EngineStore) -> list[dict]:
    if not store.path.exists():
        return []
    with reader(store) as con:
        cursor = con.execute(
            "SELECT reason_id, stage, provider, model, started_at, input_tokens, output_tokens,"
            " cached_input_tokens, cost, currency, outcome FROM reason_llm_calls"
            " WHERE client_id = ? ORDER BY started_at",
            [store.client_id],
        )
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def cache_get(store: EngineStore, key: str, max_age_hours: int) -> list[Signal] | None:
    if max_age_hours <= 0 or not store.path.exists():
        return None
    with reader(store) as con:
        row = con.execute(
            "SELECT document FROM reason_signal_cache WHERE client_id = ? AND cache_key = ?"
            " AND fetched_at >= ?",
            [store.client_id, key, datetime.now(UTC) - timedelta(hours=max_age_hours)],
        ).fetchone()
    if row is None:
        return None
    return [Signal.model_validate(item) for item in json.loads(row[0])]


def cache_put(store: EngineStore, key: str, signals: list[Signal]) -> None:
    document = json.dumps([item.model_dump(mode="json") for item in signals])
    with transaction(store) as con:
        con.execute(
            "INSERT OR REPLACE INTO reason_signal_cache VALUES (?, ?, ?, ?)",
            [store.client_id, key, datetime.now(UTC), document],
        )


def save_profile_version(store: EngineStore, profile_hash: str, actor: str, text: str) -> None:
    with transaction(store) as con:
        con.execute(
            "INSERT INTO reason_profile_versions VALUES (?, ?, ?, ?, ?)",
            [store.client_id, profile_hash, datetime.now(UTC), actor, text],
        )


def save_override(store: EngineStore, override: Override) -> None:
    with transaction(store) as con:
        con.execute(
            "INSERT INTO reason_overrides VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                store.client_id,
                override.reason_id,
                override.finding_id,
                override.action,
                override.low_pct,
                override.high_pct,
                override.actor,
                override.note,
                override.at,
            ],
        )


def overrides(store: EngineStore, reason_id: str) -> list[Override]:
    """Latest decision per finding; earlier ones remain in the table as history."""
    if not store.path.exists():
        return []
    with reader(store) as con:
        rows = con.execute(
            "SELECT reason_id, finding_id, action, low_pct, high_pct, actor, note, decided_at"
            " FROM reason_overrides WHERE client_id = ? AND reason_id = ? ORDER BY decided_at",
            [store.client_id, reason_id],
        ).fetchall()
    latest = {}
    for row in rows:
        item = Override(
            reason_id=row[0],
            finding_id=row[1],
            action=row[2],
            low_pct=row[3],
            high_pct=row[4],
            actor=row[5],
            note=row[6],
            at=row[7],
        )
        latest[item.finding_id] = item
    return list(latest.values())
