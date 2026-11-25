"""Additive context persistence in the engine's per-client DuckDB."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from threading import Lock

from duckdb import DuckDBPyConnection

from engine.reason.schemas import (
    Assessment,
    CompanyFact,
    ContextError,
    ContextSyncError,
    EvidenceBatch,
    EvidenceRevision,
    ForecastRow,
    ForecastSnapshot,
    LookupAttempt,
    ModelCall,
    Question,
    QuestionAnswer,
    ReviewManifest,
    SourceCheck,
)
from engine.store.duckdb_store import EngineStore

REASON_DDL = """
CREATE TABLE IF NOT EXISTS reason_forecast_snapshots (
    client_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    forecast_run_id BIGINT NOT NULL,
    input_digest VARCHAR NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    header JSON NOT NULL,
    PRIMARY KEY (client_id, snapshot_id),
    UNIQUE (client_id, forecast_run_id, input_digest)
);
CREATE TABLE IF NOT EXISTS reason_forecast_rows (
    client_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    series_id VARCHAR NOT NULL,
    period DATE NOT NULL,
    row_data JSON NOT NULL,
    PRIMARY KEY (client_id, snapshot_id, series_id, period)
);
CREATE TABLE IF NOT EXISTS reason_company_facts (
    client_id VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL,
    version_key VARCHAR NOT NULL,
    fact_version_id VARCHAR NOT NULL,
    known_at TIMESTAMPTZ NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    effective_from TIMESTAMPTZ NOT NULL,
    effective_to TIMESTAMPTZ,
    status VARCHAR NOT NULL,
    fact_data JSON NOT NULL,
    PRIMARY KEY (client_id, source_id, record_id, version_key),
    UNIQUE (client_id, fact_version_id)
);
CREATE TABLE IF NOT EXISTS reason_questions (
    client_id VARCHAR NOT NULL,
    question_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    question_data JSON NOT NULL,
    PRIMARY KEY (client_id, question_id)
);
CREATE TABLE IF NOT EXISTS reason_question_answers (
    client_id VARCHAR NOT NULL,
    question_id VARCHAR NOT NULL,
    revision INTEGER NOT NULL,
    answer_id VARCHAR NOT NULL,
    answer_data JSON NOT NULL,
    PRIMARY KEY (client_id, question_id, revision),
    UNIQUE (client_id, answer_id)
);
CREATE TABLE IF NOT EXISTS reason_evidence_owners (
    client_id VARCHAR NOT NULL,
    owner_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    binding JSON NOT NULL,
    PRIMARY KEY (client_id, owner_kind, owner_id)
);
CREATE TABLE IF NOT EXISTS reason_evidence_revisions (
    client_id VARCHAR NOT NULL,
    evidence_id VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL,
    revision INTEGER NOT NULL,
    first_known_at TIMESTAMPTZ NOT NULL,
    evidence_data JSON NOT NULL,
    PRIMARY KEY (client_id, evidence_id),
    UNIQUE (client_id, record_id, revision)
);
CREATE TABLE IF NOT EXISTS reason_evidence_checks (
    client_id VARCHAR NOT NULL,
    check_id VARCHAR NOT NULL,
    owner_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    query_hash VARCHAR NOT NULL,
    cache_key VARCHAR NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    cacheable BOOLEAN NOT NULL,
    check_data JSON NOT NULL,
    PRIMARY KEY (client_id, check_id)
);
CREATE TABLE IF NOT EXISTS reason_evidence_attempts (
    client_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    owner_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    query_hash VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    retry BOOLEAN NOT NULL,
    check_id VARCHAR,
    PRIMARY KEY (client_id, attempt_id)
);
CREATE TABLE IF NOT EXISTS reason_evidence_batches (
    client_id VARCHAR NOT NULL,
    batch_id VARCHAR NOT NULL,
    collection_id VARCHAR NOT NULL,
    batch_data JSON NOT NULL,
    PRIMARY KEY (client_id, batch_id),
    UNIQUE (client_id, collection_id)
);
"""

# DuckDB permits one writing process per file. Serialize advisory write transactions
# inside that process too, so repeated imports and confirmations see earlier commits.
_write_lock = Lock()


@contextmanager
def write_transaction(store: EngineStore) -> Iterator[DuckDBPyConnection]:
    with _write_lock, store.connect() as con:
        con.execute("BEGIN TRANSACTION")
        try:
            yield con
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise


def init_reason_schema(store: EngineStore) -> None:
    with write_transaction(store) as con:
        con.execute(REASON_DDL)


def table_exists(con: DuckDBPyConnection, name: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [name],
        ).fetchone()
        is not None
    )


def put_evidence(
    con: DuckDBPyConnection,
    client: str,
    revisions: tuple[EvidenceRevision, ...],
) -> tuple[EvidenceRevision, ...]:
    saved = {
        item.evidence_id: item
        for item in read_evidence(con, client, tuple(item.evidence_id for item in revisions))
    }
    new = [item for item in revisions if item.evidence_id not in saved]
    if new:
        counters = dict(
            con.execute(
                "SELECT record_id, max(revision) FROM reason_evidence_revisions "
                "WHERE client_id = ? AND record_id IN (SELECT unnest(?::VARCHAR[])) "
                "GROUP BY record_id",
                [client, list({item.record_id for item in new})],
            ).fetchall()
        )
        values = []
        for item in new:
            if item.evidence_id in saved:
                continue
            number = counters.get(item.record_id, 0) + 1
            counters[item.record_id] = number
            item = item.model_copy(update={"revision": number})
            saved[item.evidence_id] = item
            values.append(
                (
                    client,
                    item.evidence_id,
                    item.record_id,
                    number,
                    item.first_known_at,
                    item.model_dump_json(),
                )
            )
        con.executemany("INSERT INTO reason_evidence_revisions VALUES (?, ?, ?, ?, ?, ?)", values)
    return tuple(saved.values())


def read_evidence(
    con: DuckDBPyConnection, client_id: str, evidence_ids: tuple[str, ...]
) -> tuple[EvidenceRevision, ...]:
    if not evidence_ids:
        return ()
    values = con.execute(
        "SELECT evidence_data FROM reason_evidence_revisions "
        "WHERE client_id = ? AND evidence_id IN (SELECT unnest(?::VARCHAR[])) "
        "ORDER BY evidence_id",
        [client_id, list(evidence_ids)],
    ).fetchall()
    return tuple(EvidenceRevision.model_validate_json(row[0]) for row in values)


def save_check(
    con: DuckDBPyConnection,
    owner: tuple[str, str, str],
    cache_key: str,
    check: SourceCheck,
    *,
    cacheable: bool = False,
) -> None:
    client, kind, identifier = owner
    con.execute(
        "INSERT INTO reason_evidence_checks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            client,
            check.check_id,
            kind,
            identifier,
            check.scope.query_hash,
            cache_key,
            check.ended_at,
            cacheable,
            check.model_dump_json(),
        ],
    )


def read_checks(
    con: DuckDBPyConnection,
    owner: tuple[str, str, str],
    scope_hash: str | None = None,
) -> tuple[SourceCheck, ...]:
    clause = " AND query_hash = ?" if scope_hash is not None else ""
    rows = con.execute(
        "SELECT check_data FROM reason_evidence_checks "
        "WHERE client_id = ? AND owner_kind = ? AND owner_id = ?"
        + clause
        + " ORDER BY ended_at, check_id",
        [*owner, scope_hash] if scope_hash is not None else list(owner),
    ).fetchall()
    return tuple(SourceCheck.model_validate_json(row[0]) for row in rows)


def cached_check(
    con: DuckDBPyConnection,
    client: str,
    cache_key: str,
    cutoff: datetime,
    earliest: datetime | None,
    *,
    allow_partial: bool,
    owner: tuple[str, str] | None = None,
) -> SourceCheck | None:
    clause = " AND ended_at >= ?" if earliest is not None else ""
    params = [client, cache_key, cutoff]
    if earliest is not None:
        params.append(earliest)
    if owner is not None:
        clause += " AND owner_kind = ? AND owner_id = ?"
        params.extend(owner)
    rows = con.execute(
        "SELECT check_data FROM reason_evidence_checks "
        "WHERE client_id = ? AND cache_key = ? AND cacheable AND ended_at <= ?"
        + clause
        + " ORDER BY ended_at DESC, check_id",
        params,
    ).fetchall()
    for row in rows:
        check = SourceCheck.model_validate_json(row[0])
        if allow_partial or check.coverage == "complete":
            return check
    return None


def read_batch(
    con: DuckDBPyConnection,
    client: str,
    identifier: str,
    *,
    by_collection: bool = False,
) -> EvidenceBatch | None:
    column = "collection_id" if by_collection else "batch_id"
    row = con.execute(
        f"SELECT batch_data FROM reason_evidence_batches WHERE client_id = ? AND {column} = ?",
        [client, identifier],
    ).fetchone()
    return EvidenceBatch.model_validate_json(row[0]) if row else None


def save_batch(con: DuckDBPyConnection, batch: EvidenceBatch) -> None:
    con.execute(
        "INSERT INTO reason_evidence_batches VALUES (?, ?, ?, ?)",
        [batch.client_id, batch.batch_id, batch.collection_id, batch.model_dump_json()],
    )


def read_saved_run(
    con: DuckDBPyConnection, client_id: str, run_id: int
) -> tuple[dict, list[dict], dict[str, dict]]:
    if not table_exists(con, "runs"):
        raise ContextError(
            f"Forecast run {run_id} was not found for {client_id!r}. Run engine forecast."
        )
    header = con.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).pl().to_dicts()
    if not header or header[0]["client_id"] != client_id:
        raise ContextError(f"Forecast run {run_id} was not found for {client_id!r}")
    if header[0]["run_type"] != "forecast":
        raise ContextError(
            f"Run {run_id} is not a forecast run. Use a saved engine forecast run ID."
        )
    rows = (
        con.execute("SELECT * FROM forecasts WHERE run_id = ? ORDER BY series_id, period", [run_id])
        .pl()
        .to_dicts()
    )
    if any(row["client_id"] != client_id for row in rows):
        raise ContextError(f"Forecast run {run_id} contains rows for another client")
    dimensions = (
        con.execute(
            """SELECT * FROM dim_series WHERE client_id = ? AND series_id IN (
            SELECT series_id FROM forecasts WHERE run_id = ? AND client_id = ?
        ) ORDER BY series_id""",
            [client_id, run_id, client_id],
        )
        .pl()
        .to_dicts()
    )
    indexed = {row["series_id"]: row for row in dimensions}
    if len(indexed) != len(dimensions):
        raise ContextError("Series metadata is ambiguous. Resolve duplicate dim_series rows first.")
    saved = header[0]
    if table_exists(con, "forecast_availability"):
        available = con.execute(
            "SELECT available_at FROM forecast_availability WHERE client_id = ? AND run_id = ?",
            [client_id, run_id],
        ).fetchone()
        if available:
            saved["forecast_available_at"] = available[0]
    return saved, rows, indexed


def read_snapshot(
    con: DuckDBPyConnection, client_id: str, snapshot_id: str
) -> ForecastSnapshot | None:
    result = con.execute(
        "SELECT header FROM reason_forecast_snapshots WHERE client_id = ? AND snapshot_id = ?",
        [client_id, snapshot_id],
    ).fetchone()
    if result is None:
        return None
    rows = con.execute(
        """SELECT row_data FROM reason_forecast_rows WHERE client_id = ? AND snapshot_id = ?
        ORDER BY series_id, period""",
        [client_id, snapshot_id],
    ).fetchall()
    header = json.loads(result[0])
    return ForecastSnapshot(
        **header, rows=tuple(ForecastRow.model_validate_json(row[0]) for row in rows)
    )


def save_snapshot(con: DuckDBPyConnection, snapshot: ForecastSnapshot) -> None:
    con.execute(
        "INSERT INTO reason_forecast_snapshots VALUES (?, ?, ?, ?, ?, ?)",
        [
            snapshot.client_id,
            snapshot.snapshot_id,
            snapshot.forecast_run_id,
            snapshot.input_digest,
            snapshot.captured_at,
            snapshot.model_dump_json(exclude={"rows"}),
        ],
    )
    con.executemany(
        "INSERT INTO reason_forecast_rows VALUES (?, ?, ?, ?, ?)",
        [
            (row.client_id, snapshot.snapshot_id, row.series_id, row.period, row.model_dump_json())
            for row in snapshot.rows
        ],
    )


def save_fact(con: DuckDBPyConnection, fact: CompanyFact) -> bool:
    key = f"version:{fact.source_version}" if fact.source_version else f"hash:{fact.content_digest}"
    identity = [fact.client_id, fact.source_id, fact.record_id, key]
    existing = con.execute(
        """SELECT fact_data FROM reason_company_facts
        WHERE client_id = ? AND source_id = ? AND record_id = ? AND version_key = ?""",
        identity,
    ).fetchone()
    if existing:
        if CompanyFact.model_validate_json(existing[0]).content_digest != fact.content_digest:
            raise ContextSyncError(
                f"Context source {fact.source_id!r}, record {fact.record_id!r} reuses version "
                f"{fact.source_version!r} with different data. Provide a new source version.",
                invalid=1,
            )
        return False
    # Multiple revisions first seen in the same batch need unambiguous knowledge ordering.
    competing = con.execute(
        """SELECT 1 FROM reason_company_facts WHERE client_id = ? AND source_id = ?
        AND record_id = ? AND known_at = ? AND captured_at = ?""",
        [fact.client_id, fact.source_id, fact.record_id, fact.known_at, fact.captured_at],
    ).fetchone()
    if competing:
        raise ContextSyncError(
            f"Context source {fact.source_id!r}, record {fact.record_id!r} has multiple revisions "
            "with the same availability in one import. Supply distinct known_at values.",
            invalid=1,
        )
    con.execute(
        "INSERT INTO reason_company_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            *identity,
            fact.fact_version_id,
            fact.known_at,
            fact.captured_at,
            fact.effective_from,
            fact.effective_to,
            fact.status,
            fact.model_dump_json(),
        ],
    )
    return True


def eligible_facts(
    con: DuckDBPyConnection,
    client_id: str,
    known_at: datetime,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> tuple[CompanyFact, ...]:
    rows = con.execute(
        """WITH latest AS (
            SELECT * FROM reason_company_facts WHERE client_id = ? AND known_at <= ?
            QUALIFY row_number() OVER (
                PARTITION BY source_id, record_id
                ORDER BY known_at DESC, captured_at DESC, fact_version_id DESC
            ) = 1
        ) SELECT fact_data FROM latest WHERE status <> 'withdrawn'
          AND (?::TIMESTAMPTZ IS NULL OR effective_to IS NULL OR effective_to > ?)
          AND (?::TIMESTAMPTZ IS NULL OR effective_from < ?)
          ORDER BY source_id, record_id""",
        [client_id, known_at, valid_from, valid_from, valid_to, valid_to],
    ).fetchall()
    return tuple(CompanyFact.model_validate_json(row[0]) for row in rows)


def get_question(con: DuckDBPyConnection, client_id: str, question_id: str) -> Question | None:
    row = con.execute(
        "SELECT question_data FROM reason_questions WHERE client_id = ? AND question_id = ?",
        [client_id, question_id],
    ).fetchone()
    return Question.model_validate_json(row[0]) if row else None


def read_questions(
    con: DuckDBPyConnection, client_id: str, pending_only: bool
) -> tuple[Question, ...]:
    rows = con.execute(
        """SELECT question_data FROM reason_questions
        WHERE client_id = ? AND (NOT ? OR status = 'pending') ORDER BY question_id""",
        [client_id, pending_only],
    ).fetchall()
    return tuple(Question.model_validate_json(row[0]) for row in rows)


def save_question(con: DuckDBPyConnection, question: Question) -> None:
    con.execute(
        """INSERT INTO reason_questions VALUES (?, ?, ?, ?)
        ON CONFLICT (client_id, question_id) DO UPDATE
        SET status = excluded.status, question_data = excluded.question_data""",
        [question.client_id, question.question_id, question.status, question.model_dump_json()],
    )


def save_answer(con: DuckDBPyConnection, client_id: str, answer: QuestionAnswer) -> None:
    con.execute(
        "INSERT INTO reason_question_answers VALUES (?, ?, ?, ?, ?)",
        [
            client_id,
            answer.question_id,
            answer.revision,
            answer.answer_id,
            answer.model_dump_json(),
        ],
    )


def get_answer(con: DuckDBPyConnection, client_id: str, answer_id: str) -> QuestionAnswer | None:
    row = con.execute(
        "SELECT answer_data FROM reason_question_answers WHERE client_id = ? AND answer_id = ?",
        [client_id, answer_id],
    ).fetchone()
    return QuestionAnswer.model_validate_json(row[0]) if row else None


def get_fact(con: DuckDBPyConnection, client_id: str, fact_version_id: str) -> CompanyFact | None:
    row = con.execute(
        "SELECT fact_data FROM reason_company_facts WHERE client_id = ? AND fact_version_id = ?",
        [client_id, fact_version_id],
    ).fetchone()
    return CompanyFact.model_validate_json(row[0]) if row else None


REVIEW_DDL = """
CREATE TABLE IF NOT EXISTS reason_review_heads (
    client_id VARCHAR NOT NULL,
    review_id VARCHAR NOT NULL,
    revision INTEGER NOT NULL,
    lease_id VARCHAR,
    lease_until TIMESTAMPTZ,
    PRIMARY KEY (client_id, review_id)
);
CREATE TABLE IF NOT EXISTS reason_review_manifests (
    client_id VARCHAR NOT NULL,
    review_id VARCHAR NOT NULL,
    revision INTEGER NOT NULL,
    manifest_data JSON NOT NULL,
    PRIMARY KEY (client_id, review_id, revision)
);
CREATE TABLE IF NOT EXISTS reason_review_work (
    client_id VARCHAR NOT NULL,
    review_id VARCHAR NOT NULL,
    revision INTEGER NOT NULL,
    work_id VARCHAR NOT NULL,
    input_digest VARCHAR NOT NULL,
    work_data JSON NOT NULL,
    PRIMARY KEY (client_id, review_id, revision, work_id)
);
CREATE TABLE IF NOT EXISTS reason_review_assessments (
    client_id VARCHAR NOT NULL,
    review_id VARCHAR NOT NULL,
    scenario_id VARCHAR NOT NULL,
    series_id VARCHAR NOT NULL,
    period DATE NOT NULL,
    target VARCHAR NOT NULL,
    assessment_data JSON NOT NULL,
    PRIMARY KEY (client_id, review_id, scenario_id, series_id, period, target)
);
CREATE TABLE IF NOT EXISTS reason_review_model_calls (
    client_id VARCHAR NOT NULL,
    budget_owner_id VARCHAR NOT NULL,
    call_id VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    call_data JSON NOT NULL,
    PRIMARY KEY (client_id, budget_owner_id, call_id)
);
"""


def read_review(con, client_id: str, review_id: str, revision: int | None = None):
    if not table_exists(con, "reason_review_manifests"):
        return None
    sql = """SELECT manifest_data FROM reason_review_manifests
        WHERE client_id = ? AND review_id = ?"""
    params = [client_id, review_id]
    if revision is not None:
        sql += " AND revision = ?"
        params.append(revision)
    row = con.execute(sql + " ORDER BY revision DESC LIMIT 1", params).fetchone()
    return ReviewManifest.model_validate_json(row[0]) if row else None


def read_check_ids(con, client_id: str, check_ids: tuple[str, ...]) -> tuple[SourceCheck, ...]:
    if not check_ids:
        return ()
    rows = con.execute(
        """SELECT check_data FROM reason_evidence_checks WHERE client_id = ?
        AND check_id IN (SELECT unnest(?::VARCHAR[])) ORDER BY ended_at, check_id""",
        [client_id, list(check_ids)],
    ).fetchall()
    return tuple(SourceCheck.model_validate_json(row[0]) for row in rows)


def insert_review(con, manifest: ReviewManifest) -> None:
    con.execute(
        "INSERT INTO reason_review_heads VALUES (?, ?, ?, NULL, NULL)",
        [manifest.client_id, manifest.review_id, manifest.revision],
    )
    save_review(con, manifest)


def save_review(con, manifest: ReviewManifest) -> None:
    key = [manifest.client_id, manifest.review_id, manifest.revision]
    con.execute(
        "INSERT INTO reason_review_manifests VALUES (?, ?, ?, ?)",
        [*key, manifest.model_dump_json()],
    )
    con.execute(
        "UPDATE reason_review_heads SET revision = ? WHERE client_id = ? AND review_id = ?",
        [manifest.revision, manifest.client_id, manifest.review_id],
    )


def save_review_work(con, manifest: ReviewManifest, work_id: str, digest: str, data: str) -> None:
    con.execute(
        "INSERT OR REPLACE INTO reason_review_work VALUES (?, ?, ?, ?, ?, ?)",
        [manifest.client_id, manifest.review_id, manifest.revision, work_id, digest, data],
    )


def read_review_work(con, manifest: ReviewManifest, work_id: str, digest: str) -> str | None:
    row = con.execute(
        """SELECT work_data FROM reason_review_work
        WHERE client_id = ? AND review_id = ? AND work_id = ? AND input_digest = ?
        ORDER BY revision DESC LIMIT 1""",
        [manifest.client_id, manifest.review_id, work_id, digest],
    ).fetchone()
    return row[0] if row else None


def save_assessments(con, client_id: str, assessments: tuple[Assessment, ...]) -> None:
    if assessments:
        con.executemany(
            "INSERT INTO reason_review_assessments VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                [
                    client_id,
                    row.review_id,
                    row.scenario_id,
                    row.series_id,
                    row.period,
                    row.target,
                    row.model_dump_json(),
                ]
                for row in assessments
            ],
        )


def read_assessments(con, client_id: str, review_id: str) -> tuple[Assessment, ...]:
    rows = con.execute(
        """SELECT assessment_data FROM reason_review_assessments
        WHERE client_id = ? AND review_id = ? ORDER BY series_id, period, target, scenario_id""",
        [client_id, review_id],
    ).fetchall()
    return tuple(Assessment.model_validate_json(row[0]) for row in rows)


def read_lookup_attempts(con, client_id: str, owner_id: str) -> tuple[LookupAttempt, ...]:
    rows = con.execute(
        """SELECT attempt_id, query_hash, provider, started_at, retry, check_id
        FROM reason_evidence_attempts WHERE client_id = ? AND owner_kind = 'review'
        AND owner_id = ? ORDER BY started_at, attempt_id""",
        [client_id, owner_id],
    ).fetchall()
    return tuple(
        LookupAttempt(
            attempt_id=identifier,
            query_hash=query_hash,
            provider=provider,
            started_at=at,
            retry=retry,
            check_id=check_id,
        )
        for identifier, query_hash, provider, at, retry, check_id in rows
    )


def read_model_calls(con, client_id: str, owner_id: str) -> tuple[ModelCall, ...]:
    rows = con.execute(
        """SELECT call_data FROM reason_review_model_calls
        WHERE client_id = ? AND budget_owner_id = ? ORDER BY started_at, call_id""",
        [client_id, owner_id],
    ).fetchall()
    return tuple(ModelCall.model_validate_json(row[0]) for row in rows)


def save_model_call(con, client_id: str, owner_id: str, call: ModelCall) -> None:
    con.execute(
        "INSERT OR REPLACE INTO reason_review_model_calls VALUES (?, ?, ?, ?, ?)",
        [client_id, owner_id, call.call_id, call.started_at, call.model_dump_json()],
    )
