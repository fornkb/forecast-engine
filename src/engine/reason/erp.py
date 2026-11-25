"""Read deployment-configured ERP context without write access."""

from __future__ import annotations

import csv
import os
from collections.abc import Callable
from typing import Any

from engine.config.schema import ClientProfile
from engine.ingest.postgres_reader import scrub
from engine.reason.config import ContextSource, OperationalSource, ReasoningConfig
from engine.reason.schemas import ContextError, ContextSyncError

SourceRow = dict[str, Any]
PostgresFetch = Callable[[str, str, tuple[str, ...], tuple[str, str]], list[SourceRow]]


class OperationalReadError(ContextError):
    def __init__(self, message: str, *, transient: bool = False, status: str | None = None):
        super().__init__(message)
        self.transient = transient
        self.status = status


def require_reasoning(profile: ClientProfile) -> ReasoningConfig:
    if profile.reasoning is None or not profile.reasoning.enabled:
        raise ContextError(
            f"Reasoning is disabled for {profile.client_id!r}. "
            "Set reasoning.enabled: true in the client profile and configure its context."
        )
    return profile.reasoning


def _source_fields(source: ContextSource | OperationalSource) -> dict[str, str]:
    return (
        source.fields
        if isinstance(source, OperationalSource)
        else source.fields.model_dump(exclude_none=True)
    )


def _csv_rows(profile: ClientProfile, source: ContextSource | OperationalSource) -> list[SourceRow]:
    path = source.path
    assert path is not None
    if not path.is_absolute():
        if profile.profile_dir is None:
            raise ContextError("Load the profile from disk to resolve the relative CSV path")
        path = profile.profile_dir / path
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers):
            raise ContextError(f"CSV {path} has duplicate column names")
        missing = set(_source_fields(source).values()) - set(headers)
        if missing:
            raise ContextError(f"CSV {path} is missing mapped columns {sorted(missing)}")
        rows = []
        for row in reader:
            if None in row or None in row.values():
                raise ContextSyncError(
                    f"CSV {path}, line {reader.line_num}: wrong number of fields", invalid=1
                )
            rows.append(row)
            if isinstance(source, OperationalSource) and len(rows) > source.max_rows:
                break
        return rows


def fetch_postgres_context(
    query: str, dsn: str, parameters: tuple[str, ...], relation: tuple[str, str]
) -> list[SourceRow]:
    try:
        from adbc_driver_postgresql import dbapi
    except ImportError as exc:
        raise ContextError(
            'Install the PostgreSQL extra with pip install -e ".[postgres]"'
        ) from exc

    if dsn.startswith(("postgresql://", "postgres://")):
        separator = "&" if "?" in dsn else "?"
        connection_dsn = f"{dsn}{separator}connect_timeout=15"
    else:
        connection_dsn = f"{dsn} connect_timeout=15"
    with dbapi.connect(connection_dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("SET LOCAL statement_timeout = '15s'")
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_views WHERE schemaname = $1 AND viewname = $2",
            relation,
        )
        if cursor.fetchone() is None:
            raise ContextError(
                f"Configured relation {'.'.join(relation)} must be a PostgreSQL view"
            )
        cursor.execute(query, parameters or None)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _postgres_rows(
    profile: ClientProfile, source: ContextSource | OperationalSource, fetch: PostgresFetch
) -> list[SourceRow]:
    assert source.dsn_env is not None and source.relation is not None
    dsn = os.environ.get(source.dsn_env)
    if not dsn:
        raise ContextError(
            f"Set environment variable {source.dsn_env} to the read-only context DSN"
        )

    schema, view = source.relation.split(".")
    fields = _source_fields(source)
    columns = ", ".join(f'"{name}"' for name in fields.values())
    query = f'SELECT {columns} FROM "{schema}"."{view}"'
    parameters: tuple[str, ...] = ()
    if "client_id" in fields:
        query += f' WHERE "{fields["client_id"]}" = $1'
        parameters = (profile.client_id,)
    if isinstance(source, OperationalSource):
        order = ", ".join(
            f'"{fields[key]}" DESC NULLS LAST'
            for key in ("known_at", "record_id", "effective_from", "version")
            if key in fields
        )
        query += f" ORDER BY {order} LIMIT {source.max_rows + 1}"
    try:
        return fetch(query, dsn, parameters, (schema, view))
    except Exception as exc:
        if isinstance(source, OperationalSource):
            state = getattr(exc, "sqlstate", None)
            if isinstance(state, bytes):
                state = state.decode("ascii", errors="ignore")
            state = (
                state if isinstance(state, str) and len(state) == 5 and state.isalnum() else None
            )
            transient = isinstance(exc, (TimeoutError, ConnectionError)) or (
                state is not None
                and (state.startswith("08") or state in {"40001", "40P01", "57014"})
            )
            raise OperationalReadError(
                "ERP view request failed", transient=transient, status=state
            ) from None
        raise ContextError(scrub(str(exc), dsn)) from None


def read_context_source(
    profile: ClientProfile, source: ContextSource, *, fetch: PostgresFetch = fetch_postgres_context
) -> list[SourceRow]:
    """Read one configured dataset and return rows keyed by canonical names."""
    config = require_reasoning(profile)
    if source not in config.sources:
        raise ContextError(
            f"Context source {source.id!r} is not configured for {profile.client_id!r}"
        )
    return _read_mapped(profile, source, fetch)


def read_operational_source(
    profile: ClientProfile, source: OperationalSource, *, fetch: PostgresFetch
) -> tuple[list[SourceRow], bool]:
    """Use only a deployment-mapped CSV or read-only view."""
    if source not in require_reasoning(profile).evidence.erp:
        raise ContextError(f"Operational source {source.id!r} is not configured for this client")
    rows = _read_mapped(profile, source, fetch)
    return rows[: source.max_rows], len(rows) > source.max_rows


def _read_mapped(
    profile: ClientProfile, source: ContextSource | OperationalSource, fetch: PostgresFetch
) -> list[SourceRow]:
    try:
        rows = (
            _csv_rows(profile, source)
            if source.kind == "csv"
            else _postgres_rows(profile, source, fetch)
        )
        fields = _source_fields(source)
        mapped = [{name: row[column] for name, column in fields.items()} for row in rows]
        if "client_id" in fields:
            foreign = sum(row["client_id"] != profile.client_id for row in mapped)
            if foreign:
                raise ContextSyncError(
                    "Dataset contains rows outside the configured client scope", invalid=foreign
                )
        return mapped
    except OperationalReadError:
        raise
    except ContextSyncError as exc:
        raise ContextSyncError(f"Context source {source.id!r}: {exc}", invalid=exc.invalid) from exc
    except (OSError, ValueError, KeyError, csv.Error, ContextError) as exc:
        raise ContextError(f"Context source {source.id!r}: {exc}") from exc
