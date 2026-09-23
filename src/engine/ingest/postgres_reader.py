"""Database-backed panel reader.

Reads a relation the ERP publishes — in a real deployment a view, which is the
integration contract — and hands it to the same canonical mapping every file
reader uses. Nothing downstream can tell the difference.

Connection details come from the client profile; the credential comes from an
environment variable named there, so a profile stays committable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import polars as pl

from engine.config.schema import ClientProfile
from engine.ingest.base import SourceReader, register_reader
from engine.ingest.mapping import to_canonical
from engine.ingest.schema import IngestError, IngestResult

# Both DSN spellings libpq accepts: a URI with inline credentials, and the
# keyword form. A profile may legitimately use either.
_URI_PASSWORD = re.compile(r"(?<=://)([^:/@]+):([^@]*)(?=@)")
_KEYWORD_PASSWORD = re.compile(r"(password=)(\S+)")


def redact(dsn: str) -> str:
    """The DSN with its password masked, safe to name in a message."""
    return _KEYWORD_PASSWORD.sub(r"\1***", _URI_PASSWORD.sub(r"\1:***", dsn))


def _password(dsn: str) -> str | None:
    for pattern, group in ((_URI_PASSWORD, 2), (_KEYWORD_PASSWORD, 2)):
        match = pattern.search(dsn)
        if match and match.group(group):
            return match.group(group)
    return None


def scrub(text: str, dsn: str) -> str:
    """Strip the DSN and its password out of driver output.

    Driver errors quote the connection string back at you, so anything that
    reaches a log or a terminal has to pass through here first. A short password
    will also mask any coincidental occurrence of the same characters elsewhere
    in the message — a garbled message is the better failure.
    """
    out = text.replace(dsn, redact(dsn))
    password = _password(dsn)
    if password:
        out = out.replace(password, "***")
    return out


def check_quantity_type(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Refuse a quantity column that is not already a number.

    The ADBC driver maps every PostgreSQL NUMERIC to an opaque type that arrives
    as text. Casting it downstream happens to work, so an uncast view produces
    correct totals while quietly depending on string parsing — the kind of thing
    that survives review and fails on a client whose numbers are formatted
    differently. A view has to name the type it means.
    """
    dtype = frame.schema.get(column)
    if dtype is None or dtype.is_numeric():
        # A missing column is to_canonical's error to raise; it names every
        # column it looked for, which is more use than anything said here.
        return frame
    if dtype.base_type() == pl.Decimal:
        return frame.with_columns(pl.col(column).cast(pl.Float64))
    raise IngestError(
        f"Column {column!r} arrived as {dtype}, not a number. PostgreSQL NUMERIC "
        "reaches this driver as opaque text, so the relation has to convert it: "
        f"expose it as double precision, e.g. SUM({column})::double precision."
    )


def fetch(query: str, dsn: str) -> pl.DataFrame:
    """Run one query and return the result. The only part that touches the network."""
    try:
        frame = pl.read_database_uri(query, dsn, engine="adbc")
    except ModuleNotFoundError as exc:
        raise IngestError(
            'The PostgreSQL driver is not installed. Install it with: pip install -e ".[postgres]"'
        ) from exc
    except Exception as exc:
        # `from None`: a chained traceback would reprint the driver's own
        # message, unscrubbed, right underneath this one.
        raise IngestError(f"Query against {redact(dsn)} failed: {scrub(str(exc), dsn)}") from None
    return frame


@register_reader
class PostgresPanelReader(SourceReader):
    """Reads a long-format demand relation out of PostgreSQL.

    Re-ingest is a full refresh of the configured label: rows previously loaded
    under it are deleted first. Narrowing `where` between runs therefore drops
    the history outside the new window rather than leaving it in place.
    """

    name = "postgres_panel"
    requires_path = False

    def read(self, path: Path | None, client: ClientProfile) -> IngestResult:
        config = client.source.database
        if config is None:  # unreachable via the config validator; cheap to assert
            raise IngestError("source.database is not configured for this client")

        dsn = os.environ.get(config.dsn_env)
        if not dsn:
            raise IngestError(
                f"Environment variable {config.dsn_env} is not set, so there is no "
                f"connection to {config.relation}. Set it to a PostgreSQL DSN, e.g. "
                "postgresql://engine_reader:<password>@localhost:5432/forecast_erp"
            )

        frame = check_quantity_type(fetch(config.query, dsn), client.source.column_map.quantity)
        return to_canonical(frame, client, config.label)
