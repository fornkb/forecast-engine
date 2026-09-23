from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from pydantic import ValidationError

from engine.config.schema import ClientProfile, DatabaseSourceConfig
from engine.ingest import postgres_reader
from engine.ingest.base import get_reader
from engine.ingest.schema import IngestError

DSN = "postgresql://engine_reader:hunter2@localhost:5432/forecast_erp"


def _profile() -> ClientProfile:
    return ClientProfile(
        client_id="acme",
        display_name="Acme",
        source={
            "reader": "postgres_panel",
            "database": {"dsn_env": "TEST_PG_DSN", "relation": "erp.v_demand_history"},
            "column_map": {
                "series_id": "series_id",
                "customer": "customer",
                "item_code": "item_code",
                "description": "description",
                "item_group": "item_group",
                "period": "period",
                "quantity": "quantity",
                "uom": "uom",
                "source_sheet": "source_sheet",
                "ground_truth_pattern": "demand_pattern",
            },
        },
    )


def _view_rows() -> pl.DataFrame:
    """A frame shaped like erp.v_demand_history, with NUMERIC as Decimal."""
    return pl.DataFrame(
        {
            "series_id": ["S0001", "S0001", "S0002"],
            "customer": ["TEKNIC:MUMBAI"] * 3,
            "item_code": ["F2F400NN00", "F2F400NN00", "G1A100NN00"],
            "description": ["Widget", "Widget", "Gadget"],
            "item_group": ["FG", "FG", "FG"],
            "source_sheet": ["Mumbai-FG"] * 3,
            "uom": ["EA"] * 3,
            "demand_pattern": ["Smooth", "Smooth", "Erratic"],
            "period": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 1, 1)],
            "quantity": pl.Series([24.0, 30.0, 7.0], dtype=pl.Decimal(18, 3)),
        }
    )


# --- credentials ----------------------------------------------------------


def test_redact_masks_a_uri_password():
    assert "hunter2" not in postgres_reader.redact(DSN)
    assert "engine_reader" in postgres_reader.redact(DSN)


def test_redact_masks_a_keyword_password():
    dsn = "host=localhost dbname=forecast_erp user=engine_reader password=hunter2"
    assert "hunter2" not in postgres_reader.redact(dsn)


def test_scrub_removes_the_password_a_driver_echoed_back():
    """Driver errors quote the connection string. Nothing may reach a log with
    the password still in it."""
    message = f'failed to connect to "{DSN}": password authentication failed for "hunter2"'
    scrubbed = postgres_reader.scrub(message, DSN)
    assert "hunter2" not in scrubbed


def test_fetch_failure_does_not_chain_the_original_error(monkeypatch):
    """A chained traceback would reprint the driver's unscrubbed message."""

    def boom(*_args, **_kwargs):
        raise RuntimeError(f"connection to {DSN} refused")

    monkeypatch.setattr(pl, "read_database_uri", boom)
    with pytest.raises(IngestError) as excinfo:
        postgres_reader.fetch("SELECT 1", DSN)
    assert "hunter2" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_missing_environment_variable_names_the_variable(monkeypatch):
    monkeypatch.delenv("TEST_PG_DSN", raising=False)
    with pytest.raises(IngestError, match="TEST_PG_DSN"):
        get_reader("postgres_panel").read(None, _profile())


# --- config ---------------------------------------------------------------


def test_postgres_reader_without_a_database_block_is_rejected():
    with pytest.raises(ValidationError, match="requires a source.database"):
        ClientProfile(client_id="acme", display_name="Acme", source={"reader": "postgres_panel"})


def test_database_block_on_a_file_reader_is_rejected():
    """Silently ignoring it would let a profile claim a source it never reads."""
    with pytest.raises(ValidationError, match="would be ignored"):
        ClientProfile(
            client_id="acme",
            display_name="Acme",
            source={
                "reader": "excel_demand",
                "database": {"relation": "erp.v_demand_history"},
            },
        )


def test_source_label_defaults_to_the_relation_and_carries_no_credentials():
    config = DatabaseSourceConfig(relation="erp.v_demand_history")
    assert config.label == "postgres:erp.v_demand_history"


def test_where_clause_is_applied_server_side():
    config = DatabaseSourceConfig(relation="erp.v_demand_history", where="period >= '2024-01-01'")
    assert config.query == ("SELECT * FROM erp.v_demand_history WHERE period >= '2024-01-01'")


# --- mapping --------------------------------------------------------------


def test_reader_maps_the_view_onto_the_canonical_model(monkeypatch):
    monkeypatch.setenv("TEST_PG_DSN", DSN)
    monkeypatch.setattr(postgres_reader, "fetch", lambda query, dsn: _view_rows())

    result = get_reader("postgres_panel").read(None, _profile())

    assert result.fact_demand.height == 3
    assert result.dim_series.height == 2
    assert result.report.source_file == "postgres:erp.v_demand_history"
    assert result.report.canonical_quantity_sum == pytest.approx(61.0)
    # NUMERIC must not survive as Decimal into the canonical model.
    assert result.fact_demand.schema["quantity"] == pl.Float64


def test_text_quantity_is_refused_with_the_fix_in_the_message(monkeypatch):
    """An uncast NUMERIC arrives as text and parses correctly downstream, so
    the totals reconcile and nothing looks wrong. It has to fail here instead."""
    monkeypatch.setenv("TEST_PG_DSN", DSN)
    rows = _view_rows().with_columns(pl.col("quantity").cast(pl.Utf8))
    monkeypatch.setattr(postgres_reader, "fetch", lambda query, dsn: rows)

    with pytest.raises(IngestError, match="double precision"):
        get_reader("postgres_panel").read(None, _profile())


def test_float_quantity_passes_through_untouched():
    frame = _view_rows().with_columns(pl.col("quantity").cast(pl.Float64))
    assert postgres_reader.check_quantity_type(frame, "quantity") is frame


def test_reader_declares_that_it_needs_no_path():
    """The CLI uses this to decide whether --file is required."""
    assert get_reader("postgres_panel").requires_path is False
    assert get_reader("excel_demand").requires_path is True
