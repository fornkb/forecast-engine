from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from pydantic import ValidationError

from engine.config.schema import ClientProfile, PostgresOutputConfig
from engine.store import EngineStore, PublishError, publish_forecasts

# Port 1: nothing listens there. A unit test must not be able to reach a real
# database, and a plausible-looking localhost DSN can — PostgreSQL installs
# commonly trust local connections, so a wrong password is still let in. An
# earlier version of this file wrote test rows into a live server that way.
DSN = "postgresql://engine_writer:hunter2@127.0.0.1:1/forecast_erp"


def _profile(**output) -> ClientProfile:
    return ClientProfile(
        client_id="acme",
        display_name="Acme",
        output={"postgres": output} if output else {},
    )


def _store_with_forecast(tmp_path) -> tuple[EngineStore, int]:
    store = EngineStore("acme", root=tmp_path)
    run_id = store.start_run("forecast", {"horizon": 2})
    store.write_forecasts(
        run_id,
        pl.DataFrame(
            {
                "series_id": ["S1", "S1"],
                "cutoff_period": [date(2024, 12, 1)] * 2,
                "period": [date(2025, 1, 1), date(2025, 2, 1)],
                "horizon": [1, 2],
                "model": ["lightgbm_global"] * 2,
                "forecast": [10.0, 12.0],
            }
        ),
    )
    return store, run_id


# --- configuration --------------------------------------------------------


def test_publishing_without_an_output_block_says_so(tmp_path):
    store, run_id = _store_with_forecast(tmp_path)
    with pytest.raises(PublishError, match="no output.postgres block"):
        publish_forecasts(store, _profile(), run_id)


def test_missing_environment_variable_names_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_WRITE_DSN", raising=False)
    store, run_id = _store_with_forecast(tmp_path)
    with pytest.raises(PublishError, match="TEST_WRITE_DSN"):
        publish_forecasts(store, _profile(dsn_env="TEST_WRITE_DSN"), run_id)


def test_output_defaults_to_nowhere():
    """A client that never opts in must not acquire a publish target by accident."""
    assert ClientProfile(client_id="acme", display_name="Acme").output.postgres is None


def test_target_schema_must_be_a_plain_identifier():
    """It is interpolated into SQL, so it never gets to contain anything clever."""
    with pytest.raises(ValidationError, match="alphanumeric"):
        PostgresOutputConfig(target_schema="public; DROP TABLE users")


def test_read_and_write_default_to_different_variables():
    """One role reads the ERP, another writes the output. Sharing a DSN would
    quietly collapse that separation."""
    from engine.config.schema import DatabaseSourceConfig

    assert PostgresOutputConfig().dsn_env != DatabaseSourceConfig(relation="x").dsn_env


# --- refusals -------------------------------------------------------------


def test_empty_run_is_refused_before_connecting(tmp_path, monkeypatch):
    """An empty publish would clear the previous one and leave the ERP with
    nothing, so it fails instead."""
    monkeypatch.setenv("TEST_WRITE_DSN", DSN)
    store = EngineStore("acme", root=tmp_path)
    run_id = store.start_run("forecast", {})
    with pytest.raises(PublishError, match="no forecasts in the store"):
        publish_forecasts(store, _profile(dsn_env="TEST_WRITE_DSN"), run_id)


def test_connection_failure_does_not_leak_the_password(tmp_path, monkeypatch):
    """Also proves the unit suite cannot reach a live database."""
    monkeypatch.setenv("TEST_WRITE_DSN", DSN)
    store, run_id = _store_with_forecast(tmp_path)
    with pytest.raises(PublishError) as excinfo:
        publish_forecasts(store, _profile(dsn_env="TEST_WRITE_DSN"), run_id)
    assert "hunter2" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
