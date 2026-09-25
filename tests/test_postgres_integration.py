"""Round-trip against a live PostgreSQL.

Deselected by default. Run with a loaded database:

    set ENGINE_PG_DSN=postgresql://engine_reader:<password>@localhost:5432/forecast_erp
    pytest -m integration

Everything above the driver is covered by test_postgres_reader.py; what only a
real database can check is that the view's column names, types and grain match
what the client profile claims.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from engine.config.loader import load_client
from engine.ingest.base import get_reader

pytestmark = pytest.mark.integration

WORKBOOK = Path("data/raw/Demand_History_3_Years.xlsx")


@pytest.fixture(scope="module")
def result():
    if not os.environ.get("ENGINE_PG_DSN"):
        pytest.skip("ENGINE_PG_DSN is not set")
    profile = load_client("teknic_mumbai_pg")
    return get_reader(profile.source.reader).read(None, profile)


def test_view_maps_cleanly_onto_the_canonical_model(result):
    """Reaching here at all means the view satisfied every canonical contract —
    no nulls in keys, no negatives, no duplicate series-periods, no orphans."""
    assert result.fact_demand.height > 0
    assert result.report.rows_reconcile
    assert result.report.quantity_reconciles


def test_quantity_is_float_not_decimal(result):
    assert result.fact_demand.schema["quantity"] == pl.Float64


def test_database_matches_the_workbook(result):
    """The database is a different route to the same numbers. If it disagrees,
    the view's grain or the column map is wrong."""
    if not WORKBOOK.exists():
        pytest.skip(f"{WORKBOOK} not available")

    book = pl.read_excel(WORKBOOK)
    assert result.report.series_count == book["Series ID"].n_unique()
    assert result.report.canonical_quantity_sum == pytest.approx(
        float(book["Demand Quantity"].sum())
    )
    assert result.report.period_min == str(book["Demand Month"].min())
    assert result.report.period_max == str(book["Demand Month"].max())


def test_the_engine_role_cannot_write(result):
    """The grant is SELECT on one view. If this insert succeeds, the engine can
    corrupt the ERP."""
    from engine.ingest.postgres_reader import fetch
    from engine.ingest.schema import IngestError

    with pytest.raises(IngestError):
        fetch("INSERT INTO erp.sales_history (series_id) VALUES ('x')", os.environ["ENGINE_PG_DSN"])


# --- write-back -----------------------------------------------------------


@pytest.fixture(scope="module")
def published():
    """Publish the latest forecast run, and hand back a way to query it."""
    if not os.environ.get("ENGINE_PG_WRITE_DSN"):
        pytest.skip("ENGINE_PG_WRITE_DSN is not set")
    from engine.store import EngineStore, publish_forecasts

    profile = load_client("teknic_mumbai_pg")
    store = EngineStore(profile.client_id)
    run_id = store.latest_run("forecast")
    if run_id is None:
        pytest.skip("no forecast run in the store; run `engine forecast` first")
    return publish_forecasts(store, profile, run_id)


def _query(sql: str):
    import adbc_driver_postgresql.dbapi as pg

    with pg.connect(os.environ["ENGINE_PG_WRITE_DSN"]) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def test_published_rows_are_visible_through_the_current_view(published):
    rows, series = _query(
        "SELECT count(*), count(DISTINCT series_id) "
        "FROM forecast_output.v_current_forecast "
        f"WHERE client_id = 'teknic_mumbai_pg' AND run_id = {published.run_id}"
    )[0]
    assert rows == published.forecast_rows
    assert series == published.series_count


def test_republishing_replaces_rather_than_duplicates(published):
    """The command has to be safe to re-run after a partial failure."""
    from engine.store import EngineStore, publish_forecasts

    profile = load_client("teknic_mumbai_pg")
    again = publish_forecasts(EngineStore(profile.client_id), profile, published.run_id)

    total = _query(
        "SELECT count(*) FROM forecast_output.demand_forecast "
        f"WHERE client_id = 'teknic_mumbai_pg' AND run_id = {published.run_id}"
    )[0][0]
    assert total == again.forecast_rows == published.forecast_rows


def test_quantiles_arrive_sorted_and_complete(published):
    levels = [
        r[0]
        for r in _query(
            "SELECT DISTINCT level FROM forecast_output.v_current_forecast_quantile "
            f"WHERE client_id = 'teknic_mumbai_pg' AND run_id = {published.run_id} ORDER BY level"
        )
    ]
    assert levels == sorted(levels)
    assert levels == load_client("teknic_mumbai_pg").probabilistic.levels


def test_the_writer_role_cannot_reach_the_erp_schema(published):
    """Read and write are different roles. If the writer can see sales history,
    a mistake in the publish path can reach the ERP's own data."""
    import adbc_driver_postgresql.dbapi as pg

    with (
        pytest.raises(Exception, match="(?i)permission denied"),
        pg.connect(os.environ["ENGINE_PG_WRITE_DSN"]) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT 1 FROM erp.v_demand_history LIMIT 1")


# --- stock policy write-back ----------------------------------------------


@pytest.fixture(scope="module")
def published_policy(published):
    """Publish the stock policy derived from the forecast `published` published."""
    from engine.store import EngineStore, publish_policy

    profile = load_client("teknic_mumbai_pg")
    store = EngineStore(profile.client_id)
    decision_run = store.latest_run_from("decision", "forecast_run_id", published.run_id)
    if decision_run is None:
        pytest.skip("no decision run built from the published forecast; run `engine decide`")
    return publish_policy(store, profile, decision_run, published.run_id)


def test_every_series_reaches_the_reorder_view(published_policy):
    """The point of the whole layer: an item with no reorder point is an item
    nobody reorders."""
    rows, series = _query(
        "SELECT count(*), count(DISTINCT series_id) "
        "FROM forecast_output.v_current_stock_policy "
        f"WHERE client_id = 'teknic_mumbai_pg' AND run_id = {published_policy.run_id}"
    )[0]
    assert rows == series == published_policy.series_count


def test_the_published_policy_names_the_forecast_it_came_from(published_policy, published):
    """Without the link nobody can tell whether a reorder point belongs to the
    forecast published beside it."""
    linked = _query(
        "SELECT forecast_run_id FROM forecast_output.policy_run "
        f"WHERE client_id = 'teknic_mumbai_pg' AND run_id = {published_policy.run_id}"
    )[0][0]
    assert linked == published.run_id


def test_stock_levels_are_ordered_and_non_negative(published_policy):
    """target_stock covers lead time plus review, reorder_point covers lead time
    alone, so one cannot exceed the other."""
    bad = _query(
        "SELECT count(*) FROM forecast_output.v_current_stock_policy "
        f"WHERE client_id = 'teknic_mumbai_pg' AND run_id = {published_policy.run_id} "
        "AND (safety_stock < 0 OR reorder_point < 0 OR reorder_point > target_stock)"
    )[0][0]
    assert bad == 0


def test_republishing_a_policy_replaces_rather_than_duplicates(published_policy):
    from engine.store import EngineStore, publish_policy

    profile = load_client("teknic_mumbai_pg")
    again = publish_policy(
        EngineStore(profile.client_id),
        profile,
        published_policy.run_id,
        _query(
            "SELECT forecast_run_id FROM forecast_output.policy_run "
            f"WHERE run_id = {published_policy.run_id}"
        )[0][0],
    )
    total = _query(
        "SELECT count(*) FROM forecast_output.stock_policy "
        f"WHERE client_id = 'teknic_mumbai_pg' AND run_id = {published_policy.run_id}"
    )[0][0]
    assert total == again.series_count == published_policy.series_count
