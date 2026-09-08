from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from engine.config.schema import ClientProfile
from engine.ingest import get_reader
from engine.ingest.schema import IngestError, validate_canonical
from engine.store import EngineStore

SOURCE_FILE = Path("data/raw/Demand_History_3_Years.xlsx")


def _dim(series_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"series_id": series_ids})


def _fact(rows: list[tuple[str, date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": [r[0] for r in rows],
            "period": [r[1] for r in rows],
            "quantity": [r[2] for r in rows],
        }
    )


def test_duplicate_series_period_is_rejected():
    fact = _fact([("S1", date(2024, 1, 1), 5.0), ("S1", date(2024, 1, 1), 6.0)])
    with pytest.raises(IngestError, match="duplicate"):
        validate_canonical(_dim(["S1"]), fact)


def test_negative_quantity_is_rejected():
    fact = _fact([("S1", date(2024, 1, 1), -5.0)])
    with pytest.raises(IngestError, match="negative"):
        validate_canonical(_dim(["S1"]), fact)


def test_null_key_is_rejected():
    fact = pl.DataFrame(
        {"series_id": ["S1"], "period": [None], "quantity": [1.0]},
        schema={"series_id": pl.Utf8, "period": pl.Date, "quantity": pl.Float64},
    )
    with pytest.raises(IngestError, match="null"):
        validate_canonical(_dim(["S1"]), fact)


def test_orphan_fact_row_is_rejected():
    fact = _fact([("S9", date(2024, 1, 1), 1.0)])
    with pytest.raises(IngestError, match="unknown series_id"):
        validate_canonical(_dim(["S1"]), fact)


def test_missing_source_file_is_reported(tmp_path):
    profile = ClientProfile(client_id="test_client", display_name="Test")
    with pytest.raises(IngestError, match="not found"):
        get_reader("excel_demand").read(tmp_path / "absent.xlsx", profile)


@pytest.mark.skipif(not SOURCE_FILE.exists(), reason="source workbook not present")
def test_real_workbook_reconciles():
    profile = ClientProfile(client_id="test_client", display_name="Test")
    result = get_reader(profile.source.reader).read(SOURCE_FILE, profile)

    assert result.report.rows_reconcile
    assert result.report.quantity_reconciles
    assert result.fact_demand.height == result.report.source_rows
    # Series attributes were collapsed into exactly one dimension row each.
    assert result.dim_series["series_id"].n_unique() == result.dim_series.height


@pytest.mark.skipif(not SOURCE_FILE.exists(), reason="source workbook not present")
def test_reingesting_same_file_is_idempotent(tmp_path):
    profile = ClientProfile(client_id="test_client", display_name="Test")
    result = get_reader(profile.source.reader).read(SOURCE_FILE, profile)
    store = EngineStore(profile.client_id, root=tmp_path)

    store.replace_source(result.dim_series, result.fact_demand)
    first = store.load_fact().height
    store.replace_source(result.dim_series, result.fact_demand)

    assert store.load_fact().height == first
    assert store.load_dim().height == result.dim_series.height


def test_store_round_trips_profiles(tmp_path):
    store = EngineStore("test_client", root=tmp_path)
    store.init_schema()
    profiles = pl.DataFrame(
        {
            "series_id": ["S1"],
            "n_periods": [36],
            "n_zeros": [0],
            "zero_fraction": [0.0],
            "mean_demand": [10.0],
            "cv": [0.1],
            "adi": [1.0],
            "cv2": [0.01],
            "trend_strength": [0.2],
            "seasonal_strength": [0.3],
            "residual_cv": [0.05],
            "slope": [0.0],
            "pattern": ["Stable"],
        }
    )
    store.write_series_profiles(profiles)
    loaded = store.load_series_profiles()

    assert loaded.height == 1
    assert loaded["pattern"].item() == "Stable"
    assert isinstance(loaded["computed_at"].item(), datetime)
