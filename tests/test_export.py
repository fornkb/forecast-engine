from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from engine.store.export import export_forecasts


def _forecasts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": ["S2", "S1", "S1"],
            "period": [date(2026, 9, 1), date(2026, 10, 1), date(2026, 9, 1)],
            "horizon": [1, 2, 1],
            "model": ["chronos2", "naive", "naive"],
            "forecast": [172.4, 31.0, 30.0],
            "q90": [187.0, None, None],
            "q10": [162.0, None, None],
            "cutoff_period": [date(2026, 8, 1)] * 3,
        }
    )


def _series_meta() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": ["S1", "S2"],
            "customer": ["TEKNIC:MUMBAI", "TEKNIC:MUMBAI"],
            "item_code": ["F2F400NN00", "LCL110B800"],
            "description": ["2ALRF4L", "3PLBR8L"],
            "uom": ["EA", "EA"],
            "item_group": ["G1", "G1"],
        }
    )


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_every_row_is_written_with_item_attributes(tmp_path, suffix):
    path = tmp_path / f"forecasts{suffix}"
    rows = export_forecasts(_forecasts(), _series_meta(), path)

    written = pl.read_csv(path) if suffix == ".csv" else pl.read_parquet(path)
    assert rows == written.height == 3
    # A bare series id means nothing to a planner.
    assert written["item_code"].to_list() == ["F2F400NN00", "F2F400NN00", "LCL110B800"]


def test_columns_read_left_to_right_as_a_planner_would(tmp_path):
    path = tmp_path / "forecasts.csv"
    export_forecasts(_forecasts(), _series_meta(), path)

    assert pl.read_csv(path).columns == [
        "series_id",
        "customer",
        "item_code",
        "description",
        "uom",
        "period",
        "horizon",
        "model",
        "forecast",
        "q10",
        "q90",
    ]


def test_rows_are_ordered_by_item_then_period(tmp_path):
    path = tmp_path / "forecasts.parquet"
    export_forecasts(_forecasts(), _series_meta(), path)

    written = pl.read_parquet(path)
    assert written.select("series_id", "horizon").rows() == [("S1", 1), ("S1", 2), ("S2", 1)]


def test_csv_opens_as_utf8_in_excel(tmp_path):
    """Excel only reads a CSV as UTF-8 when it starts with a byte-order mark."""
    path = tmp_path / "forecasts.csv"
    export_forecasts(_forecasts(), _series_meta(), path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_missing_folders_are_created(tmp_path):
    path = tmp_path / "exports" / "september" / "forecasts.csv"
    export_forecasts(_forecasts(), _series_meta(), path)
    assert path.exists()


@pytest.mark.parametrize("name", ["forecasts.xlsx", "forecasts.json", "forecasts"])
def test_unsupported_formats_are_refused_clearly(tmp_path, name):
    with pytest.raises(ValueError, match=".csv or .parquet"):
        export_forecasts(_forecasts(), _series_meta(), tmp_path / name)
