from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from engine.config.schema import ClientProfile

PROJECT_DATA = "data/raw/Demand_History_3_Years.xlsx"


@pytest.fixture
def client() -> ClientProfile:
    return ClientProfile(client_id="test_client", display_name="Test Client")


def monthly_series(series_id: str, values: list[float], start: date = date(2024, 1, 1)):
    periods = [
        date(start.year + (start.month - 1 + i) // 12, (start.month - 1 + i) % 12 + 1, 1)
        for i in range(len(values))
    ]
    return pl.DataFrame(
        {
            "series_id": [series_id] * len(values),
            "period": periods,
            "quantity": [float(v) for v in values],
        }
    )


@pytest.fixture
def history() -> pl.DataFrame:
    """Two clean 36-month series: one flat, one with a strong yearly cycle."""
    flat = monthly_series("S1", [100.0] * 36)
    seasonal = monthly_series("S2", [50 + 40 * (month % 12 == 0) for month in range(1, 37)])
    return pl.concat([flat, seasonal])


@pytest.fixture
def series_meta() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": ["S1", "S2"],
            "customer": ["C1", "C2"],
            "item_group": ["G1", "G1"],
            "source_sheet": ["Sheet1", "Sheet1"],
        }
    )
