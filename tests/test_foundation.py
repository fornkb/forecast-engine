from __future__ import annotations

import pytest

from engine.forecast.base import available_forecasters, get_forecaster
from engine.forecast.foundation import _median_column

chronos = pytest.importorskip("chronos", reason="chronos-forecasting not installed")


def test_chronos_is_registered():
    assert "chronos2" in available_forecasters()


def test_median_column_prefers_the_requested_median():
    assert _median_column([0.1, 0.5, 0.9]) == "0.5"


def test_median_column_falls_back_when_median_not_requested():
    """Chronos still returns a point forecast when 0.5 is not among the levels."""
    assert _median_column([0.1, 0.9]) == "predictions"


@pytest.mark.slow
def test_chronos_produces_a_forecast_for_every_series(history, client, series_meta):
    """Loads the real pretrained model — slow, and needs the weights cached."""
    from engine.forecast.base import ForecastContext

    ctx = ForecastContext(client=client, series_meta=series_meta)
    result = get_forecaster("chronos2").predict(history, 3, ctx)

    assert result.height == 2 * 3
    assert result["forecast"].is_null().sum() == 0
    assert (result["forecast"] >= 0).all()
