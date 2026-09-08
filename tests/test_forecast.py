from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from engine.forecast.arbitration import (
    GLOBAL_STRATUM,
    apply_selection,
    pick_champions,
    select_per_series,
)
from engine.forecast.base import (
    ForecastContext,
    available_forecasters,
    enforce_contract,
    get_forecaster,
)
from tests.conftest import monthly_series


def test_registry_contains_the_phase0_portfolio():
    registered = available_forecasters()
    for name in ("naive", "seasonal_naive", "auto_ets", "croston", "lightgbm_global"):
        assert name in registered


def test_contract_clips_negative_and_null_forecasts():
    """Statistical models can extrapolate below zero; demand cannot go there."""
    raw = pl.DataFrame(
        {
            "series_id": ["S1", "S1"],
            "period": [date(2026, 1, 1), date(2026, 2, 1)],
            "horizon": [1, 2],
            "forecast": [-5.0, None],
        }
    )
    result = enforce_contract(raw, "test_model")
    assert result["forecast"].to_list() == [0.0, 0.0]


def test_contract_rejects_missing_columns():
    with pytest.raises(ValueError, match="forecast"):
        enforce_contract(pl.DataFrame({"series_id": ["S1"]}), "test_model")


def test_seasonal_naive_repeats_the_value_one_cycle_back(client, series_meta):
    values = list(range(1, 37))  # distinct values make the lookup unambiguous
    history = monthly_series("S1", [float(v) for v in values], start=date(2024, 1, 1))
    ctx = ForecastContext(client=client, series_meta=series_meta)

    result = get_forecaster("seasonal_naive").predict(history, 3, ctx).sort("horizon")

    # History ends Dec 2026 (value 36); Jan-Mar 2027 repeat Jan-Mar 2026 = 25, 26, 27.
    assert result["forecast"].to_list() == [25.0, 26.0, 27.0]
    assert result["period"].to_list() == [date(2027, 1, 1), date(2027, 2, 1), date(2027, 3, 1)]


def test_naive_repeats_the_last_value(client, series_meta):
    history = monthly_series("S1", [1.0, 2.0, 7.0])
    ctx = ForecastContext(client=client, series_meta=series_meta)

    result = get_forecaster("naive").predict(history, 2, ctx)
    assert result["forecast"].to_list() == [7.0, 7.0]


@pytest.mark.parametrize(
    "model", ["naive", "seasonal_naive", "auto_ets", "croston", "lightgbm_global"]
)
def test_every_forecaster_covers_every_series_and_horizon(model, history, client, series_meta):
    ctx = ForecastContext(client=client, series_meta=series_meta)
    result = get_forecaster(model).predict(history, 3, ctx)

    assert result.height == 2 * 3  # two series, three horizons
    assert result["forecast"].is_null().sum() == 0
    assert (result["forecast"] >= 0).all()
    assert sorted(result["horizon"].unique().to_list()) == [1, 2, 3]


def _scores() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": ["S1", "S1", "S2", "S2"],
            "model": ["champ", "challenger", "champ", "challenger"],
            "score": [10.0, 9.5, 10.0, 5.0],
        }
    )


def test_arbitration_keeps_champion_without_a_clear_margin():
    """Per-series selection on few observations is mostly noise, so a challenger
    must win by a margin before it displaces the champion."""
    selection = select_per_series(
        _scores(), "mae", champions={GLOBAL_STRATUM: "champ"}, min_improvement=0.15
    )
    chosen = dict(zip(selection["series_id"], selection["model"], strict=True))

    assert chosen["S1"] == "champ"  # 5% better is not enough
    assert chosen["S2"] == "challenger"  # 50% better is


def test_arbitration_without_margin_takes_the_best_score():
    selection = select_per_series(
        _scores(), "mae", champions={GLOBAL_STRATUM: "champ"}, min_improvement=0.0
    )
    assert set(selection["model"].to_list()) == {"challenger"}


def test_pick_champions_takes_the_lowest_error_in_each_stratum():
    scores = pl.DataFrame(
        {
            "stratum": ["Head", "Head", "Tail", "Tail"],
            "model": ["a", "b", "a", "b"],
            "score": [12.0, 7.0, 3.0, 9.0],
        }
    )
    assert pick_champions(scores, "score") == {"Head": "b", "Tail": "a"}


def test_each_stratum_defaults_to_its_own_champion():
    """A single global default is wrong for one group or another once models have
    complementary strengths across volume segments."""
    scores = pl.DataFrame(
        {
            "series_id": ["big", "small"],
            "model": ["pooled", "foundation"],
            "score": [10.0, 10.0],
        }
    )
    strata = pl.DataFrame({"series_id": ["big", "small"], "stratum": ["Head", "Tail"]})

    selection = select_per_series(
        scores,
        "mae",
        champions={"Head": "pooled", "Tail": "foundation"},
        strata=strata,
        min_improvement=0.15,
    )
    chosen = dict(zip(selection["series_id"], selection["model"], strict=True))

    assert chosen["big"] == "pooled"
    assert chosen["small"] == "foundation"


def test_apply_selection_keeps_one_model_per_series():
    forecasts = pl.DataFrame(
        {
            "series_id": ["S1", "S1", "S2", "S2"],
            "model": ["champ", "challenger", "champ", "challenger"],
            "period": [date(2026, 1, 1)] * 4,
            "forecast": [1.0, 2.0, 3.0, 4.0],
        }
    )
    selection = pl.DataFrame({"series_id": ["S1", "S2"], "model": ["champ", "challenger"]})

    result = apply_selection(forecasts, selection)
    assert result.height == 2
    assert result["forecast"].to_list() == [1.0, 4.0]
