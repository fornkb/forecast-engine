"""Behaviour that only shows up past one seasonal cycle.

Every Forecast Value Add figure is measured against seasonal_naive, so a
baseline that quietly stops being seasonal inflates every result above it. That
went unnoticed for the whole project because no run ever asked for a horizon
longer than four periods.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from engine.config.schema import ClientProfile
from engine.features.panel import lags_for, windows_for
from engine.forecast.base import ForecastContext, get_forecaster

SEASONAL_PERIOD = 12


@pytest.fixture
def three_years() -> pl.DataFrame:
    """36 monthly periods with a pronounced, unambiguous yearly cycle."""
    values = [100.0 + 60.0 * (month % 12 == 1) for month in range(36)]
    periods = [date(2023 + i // 12, i % 12 + 1, 1) for i in range(36)]
    return pl.DataFrame({"series_id": ["S1"] * 36, "period": periods, "quantity": values})


@pytest.fixture
def ctx(three_years) -> ForecastContext:
    meta = pl.DataFrame({"series_id": ["S1"], "customer": ["C"], "item_group": ["G"]})
    return ForecastContext(client=ClientProfile(client_id="t", display_name="T"), series_meta=meta)


def _forecasts(model: str, history, ctx, horizon: int) -> pl.DataFrame:
    return get_forecaster(model).predict(history, horizon, ctx).sort("horizon")


def test_seasonal_naive_stays_seasonal_past_one_cycle(three_years, ctx):
    """The bug: the lookup landed after the origin, found nothing, and fell back
    to the last observed value — so from horizon 12 on, the baseline was naive."""
    horizon = 2 * SEASONAL_PERIOD + 3
    seasonal = _forecasts("seasonal_naive", three_years, ctx, horizon)
    naive = _forecasts("naive", three_years, ctx, horizon)

    joined = seasonal.select("horizon", pl.col("forecast").alias("s")).join(
        naive.select("horizon", pl.col("forecast").alias("n")), on="horizon"
    )
    # Past two full cycles the two models must still disagree somewhere. At
    # horizon 12 and 24 they coincide legitimately: one cycle back from those is
    # the last observed period.
    beyond = joined.filter(pl.col("horizon") > SEASONAL_PERIOD)
    differing = beyond.filter((pl.col("s") - pl.col("n")).abs() > 1e-9)
    assert differing.height > 0


def test_the_seasonal_peak_is_still_found_two_cycles_out(three_years, ctx):
    """February carries the spike. A forecast 14 months out lands on February and
    must predict the spike, not the flat last value."""
    out = _forecasts("seasonal_naive", three_years, ctx, 2 * SEASONAL_PERIOD + 2)
    by_month = out.with_columns(pl.col("period").dt.month().alias("month"))

    february = by_month.filter(pl.col("month") == 2)["forecast"]
    other = by_month.filter(pl.col("month") != 2)["forecast"]
    assert february.min() > other.max()


def test_quantiles_survive_the_long_horizon(three_years, ctx):
    out = _forecasts("seasonal_naive", three_years, ctx, 2 * SEASONAL_PERIOD + 1)
    for name in ("q10", "q50", "q90"):
        assert out[name].null_count() == 0
    assert (out["q10"] <= out["q90"]).all()


def test_the_spread_widens_when_the_forecast_reaches_further_back(ctx):
    """A two-cycle-old observation is a worse predictor than a one-cycle-old one,
    and the interval has to say so.

    The fixture is a random walk rather than a trend: the variance of a lag-k
    difference grows with k there, which is the property under test. On a purely
    linear trend every lag-12 difference is the same number, so both spreads are
    zero and the test cannot fail.
    """
    rng = np.random.default_rng(11)
    walk = 200.0 + np.cumsum(rng.normal(0, 14, 60))
    history = pl.DataFrame(
        {
            "series_id": ["S1"] * 60,
            "period": [date(2021 + i // 12, i % 12 + 1, 1) for i in range(60)],
            "quantity": np.clip(walk, 1.0, None),
        }
    )

    out = _forecasts("seasonal_naive", history, ctx, 2 * SEASONAL_PERIOD)
    width = out.with_columns((pl.col("q90") - pl.col("q10")).alias("w"))

    one_cycle = width.filter(pl.col("horizon") <= SEASONAL_PERIOD)["w"].mean()
    two_cycles = width.filter(pl.col("horizon") > SEASONAL_PERIOD)["w"].mean()
    assert one_cycle > 0
    assert two_cycles > one_cycle


def test_a_series_shorter_than_the_lookback_still_gets_a_forecast(ctx):
    """Reaching back two cycles cannot leave short series unscoreable; the
    documented fallback to the last value has to still apply."""
    short = pl.DataFrame(
        {
            "series_id": ["S1"] * 8,
            "period": [date(2025, m, 1) for m in range(1, 9)],
            "quantity": [float(m) for m in range(1, 9)],
        }
    )
    out = _forecasts("seasonal_naive", short, ctx, 2 * SEASONAL_PERIOD)
    assert out["forecast"].null_count() == 0


# --- granularity-shaped features -----------------------------------------


def test_every_granularity_reaches_one_full_seasonal_cycle():
    """A lag set counted in bare periods means something different at each
    granularity. Weekly clients previously topped out at lag 12 — a quarter —
    so nothing in the feature set could see an annual cycle."""
    for granularity, cycle in (("monthly", 12), ("weekly", 52), ("daily", 364)):
        assert max(lags_for(granularity)) >= cycle, granularity
        assert max(windows_for(granularity)) >= cycle, granularity


def test_lags_are_sorted_distinct_and_positive():
    for granularity in ("daily", "weekly", "monthly"):
        lags = lags_for(granularity)
        assert list(lags) == sorted(set(lags))
        assert min(lags) >= 1
