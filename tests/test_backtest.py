from __future__ import annotations

from datetime import date

import pytest

from engine.evaluate.backtest import BacktestError, build_plan, truncate_history
from engine.evaluate.backtest import run as run_backtest
from tests.conftest import monthly_series


def test_plan_leaves_a_full_horizon_after_every_origin(client, history):
    plan = build_plan(history, client)

    # 36 months (Jan 2024 - Dec 2026), horizon 3: the last usable cutoff is
    # Sep 2026, leaving Oct-Dec to score against.
    assert plan.n_origins == 3
    assert plan.cutoffs == [date(2026, 7, 1), date(2026, 8, 1), date(2026, 9, 1)]
    assert plan.horizon == 3


def test_plan_rejects_history_shorter_than_the_horizon(client):
    short = monthly_series("S1", [1.0, 2.0])
    with pytest.raises(BacktestError, match="too few"):
        build_plan(short, client)


def test_plan_respects_the_minimum_training_window(client):
    profile = client.model_copy(
        update={"evaluation": client.evaluation.model_copy(update={"min_train_periods": 40})}
    )
    history = monthly_series("S1", [1.0] * 36)
    with pytest.raises(BacktestError, match="below the configured minimum"):
        build_plan(history, profile)


def test_truncation_keeps_the_most_recent_history(history):
    """Cold start is simulated by taking history away, not by dropping series."""
    truncated = truncate_history(history, 4)

    assert truncated.height == 2 * 4  # two series, four periods each
    assert truncated["period"].max() == history["period"].max()
    assert truncated["period"].min() > history["period"].min()


def test_truncation_beyond_available_history_is_a_no_op(history):
    assert truncate_history(history, 999).height == history.height


def test_backtest_never_trains_on_the_period_it_forecasts(client, history, series_meta):
    """The guard against leakage: forecast periods must sit strictly after the cutoff."""
    profile = client.model_copy(
        update={"models": client.models.model_copy(update={"enabled": ["naive", "seasonal_naive"]})}
    )
    plan = build_plan(history, profile)
    predictions = run_backtest(history, series_meta, profile, plan)

    assert (predictions["period"] > predictions["cutoff_period"]).all()
    assert predictions["model"].n_unique() == 2
    # two series x three horizons x three origins x two models
    assert predictions.height == 2 * 3 * 3 * 2
    assert predictions["actual"].is_null().sum() == 0
