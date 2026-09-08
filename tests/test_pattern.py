from __future__ import annotations

import polars as pl

from engine.config.schema import PatternThresholds
from engine.features.pattern import agreement_rate, classify, profile_series
from tests.conftest import monthly_series

THRESHOLDS = PatternThresholds()


def _stats(**overrides) -> dict[str, float]:
    base = {
        "zero_fraction": 0.0,
        "adi": 1.0,
        "cv2": 0.1,
        "trend_strength": 0.1,
        "seasonal_strength": 0.1,
        "residual_cv": 0.05,
        "slope": 0.0,
    }
    return {**base, **overrides}


def test_intermittency_is_checked_before_anything_else():
    """A mostly-zero series produces meaningless trend/seasonality scores."""
    stats = _stats(zero_fraction=0.5, trend_strength=0.9, seasonal_strength=0.95)
    assert classify(stats, THRESHOLDS) == "Intermittent"


def test_trend_direction_splits_growing_and_declining():
    assert classify(_stats(trend_strength=0.8, slope=0.05), THRESHOLDS) == "Growing"
    assert classify(_stats(trend_strength=0.8, slope=-0.05), THRESHOLDS) == "Declining"


def test_seasonal_wins_when_stronger_than_trend():
    stats = _stats(seasonal_strength=0.95, trend_strength=0.7)
    assert classify(stats, THRESHOLDS) == "Seasonal"


def test_volatility_separates_noisy_from_stable():
    assert classify(_stats(residual_cv=0.5), THRESHOLDS) == "Volatile"
    assert classify(_stats(residual_cv=0.02), THRESHOLDS) == "Stable"


def test_profiles_flat_and_intermittent_series():
    flat = monthly_series("flat", [100.0] * 36)
    sparse = monthly_series("sparse", [0.0, 0.0, 40.0] * 12)
    profiles = profile_series(pl.concat([flat, sparse]), THRESHOLDS, seasonal_period=12)

    by_id = {row["series_id"]: row for row in profiles.iter_rows(named=True)}
    assert by_id["flat"]["pattern"] == "Stable"
    assert by_id["sparse"]["pattern"] == "Intermittent"
    assert by_id["sparse"]["zero_fraction"] > 0.6
    assert by_id["flat"]["n_periods"] == 36


def test_agreement_rate_counts_diagonal():
    matrix = pl.DataFrame(
        {
            "ground_truth": ["Stable", "Stable", "Seasonal"],
            "predicted": ["Stable", "Seasonal", "Seasonal"],
            "n": [8, 2, 10],
        }
    )
    assert agreement_rate(matrix) == 18 / 20
