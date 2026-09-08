from __future__ import annotations

import polars as pl

from engine.evaluate.metrics import add_forecast_value_add, aggregate, aggregate_series_level


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "series_id": ["S1", "S2", "S1", "S2"],
            "actual": [100.0, 200.0, 100.0, 200.0],
            "forecast": [90.0, 220.0, 100.0, 260.0],
        }
    )


def test_error_metrics_are_computed_as_defined():
    metrics = aggregate(_predictions(), ["model"]).sort("model")
    model_a = metrics.filter(pl.col("model") == "a").to_dicts()[0]

    assert model_a["mae"] == 15.0  # (10 + 20) / 2
    assert model_a["wape"] == 30.0 / 300.0
    assert model_a["bias"] == 5.0  # (-10 + 20) / 2
    assert model_a["n_obs"] == 2


def test_wape_is_null_when_actuals_are_all_zero():
    zeros = pl.DataFrame({"model": ["a"], "series_id": ["S1"], "actual": [0.0], "forecast": [5.0]})
    assert aggregate(zeros, ["model"])["wape"].item() is None


def test_forecast_value_add_is_relative_error_reduction():
    metrics = aggregate(_predictions(), ["model"])
    scored = add_forecast_value_add(metrics, baseline_model="b", scope_column=None)

    model_a = scored.filter(pl.col("model") == "a").to_dicts()[0]
    model_b = scored.filter(pl.col("model") == "b").to_dicts()[0]

    # Model b: MAE 30; model a: MAE 15 -> a halves the baseline error.
    assert model_a["fva_mae_pct"] == 50.0
    assert model_b["fva_mae_pct"] == 0.0


def test_series_level_metric_weights_every_series_equally():
    """One huge series must not decide the number, which is what MAE does."""
    predictions = pl.DataFrame(
        {
            "model": ["a"] * 3,
            "series_id": ["big", "small1", "small2"],
            "actual": [100_000.0, 100.0, 100.0],
            "forecast": [50_000.0, 100.0, 100.0],
        }
    )
    weighted = aggregate(predictions, ["model"])["wape"].item()
    per_series = aggregate_series_level(predictions, ["model"])["median_series_wape"].item()

    assert weighted > 0.4  # dominated by the one large series
    assert per_series == 0.0  # the median series is forecast perfectly
