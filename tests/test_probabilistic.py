from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from engine.config.schema import ClientProfile, ProbabilisticConfig, quantile_column
from engine.evaluate.metrics import aggregate_probabilistic
from engine.forecast.base import ForecastContext, conform, enforce_contract, get_forecaster


def test_quantile_column_names_are_sql_safe():
    assert quantile_column(0.1) == "q10"
    assert quantile_column(0.5) == "q50"
    assert quantile_column(0.975) == "q97_5"


def test_levels_must_lie_inside_the_unit_interval():
    with pytest.raises(ValueError, match="between 0 and 1"):
        ProbabilisticConfig(levels=[0.5, 1.0])


def test_levels_are_sorted_and_unique():
    assert ProbabilisticConfig(levels=[0.9, 0.1]).levels == [0.1, 0.9]
    with pytest.raises(ValueError, match="not repeat"):
        ProbabilisticConfig(levels=[0.5, 0.5])


def test_crossed_quantiles_are_repaired():
    """Each level is fitted independently, so nothing stops q90 landing below
    q10; a negative-width interval must never reach a safety-stock calculation."""
    raw = pl.DataFrame(
        {
            "series_id": ["S1"],
            "period": [date(2026, 1, 1)],
            "horizon": [1],
            "forecast": [10.0],
            "q10": [14.0],
            "q50": [9.0],
            "q90": [11.0],
        }
    )
    result = enforce_contract(raw, "test_model")
    assert result["q10"].item() <= result["q50"].item() <= result["q90"].item()


def test_quantiles_are_clipped_at_zero():
    raw = pl.DataFrame(
        {
            "series_id": ["S1"],
            "period": [date(2026, 1, 1)],
            "horizon": [1],
            "forecast": [2.0],
            "q10": [-5.0],
        }
    )
    assert enforce_contract(raw, "test_model")["q10"].item() == 0.0


def test_point_only_forecasters_keep_working():
    raw = pl.DataFrame(
        {
            "series_id": ["S1"],
            "period": [date(2026, 1, 1)],
            "horizon": [1],
            "forecast": [3.0],
        }
    )
    result = enforce_contract(raw, "test_model")
    assert result.columns == ["series_id", "period", "horizon", "forecast"]


def _scored() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model": ["a", "a", "a", "a"],
            "actual": [10.0, 10.0, 10.0, 10.0],
            "q10": [8.0, 8.0, 8.0, 12.0],
            "q90": [12.0, 12.0, 12.0, 14.0],
        }
    )


def test_interval_coverage_counts_actuals_inside_the_band():
    scored = aggregate_probabilistic(_scored(), ["model"], [0.1, 0.9])
    # Three of four actuals fall within [q10, q90]; the fourth sits below q10.
    assert scored["interval_coverage"].item() == 0.75
    assert scored["nominal_coverage"].item() == pytest.approx(0.8)


def test_pinball_penalises_the_expensive_direction_harder():
    """Missing high on a high quantile should cost less than missing low."""
    under = pl.DataFrame({"model": ["a"], "actual": [10.0], "q90": [5.0]})
    over = pl.DataFrame({"model": ["a"], "actual": [10.0], "q90": [15.0]})

    under_loss = aggregate_probabilistic(under, ["model"], [0.9])["mean_pinball"].item()
    over_loss = aggregate_probabilistic(over, ["model"], [0.9])["mean_pinball"].item()

    assert under_loss > over_loss


def test_no_quantiles_scores_as_none():
    frame = pl.DataFrame({"model": ["a"], "actual": [1.0], "forecast": [1.0]})
    assert aggregate_probabilistic(frame, ["model"], [0.1, 0.9]) is None


@pytest.mark.parametrize("model", ["naive", "seasonal_naive", "lightgbm_global"])
def test_baselines_and_pooled_model_emit_usable_intervals(model, history, client, series_meta):
    ctx = ForecastContext(client=client, series_meta=series_meta)
    result = get_forecaster(model).predict(history, 3, ctx)

    for name in ("q10", "q50", "q90"):
        assert name in result.columns, f"{model} produced no {name}"
        assert result[name].is_null().sum() == 0

    assert (result["q10"] <= result["q90"]).all()


def test_models_with_and_without_quantiles_combine_into_one_frame(history, client, series_meta):
    """Regression: `engine forecast` crashed when one model emitted quantiles and
    another did not — nulls appended afterwards land in a different column
    position, and a vertical concat matches by position, not by name.

    The quantile-less side is built by hand rather than taken from a model that
    happens not to produce quantiles. Every enabled model produces them now, and
    a regression test should not stop exercising its own case the moment that
    changes — which is what happened when croston learned to emit intervals.
    """
    ctx = ForecastContext(client=client, series_meta=series_meta)
    quantiles = ["q10", "q50", "q90"]
    columns = ["series_id", "period", "horizon", "model", "forecast"]

    with_quantiles = conform(
        get_forecaster("seasonal_naive")
        .predict(history, 2, ctx)
        .with_columns(pl.lit("seasonal_naive").alias("model")),
        columns,
        quantiles,
    )
    without = conform(
        with_quantiles.select(columns).with_columns(pl.lit("point_only").alias("model")),
        columns,
        quantiles,
    )
    combined = pl.concat([without, with_quantiles])

    assert combined.columns == columns + quantiles
    assert combined.filter(pl.col("model") == "point_only")["q10"].is_null().all()
    assert combined.filter(pl.col("model") == "seasonal_naive")["q10"].is_not_null().all()


def test_probabilistic_can_be_switched_off(history, series_meta):
    profile = ClientProfile(
        client_id="no_quantiles",
        display_name="No quantiles",
        probabilistic={"enabled": False},
    )
    ctx = ForecastContext(client=profile, series_meta=series_meta)
    result = get_forecaster("seasonal_naive").predict(history, 2, ctx)

    assert not [c for c in result.columns if c.startswith("q")]
