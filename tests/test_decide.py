from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from pydantic import ValidationError

from engine.config.schema import ClientProfile, CostModelConfig
from engine.decide import (
    add_decision_value_add,
    aggregate,
    build_policy,
    coverage_gap,
    to_long,
    with_decision_costs,
)

COST = CostModelConfig(
    holding_cost_per_unit=1.0,
    shortage_cost_per_unit=9.0,
    lead_time_periods=1,
    review_period_periods=1,
)


def _scored(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    """One row per (model, stock level, actual)."""
    return pl.DataFrame(
        {
            "model": [r[0] for r in rows],
            "series_id": ["S1"] * len(rows),
            "period": [date(2025, 1, 1)] * len(rows),
            "horizon": [1] * len(rows),
            "value": [r[1] for r in rows],
            "actual": [r[2] for r in rows],
        }
    )


# --- the cost model -------------------------------------------------------


def test_critical_ratio_sets_the_service_level():
    """9:1 means a stockout hurts nine times as much, so stock to the 90th."""
    assert COST.service_level == pytest.approx(0.9)
    assert CostModelConfig(
        holding_cost_per_unit=1.0, shortage_cost_per_unit=1.0
    ).service_level == pytest.approx(0.5)


def test_protection_window_is_lead_time_plus_review():
    assert COST.protection_window == 2


def test_a_service_level_that_is_not_forecast_is_rejected():
    """Interpolating to an unforecast quantile would invent a distribution shape
    rather than report one the model produced."""
    with pytest.raises(ValidationError, match="probabilistic.levels"):
        ClientProfile(
            client_id="a",
            display_name="A",
            cost_model={"holding_cost_per_unit": 1.0, "shortage_cost_per_unit": 2.0},
        )


def test_cost_model_needs_quantiles_switched_on():
    with pytest.raises(ValidationError, match="probabilistic.enabled is false"):
        ClientProfile(
            client_id="a",
            display_name="A",
            probabilistic={"enabled": False, "levels": [0.9]},
            cost_model={},
        )


def test_absent_cost_model_means_no_decision_layer():
    assert ClientProfile(client_id="a", display_name="A").cost_model is None


# --- costing --------------------------------------------------------------


def test_overstocking_is_charged_holding_and_understocking_shortage():
    costed = with_decision_costs(_scored([("over", 110.0, 100.0), ("under", 90.0, 100.0)]), COST)
    by_model = {r["model"]: r for r in costed.iter_rows(named=True)}

    assert by_model["over"]["units_held"] == 10
    assert by_model["over"]["decision_cost"] == pytest.approx(10.0)
    assert by_model["under"]["units_short"] == 10
    # The same ten units, nine times the cost. That asymmetry is the whole point.
    assert by_model["under"]["decision_cost"] == pytest.approx(90.0)


def test_a_perfect_forecast_costs_nothing():
    """Which is what makes decision cost readable directly as regret."""
    costed = with_decision_costs(_scored([("perfect", 100.0, 100.0)]), COST)
    assert costed["decision_cost"].item() == pytest.approx(0.0)


def test_fill_rate_reports_the_service_level_actually_achieved():
    costed = with_decision_costs(_scored([("m", 90.0, 100.0)]), COST)
    summary = aggregate(costed, ["model"])
    assert summary["fill_rate"].item() == pytest.approx(0.9)


def test_zero_demand_leaves_fill_rate_undefined_rather_than_perfect():
    """Nothing was demanded, so nothing was served. Reporting 100% would flatter."""
    costed = with_decision_costs(_scored([("m", 5.0, 0.0)]), COST)
    assert aggregate(costed, ["model"])["fill_rate"].item() is None


# --- decision value add ---------------------------------------------------


def test_dva_is_measured_against_the_baseline():
    costed = with_decision_costs(_scored([("base", 90.0, 100.0), ("better", 91.0, 100.0)]), COST)
    summary = add_decision_value_add(aggregate(costed, ["model"]), "base")
    by_model = {r["model"]: r["dva_pct"] for r in summary.iter_rows(named=True)}

    assert by_model["base"] == pytest.approx(0.0)
    # Baseline is 10 short at 9 each = 90; better is 9 short = 81. One tenth saved.
    assert by_model["better"] == pytest.approx(10.0)


def test_dva_is_scoped_so_each_segment_uses_its_own_baseline():
    """A segmented summary holds one baseline row per segment; comparing Head
    against the baseline's total would be meaningless."""
    costed = with_decision_costs(
        _scored([("base", 90.0, 100.0), ("other", 95.0, 100.0)]), COST
    ).with_columns(pl.lit("Head").alias("volume_segment"))
    costed = pl.concat(
        [
            costed,
            costed.with_columns(pl.lit("Tail").alias("volume_segment"), pl.col("value") * 0.5),
        ]
    )
    summary = add_decision_value_add(
        aggregate(costed, ["model", "volume_segment"]), "base", "volume_segment"
    )
    baselines = summary.filter(pl.col("model") == "base")
    assert baselines.height == 2
    assert all(v == pytest.approx(0.0) for v in baselines["dva_pct"])


def test_a_missing_baseline_yields_no_dva_rather_than_an_error():
    summary = add_decision_value_add(
        aggregate(with_decision_costs(_scored([("m", 90.0, 100.0)]), COST), ["model"]), "absent"
    )
    assert summary["dva_pct"].item() is None


def test_to_long_emits_one_row_per_metric():
    summary = add_decision_value_add(
        aggregate(with_decision_costs(_scored([("m", 90.0, 100.0)]), COST), ["model"]), "m"
    )
    long = to_long(summary, "holdout", None)
    assert set(long.columns) == {"model", "scope_type", "scope_value", "metric", "value", "n_obs"}
    assert "decision_cost" in long["metric"].to_list()
    assert long["scope_value"].unique().to_list() == ["__all__"]


# --- policy ---------------------------------------------------------------


def _forward() -> tuple[pl.DataFrame, pl.DataFrame]:
    forecasts = pl.DataFrame(
        {
            "series_id": ["S1", "S1", "S1"],
            "model": ["m"] * 3,
            "horizon": [1, 2, 3],
            "forecast": [10.0, 10.0, 10.0],
        }
    )
    quantiles = pl.DataFrame(
        {
            "series_id": ["S1"] * 6,
            "model": ["m"] * 6,
            "horizon": [1, 2, 3, 1, 2, 3],
            "level": [0.9] * 3 + [0.5] * 3,
            "value": [14.0, 14.0, 14.0, 10.0, 10.0, 10.0],
        }
    )
    return forecasts, quantiles


def test_policy_covers_the_protection_window_not_the_whole_horizon():
    """Lead 1 + review 1 = 2 periods, so the third horizon is not stocked for."""
    policy = build_policy(*_forward(), COST)
    row = policy.row(0, named=True)

    assert row["window_periods"] == 2
    assert row["expected_demand"] == pytest.approx(20.0)
    assert row["target_stock"] == pytest.approx(28.0)
    assert row["safety_stock"] == pytest.approx(8.0)
    # Reorder point bridges lead time alone: one period at the 90th percentile.
    assert row["reorder_point"] == pytest.approx(14.0)


def test_policy_uses_the_level_the_cost_ratio_implies():
    """Not the median, which is what a point forecast would give."""
    policy = build_policy(*_forward(), COST)
    assert policy["service_level"].item() == pytest.approx(0.9)
    assert policy["target_stock"].item() > policy["expected_demand"].item()


def test_a_missing_service_level_is_refused_with_the_fix_named():
    forecasts, quantiles = _forward()
    with pytest.raises(ValueError, match="probabilistic.levels"):
        build_policy(forecasts, quantiles.filter(pl.col("level") == 0.5), COST)


def test_safety_stock_never_goes_negative():
    """A model whose upper quantile sits below its own point forecast is
    miscalibrated; a negative safety stock is not an instruction."""
    forecasts, quantiles = _forward()
    inverted = quantiles.with_columns(
        pl.when(pl.col("level") == 0.9).then(1.0).otherwise(pl.col("value")).alias("value")
    )
    assert build_policy(forecasts, inverted, COST)["safety_stock"].item() == 0.0


def test_zero_lead_time_means_no_reorder_point():
    """Instant replenishment leaves nothing to bridge."""
    instant = CostModelConfig(
        holding_cost_per_unit=1.0, shortage_cost_per_unit=9.0, lead_time_periods=0
    )
    assert build_policy(*_forward(), instant)["reorder_point"].item() == 0.0


def test_coverage_gap_names_the_models_that_block_a_decision():
    """A model without quantiles leaves its series off the reorder screen
    entirely, which must not pass silently."""
    forecasts, quantiles = _forward()
    forecasts = pl.concat(
        [
            forecasts,
            pl.DataFrame(
                {
                    "series_id": ["S2"] * 3,
                    "model": ["no_quantiles"] * 3,
                    "horizon": [1, 2, 3],
                    "forecast": [5.0] * 3,
                }
            ),
        ]
    )
    gap = coverage_gap(forecasts, build_policy(forecasts, quantiles, COST))
    assert gap.row(0, named=True) == {"model": "no_quantiles", "series": 1}
