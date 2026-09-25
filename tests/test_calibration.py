from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from engine.forecast.calibration import (
    apply_offsets,
    apply_to_wide,
    coverage,
    fit_offsets,
)

LEVEL = 0.9


def _scored(actuals: list[float], quantile: float, origin: int = 1) -> pl.DataFrame:
    """One series, one model, `actuals` periods, all sharing one quantile value."""
    n = len(actuals)
    return pl.DataFrame(
        {
            "model": ["m"] * n,
            "origin": [origin] * n,
            "series_id": ["S1"] * n,
            "period": [date(2025, 1 + i, 1) for i in range(n)],
            "horizon": [1] * n,
            "level": [LEVEL] * n,
            "actual": actuals,
            "value": [quantile] * n,
        }
    )


def _level(mean_demand: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame({"series_id": ["S1"], "mean_demand": [mean_demand]})


# --- coverage -------------------------------------------------------------


def test_coverage_counts_periods_the_quantile_covered():
    scored = _scored([50.0, 60.0, 200.0, 70.0], quantile=100.0)
    assert coverage(scored, ["model"])["coverage"].item() == pytest.approx(0.75)


# --- fitting --------------------------------------------------------------


def test_a_quantile_already_at_the_right_place_gets_no_offset():
    """A quantile sitting on the empirical 90th percentile needs no correction."""
    scored = _scored([float(i) for i in range(1, 11)], quantile=9.0)
    assert fit_offsets(scored, _level())["scaled_offset"].item() == pytest.approx(0.0, abs=1e-9)


def test_an_over_wide_quantile_is_tightened_to_the_target():
    """Over-covering is not free: every unit above what the service level asks for
    is stock held for nothing. The correction pulls the quantile down to exactly
    the promised coverage, not merely into the right neighbourhood."""
    scored = _scored([50.0] * 9 + [150.0], quantile=500.0)
    fixed = apply_offsets(scored, fit_offsets(scored, _level()), _level())

    assert coverage(scored, ["model"])["coverage"].item() == pytest.approx(1.0)
    assert coverage(fixed, ["model"])["coverage"].item() == pytest.approx(0.9)
    assert fixed["value"].max() < 500.0


def test_a_narrow_quantile_gets_a_positive_offset():
    """Under-covering means the quantile sits too low and has to move up."""
    scored = _scored([50.0] * 5 + [150.0] * 5, quantile=100.0)
    assert fit_offsets(scored, _level())["scaled_offset"].item() > 0


def test_a_wide_quantile_gets_a_negative_offset():
    scored = _scored([50.0] * 10, quantile=500.0)
    assert fit_offsets(scored, _level())["scaled_offset"].item() < 0


def test_the_offset_restores_the_promised_coverage_on_its_own_data():
    """The defining property: fitted on a set, it makes coverage exact there."""
    scored = _scored([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], quantile=30.0)
    fixed = apply_offsets(scored, fit_offsets(scored, _level()), _level())
    assert coverage(fixed, ["model"])["coverage"].item() == pytest.approx(LEVEL, abs=0.05)


def test_the_holdout_origin_is_excluded_from_the_fit():
    """Fitting on the origin the correction is judged against would make the
    reported coverage a measure of its own fit."""
    calibration_rows = _scored([50.0] * 10, quantile=100.0, origin=1)
    holdout = _scored([9999.0] * 10, quantile=100.0, origin=2)
    both = pl.concat([calibration_rows, holdout])

    with_holdout = fit_offsets(both, _level())["scaled_offset"].item()
    without = fit_offsets(both, _level(), exclude_origin=2)["scaled_offset"].item()
    assert without != pytest.approx(with_holdout)
    # The wildly-uncovered holdout would have demanded a large upward shift.
    assert without < with_holdout


def test_offsets_are_scaled_by_the_series_level():
    """Demand here spans four orders of magnitude, so an offset in raw units would
    swamp the small series and vanish on the large ones."""
    scored = _scored([50.0] * 5 + [150.0] * 5, quantile=100.0)
    small = fit_offsets(scored, _level(10.0))["scaled_offset"].item()
    large = fit_offsets(scored, _level(1000.0))["scaled_offset"].item()
    assert small == pytest.approx(large * 100, rel=0.01)


def test_series_with_no_meaningful_level_are_left_out_of_the_fit():
    """Dividing by a near-zero mean would let one series dominate every offset."""
    scored = _scored([50.0] * 5 + [150.0] * 5, quantile=100.0)
    assert fit_offsets(scored, pl.DataFrame({"series_id": ["S1"], "mean_demand": [0.0]})).is_empty()


def test_no_calibration_data_yields_no_offsets_rather_than_an_error():
    scored = _scored([50.0], quantile=100.0, origin=1)
    assert fit_offsets(scored, _level(), exclude_origin=1).is_empty()


# --- applying -------------------------------------------------------------


def test_applying_no_offsets_changes_nothing():
    scored = _scored([50.0], quantile=100.0)
    assert apply_offsets(scored, pl.DataFrame(), _level()).equals(scored)


def test_corrected_quantiles_are_never_negative():
    """A large negative offset must not produce a stock level below zero."""
    scored = _scored([0.0] * 10, quantile=1.0)
    fixed = apply_offsets(scored, fit_offsets(scored, _level()), _level())
    assert fixed["value"].min() >= 0


def test_a_model_without_an_offset_is_left_untouched():
    """Shifting it by another model's correction would be worse than not shifting."""
    scored = pl.concat(
        [
            _scored([50.0], quantile=100.0),
            _scored([50.0], quantile=100.0).with_columns(pl.lit("other").alias("model")),
        ]
    )
    offsets = pl.DataFrame({"model": ["m"], "level": [LEVEL], "scaled_offset": [0.5]})
    fixed = apply_offsets(scored, offsets, _level())
    by_model = {r["model"]: r["value"] for r in fixed.iter_rows(named=True)}
    assert by_model["other"] == 100.0
    assert by_model["m"] == pytest.approx(150.0)


# --- wide form ------------------------------------------------------------


def test_wide_and_long_corrections_agree():
    """The forward forecast carries one column per level; it must not drift from
    the long form that evaluation uses."""
    offsets = pl.DataFrame({"model": ["m"], "level": [LEVEL], "scaled_offset": [0.25]})
    wide = pl.DataFrame({"series_id": ["S1"], "model": ["m"], "q90": [100.0]})

    corrected = apply_to_wide(wide, offsets, _level(), [LEVEL])
    long_form = apply_offsets(
        pl.DataFrame({"series_id": ["S1"], "model": ["m"], "level": [LEVEL], "value": [100.0]}),
        offsets,
        _level(),
    )
    assert corrected["q90"].item() == pytest.approx(long_form["value"].item())


def test_wide_correction_leaves_null_quantiles_null():
    """A model that emits no quantiles must not acquire one made of an offset."""
    offsets = pl.DataFrame({"model": ["m"], "level": [LEVEL], "scaled_offset": [0.25]})
    wide = pl.DataFrame({"series_id": ["S1"], "model": ["m"], "q90": [None]}, strict=False)
    assert apply_to_wide(wide, offsets, _level(), [LEVEL])["q90"].item() is None


def test_wide_correction_ignores_levels_it_has_no_column_for():
    offsets = pl.DataFrame(
        {"model": ["m", "m"], "level": [0.1, LEVEL], "scaled_offset": [0.1, 0.25]}
    )
    wide = pl.DataFrame({"series_id": ["S1"], "model": ["m"], "q90": [100.0]})
    out = apply_to_wide(wide, offsets, _level(), [0.1, LEVEL])
    assert out.columns == ["series_id", "model", "q90"]
