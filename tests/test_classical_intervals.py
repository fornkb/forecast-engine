from __future__ import annotations

import pytest

from engine.forecast.classical import interval_levels


def test_a_mirror_pair_becomes_one_requested_interval():
    """statsforecast intervals are symmetric: one level of 80 yields both tails."""
    levels, columns = interval_levels([0.1, 0.5, 0.9])
    assert levels == [80]
    assert columns == {0.1: "lo-80", 0.9: "hi-80"}


def test_the_median_is_not_requested_as_an_interval():
    """It is the point forecast; a band around it would only restate it."""
    _, columns = interval_levels([0.5])
    assert columns == {}


def test_several_pairs_become_several_intervals():
    levels, columns = interval_levels([0.1, 0.25, 0.75, 0.9])
    assert levels == [50, 80]
    assert columns == {0.1: "lo-80", 0.25: "lo-50", 0.75: "hi-50", 0.9: "hi-80"}


def test_an_unmirrored_level_is_left_out_rather_than_guessed():
    """A symmetric interval cannot deliver a 95th percentile without a 5th, and
    inventing one would report a bound no model produced."""
    levels, columns = interval_levels([0.5, 0.95])
    assert levels == []
    assert columns == {}


def test_narrower_pairs_map_to_smaller_levels():
    levels, _ = interval_levels([0.05, 0.95])
    assert levels == [90]


@pytest.mark.parametrize("level", [0.1, 0.9])
def test_each_side_of_a_pair_reads_its_own_bound(level):
    _, columns = interval_levels([0.1, 0.9])
    assert columns[level].startswith("lo-" if level < 0.5 else "hi-")
