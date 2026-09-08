from __future__ import annotations

import polars as pl

from engine.features.segments import assign_volume_segments, segment_summary


def _profiles(volumes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": [f"S{i}" for i in range(len(volumes))],
            "total_volume": volumes,
        }
    )


def test_head_holds_the_dominant_volume():
    """A few large series carry most of the demand; the rest are the long tail."""
    profiles = _profiles([800.0, 150.0, 25.0, 15.0, 10.0])
    result = assign_volume_segments(profiles, head_share=0.80, mid_share=0.95)
    segments = dict(zip(result["series_id"], result["volume_segment"], strict=True))

    assert segments["S0"] == "Head"  # 80% of volume on its own
    assert segments["S1"] == "Mid"  # crosses the head boundary, so falls to Mid
    assert segments["S4"] == "Tail"


def test_every_series_is_assigned_exactly_one_segment():
    profiles = _profiles([100.0] * 10)
    result = assign_volume_segments(profiles)

    assert result.height == 10
    assert result["volume_segment"].is_null().sum() == 0
    assert set(result["volume_segment"].unique()) <= {"Head", "Mid", "Tail"}


def test_zero_volume_panel_does_not_invent_a_head():
    result = assign_volume_segments(_profiles([0.0, 0.0]))
    assert result["volume_segment"].to_list() == ["Tail", "Tail"]


def test_summary_reports_share_of_volume():
    profiles = assign_volume_segments(_profiles([900.0, 90.0, 10.0]))
    summary = segment_summary(profiles)

    total_pct = summary["volume_pct"].sum()
    assert abs(total_pct - 100.0) < 0.1
    assert summary["series"].sum() == 3
