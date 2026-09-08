"""Calendar features and granularity-aware period arithmetic."""

from __future__ import annotations

import numpy as np
import polars as pl

from engine.config.schema import Granularity

_OFFSET_UNIT: dict[Granularity, str] = {"monthly": "mo", "weekly": "w", "daily": "d"}

_CYCLE_LENGTH: dict[Granularity, int] = {"monthly": 12, "weekly": 52, "daily": 7}


def offset_expr(column: str, steps: int, granularity: Granularity) -> pl.Expr:
    """Shift a date column forward by `steps` periods of the client's granularity."""
    return pl.col(column).dt.offset_by(f"{steps}{_OFFSET_UNIT[granularity]}")


def add_calendar_features(
    frame: pl.DataFrame, period_column: str, granularity: Granularity
) -> pl.DataFrame:
    """Seasonal position of the period being forecast.

    Encoded as sine/cosine of the position in the seasonal cycle so that December
    and January sit next to each other rather than at opposite ends of a scale.
    """
    cycle = _CYCLE_LENGTH[granularity]
    if granularity == "monthly":
        position = pl.col(period_column).dt.month()
    elif granularity == "weekly":
        position = pl.col(period_column).dt.week()
    else:
        position = pl.col(period_column).dt.weekday()

    angle = 2 * np.pi * (position.cast(pl.Float64) - 1) / cycle
    return frame.with_columns(
        position.cast(pl.Int32).alias("cal_position"),
        pl.col(period_column).dt.quarter().cast(pl.Int32).alias("cal_quarter"),
        pl.col(period_column).dt.year().cast(pl.Int32).alias("cal_year"),
        angle.sin().alias("cal_sin"),
        angle.cos().alias("cal_cos"),
    )


CALENDAR_FEATURES = ["cal_position", "cal_quarter", "cal_sin", "cal_cos"]
