"""Forecast export to files, for people and spreadsheets.

A file is a snapshot handed to a person. It is not how the forecast reaches an
ERP — that means writing into the ERP's own database, which is a separate path.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

LEADING_COLUMNS = ["series_id", "customer", "item_code", "description", "uom"]
FORECAST_COLUMNS = ["period", "horizon", "model", "forecast"]
SUPPORTED = (".csv", ".parquet")


def export_forecasts(forecasts: pl.DataFrame, series_meta: pl.DataFrame, path: Path) -> int:
    """Write every forecast row, labelled with what each item actually is.

    A bare series id means nothing to a planner, so customer, item code and
    description are joined on. Returns the number of rows written.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(
            f"Cannot export to {path.name!r}: use a {' or '.join(SUPPORTED)} file name."
        )

    attributes = [c for c in LEADING_COLUMNS if c in series_meta.columns]
    quantiles = sorted(
        (c for c in forecasts.columns if c.startswith("q") and c[1:2].isdigit()),
        key=lambda name: float(name[1:].replace("_", ".")),
    )

    frame = (
        forecasts.join(series_meta.select(attributes), on="series_id", how="left")
        .select(attributes + FORECAST_COLUMNS + quantiles)
        .sort("series_id", "period")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        # The byte-order mark is what makes Excel read the file as UTF-8, so
        # non-ASCII item descriptions survive being double-clicked open.
        frame.write_csv(path, include_bom=True)
    else:
        frame.write_parquet(path)
    return frame.height
