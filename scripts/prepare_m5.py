"""Materialise the M5 (Walmart) dataset into the engine's canonical long format.

Dataset acquisition is deliberately not the engine's job: this script turns a
research dataset into an ordinary panel file, which the engine then ingests
through the same reader any client file would use.

M5 is daily at item-store level (30,490 series x 1,969 days). It is aggregated
to weekly here because event and promotion effects survive weekly buckets but
are washed out by monthly ones, and because weekly is a common planning
granularity in its own right.

Exogenous columns (sell_price, SNAP flags, event calendars) are written to a
separate file rather than discarded — the context layer in a later phase is
their consumer.

    python scripts/prepare_m5.py --states CA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw")


def load_m5(directory: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    from datasetsforecast.m5 import M5

    target, exogenous, static = M5.load(str(directory))
    return (
        pl.from_pandas(target),
        pl.from_pandas(exogenous),
        pl.from_pandas(static),
    )


def build_panel(
    target: pl.DataFrame, static: pl.DataFrame, states: list[str] | None
) -> pl.DataFrame:
    """Aggregate to weekly and map onto the engine's canonical column names."""
    if states:
        keep = static.filter(pl.col("state_id").is_in(states))["unique_id"]
        target = target.filter(pl.col("unique_id").is_in(keep))
        static = static.filter(pl.col("unique_id").is_in(keep))

    weekly = (
        target.with_columns(pl.col("ds").dt.truncate("1w").alias("period"))
        .group_by("unique_id", "period")
        .agg(pl.col("y").sum().alias("quantity"))
    )

    # The final bucket is usually a partial week and would look like a demand
    # collapse to every model.
    last_complete = weekly["period"].max()
    weekly = weekly.filter(pl.col("period") < last_complete)

    return (
        weekly.join(static, on="unique_id", how="left")
        .select(
            pl.col("unique_id").alias("series_id"),
            pl.col("store_id").alias("customer"),
            pl.col("item_id").alias("item_code"),
            pl.col("item_id").alias("description"),
            pl.col("dept_id").alias("item_group"),
            pl.col("cat_id").alias("source_sheet"),
            pl.lit("EA").alias("uom"),
            "period",
            "quantity",
        )
        .sort("series_id", "period")
    )


def build_exogenous(exogenous: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Weekly exogenous drivers, kept for the context layer in a later phase."""
    keep = panel["series_id"].unique()
    return (
        exogenous.filter(pl.col("unique_id").is_in(keep))
        .with_columns(pl.col("ds").dt.truncate("1w").alias("period"))
        .group_by("unique_id", "period")
        .agg(
            pl.col("sell_price").mean().alias("mean_sell_price"),
            pl.col("snap_CA").max().alias("snap_ca"),
            pl.col("snap_TX").max().alias("snap_tx"),
            pl.col("snap_WI").max().alias("snap_wi"),
            pl.col("event_name_1").drop_nulls().first().alias("event_name"),
            pl.col("event_type_1").drop_nulls().first().alias("event_type"),
        )
        .rename({"unique_id": "series_id"})
        .sort("series_id", "period")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        default="CA",
        help="Comma-separated state codes to keep (CA, TX, WI), or 'all'.",
    )
    parser.add_argument("--out", default="m5_walmart", help="Output file stem.")
    args = parser.parse_args()

    states = None if args.states.lower() == "all" else [s.strip() for s in args.states.split(",")]

    print("Loading M5 (downloads on first run)...")
    target, exogenous, static = load_m5(RAW_DIR / "m5")

    panel = build_panel(target, static, states)
    panel_path = RAW_DIR / f"{args.out}.parquet"
    panel.write_parquet(panel_path)

    exog = build_exogenous(exogenous, panel)
    exog_path = RAW_DIR / f"{args.out}_exogenous.parquet"
    exog.write_parquet(exog_path)

    print(f"  panel      {panel.height:>10,} rows  {panel['series_id'].n_unique():>7,} series")
    print(f"             {panel['period'].min()} to {panel['period'].max()}")
    print(f"  written    {panel_path}")
    print(f"  exogenous  {exog.height:>10,} rows -> {exog_path}")


if __name__ == "__main__":
    main()
