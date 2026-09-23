"""Load a demand-history workbook into the ERP-shaped PostgreSQL source.

This stands in for whatever populates a real client's ERP. It writes the base
table only — the view the engine reads is defined in scripts/sql/001_erp_source.sql
and is never written to directly.

    set ENGINE_PG_ADMIN_DSN=postgresql://postgres:<password>@localhost:5432/forecast_erp
    python scripts/load_to_postgres.py --file data/raw/Demand_History_3_Years.xlsx

The admin DSN is separate from ENGINE_PG_DSN on purpose: loading needs INSERT,
the engine needs SELECT, and they should not be the same credential.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

TABLE = "erp.sales_history"

# Must match NUMERIC(18,3) in scripts/sql/001_erp_source.sql. ADBC matches Arrow
# types to column types exactly; a mismatch fails the load.
QUANTITY_PRECISION = 18
QUANTITY_SCALE = 3

# Workbook column -> base-table column. The engine never sees these names; it
# reads the view, whose column names are the contract.
COLUMNS = {
    "Series ID": "series_id",
    "Customer": "customer_code",
    "Item Code": "item_code",
    "Description": "item_desc",
    "Item Group": "item_group",
    "Source Sheet": "source_sheet",
    "UOM": "uom",
    "Demand Month": "posting_date",
    "Demand Quantity": "quantity",
    "Demand Pattern": "demand_pattern",
}


def read_workbook(path: Path) -> pl.DataFrame:
    frame = pl.read_excel(path)
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns {missing}; found {frame.columns}")
    return (
        frame.select(list(COLUMNS))
        .rename(COLUMNS)
        .with_columns(
            pl.col("posting_date").cast(pl.Date),
            # Decimal, not Float64, to match NUMERIC(18,3) in the table. ADBC's
            # bulk-COPY writer maps Arrow decimal to PostgreSQL numeric but has
            # no double-to-numeric conversion at all, so a float column fails
            # the whole load rather than rounding.
            pl.col("quantity").cast(pl.Decimal(QUANTITY_PRECISION, QUANTITY_SCALE)),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True, help="Demand-history workbook.")
    parser.add_argument(
        "--dsn-env",
        default="ENGINE_PG_ADMIN_DSN",
        help="Environment variable holding the write-capable DSN.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Empty the table first. Without it, rows are appended.",
    )
    args = parser.parse_args()

    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        raise SystemExit(f"Environment variable {args.dsn_env} is not set.")

    frame = read_workbook(args.file)
    print(f"read {frame.height:,} rows from {args.file.name}")

    if args.truncate:
        import adbc_driver_postgresql.dbapi as pg

        with pg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f"TRUNCATE {TABLE}")
            conn.commit()
        print(f"truncated {TABLE}")

    # ADBC bulk ingest: one Arrow stream, not a row-per-INSERT loop.
    frame.write_database(TABLE, dsn, engine="adbc", if_table_exists="append")
    print(f"loaded {frame.height:,} rows into {TABLE}")

    quantity_sum = float(frame["quantity"].sum())
    print(
        f"   {frame['series_id'].n_unique()} series, "
        f"{frame['posting_date'].min()} to {frame['posting_date'].max()}, "
        f"quantity sum {quantity_sum:,.0f}"
    )
    print("   verify with: engine ingest --client teknic_mumbai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
