"""Generate the synthetic AquaPure demo dataset (packaged drinking water, Maharashtra).

Monthly sales for 36 months, ending the month before the demo review date:
  * summer peak (Mar–May), monsoon dip (Jul–Sep), mild winter
  * ~8% yearly growth and month-to-month noise
  * a hotter-than-usual spring in 2024, after the 2023–24 El Niño, when bottled and
    flavoured water sold noticeably above their usual seasonal pattern

The reasoning layer can measure that 2024 spring effect from the data and cite it
when El Niño returns. All names are fictional. Output is the engine's canonical
CSV, read by the `csv_panel` reader:

    python scripts/make_demo_water.py
    engine ingest --client demo_aquapure --file data/demo/aquapure_sales.csv
"""

from __future__ import annotations

import math
import random
from datetime import date
from pathlib import Path

import polars as pl

OUT = Path(__file__).resolve().parents[1] / "data" / "demo" / "aquapure_sales.csv"
FIRST, MONTHS = date(2023, 11, 1), 36

SEASON = {
    1: 0.80,
    2: 0.90,
    3: 1.15,
    4: 1.35,
    5: 1.42,
    6: 1.05,
    7: 0.78,
    8: 0.74,
    9: 0.82,
    10: 1.00,
    11: 0.92,
    12: 0.85,
}
# item_code: (description, item_group, unit, base volume, seasonality strength, el_nino lift)
ITEMS = {
    "AP-500-PET": ("500 ml PET bottle, case of 24", "PETBTL", "CASE", 900, 1.0, 0.13),
    "AP-1L-PET": ("1 litre PET bottle, case of 12", "PETBTL", "CASE", 1400, 1.0, 0.12),
    "AP-2L-PET": ("2 litre PET bottle, case of 6", "PETBTL", "CASE", 500, 0.9, 0.10),
    "AP-20L-JAR": ("20 litre returnable jar", "JAR20", "JAR", 2600, 0.45, 0.05),
    "AP-FL-LEMON": ("Lemon flavoured water 500 ml, case of 24", "FLAV", "CASE", 300, 1.3, 0.16),
    "AP-FL-MINT": ("Mint flavoured water 500 ml, case of 24", "FLAV", "CASE", 220, 1.3, 0.15),
}
CUSTOMERS = {
    "SAHYADRI DISTRIBUTORS": 1.0,
    "KONKAN RETAIL MART": 0.7,
    "DECCAN HOTELS GROUP": 0.45,
    "PUNE OFFICE SUPPLIES": 0.55,
}
# Which customers buy which items (hotels skip jars, offices mostly buy jars).
BUYS = {
    "SAHYADRI DISTRIBUTORS": list(ITEMS),
    "KONKAN RETAIL MART": ["AP-500-PET", "AP-1L-PET", "AP-2L-PET", "AP-FL-LEMON", "AP-FL-MINT"],
    "DECCAN HOTELS GROUP": ["AP-500-PET", "AP-1L-PET", "AP-FL-LEMON"],
    "PUNE OFFICE SUPPLIES": ["AP-20L-JAR", "AP-1L-PET"],
}
EL_NINO_SPRING = {
    date(2024, 3, 1): 0.8,
    date(2024, 4, 1): 1.0,
    date(2024, 5, 1): 1.0,
    date(2024, 6, 1): 0.4,
}


def months() -> list[date]:
    return [
        date(FIRST.year + (FIRST.month - 1 + i) // 12, (FIRST.month - 1 + i) % 12 + 1, 1)
        for i in range(MONTHS)
    ]


def main() -> None:
    rng = random.Random(2027)
    rows = []
    for customer, scale in CUSTOMERS.items():
        for item in BUYS[customer]:
            description, group, unit, base, strength, lift = ITEMS[item]
            series = f"{customer.split()[0][:4]}-{item}"
            for index, month in enumerate(months()):
                season = 1 + strength * (SEASON[month.month] - 1)
                growth = 1.08 ** (index / 12)
                event = 1 + lift * EL_NINO_SPRING.get(month, 0.0)
                noise = math.exp(rng.gauss(0, 0.05))
                quantity = round(base * scale * season * growth * event * noise)
                rows.append(
                    {
                        "series_id": series,
                        "customer": customer,
                        "item_code": item,
                        "description": description,
                        "item_group": group,
                        "period": month,
                        "quantity": float(quantity),
                        "uom": unit,
                        "source_sheet": "sales",
                    }
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT)
    print(f"Wrote {len(rows)} rows for {len({r['series_id'] for r in rows})} series to {OUT}")


if __name__ == "__main__":
    main()
