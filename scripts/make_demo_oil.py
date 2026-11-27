"""Generate the Haveli Harvest Oils sample dataset (fictional edible-oil packer, North India).

Monthly sales from Nov 2023 to Oct 2026 per customer and item. What moves them:

  * REAL festival dates (Diwali, Dussehra, Holi) from the `holidays` package. Shoppers and
    sweet shops buy in the weeks before each festival, so an early Diwali pulls demand into
    the previous month.
  * REAL New Delhi monthly temperatures (data/demo/haveliharvest/weather_delhi_monthly.csv,
    Open-Meteo historical archive). Colder winters lift mustard oil, which North Indian
    households use more in winter. Very hot summers trim sweet-shop and caterer tins.
  * India's REAL edible-oil import duty changes. Crude palm, soybean and sunflower oil duty
    went from 0% to 20% on 14 Sep 2024 and back down to 10% on 31 May 2025. Imported oils got
    dearer, buyers bought less of them and moved part of their buying to mustard oil, then
    recovered after the cut. Distributors also bought ahead in September 2024.
  * Synthetic: growth per customer, the wedding-season pattern of caterer tins, and noise.

All company, brand and customer names are fictional. Writes three files to
data/demo/haveliharvest/: sales.csv (the engine's canonical columns, read by `csv_panel`),
item_master.csv and customer_master.csv.

    python scripts/make_demo_oil.py
    engine ingest --client demo_haveliharvest --file data/demo/haveliharvest/sales.csv
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import holidays
import polars as pl

FOLDER = Path(__file__).resolve().parents[1] / "data" / "demo" / "haveliharvest"
FIRST, MONTHS = date(2023, 11, 1), 36

# item_code: (description, item_group, uom, units per pack, size, size unit, MRP per unit,
#             distributor price per unit, base packs per month for a 1.0-size customer)
ITEMS = {
    "HH-MUS-1LP": ("HH Kachi Ghani Mustard Oil 1 L pouch, carton of 12", "MUST", "CTN", 12, 1, "L", 185, 160, 1400),
    "HH-MUS-1LB": ("HH Kachi Ghani Mustard Oil 1 L bottle, carton of 12", "MUST", "CTN", 12, 1, "L", 199, 172, 600),
    "HH-MUS-5LJ": ("HH Kachi Ghani Mustard Oil 5 L jar, carton of 4", "MUST", "CTN", 4, 5, "L", 920, 800, 300),
    "HH-MUS-15T": ("HH Kachi Ghani Mustard Oil 15 kg tin", "MUST", "TIN", 1, 15, "kg", 2650, 2380, 380),
    "HH-SOY-1LP": ("HH Gold Refined Soybean Oil 1 L pouch, carton of 12", "SOYA", "CTN", 12, 1, "L", 150, 128, 1200),
    "HH-SOY-5LJ": ("HH Gold Refined Soybean Oil 5 L jar, carton of 4", "SOYA", "CTN", 4, 5, "L", 740, 640, 240),
    "HH-SOY-15T": ("HH Gold Refined Soybean Oil 15 kg tin", "SOYA", "TIN", 1, 15, "kg", 2250, 2020, 440),
    "HH-PAL-1LP": ("HH Rasoi Refined Palmolein 1 L pouch, carton of 12", "PALM", "CTN", 12, 1, "L", 132, 112, 500),
    "HH-PAL-15T": ("HH Rasoi Refined Palmolein 15 kg tin", "PALM", "TIN", 1, 15, "kg", 1990, 1790, 740),
    "HH-SUN-1LP": ("HH Sunlite Refined Sunflower Oil 1 L pouch, carton of 12", "SUNF", "CTN", 12, 1, "L", 165, 142, 380),
    "HH-SUN-5LJ": ("HH Sunlite Refined Sunflower Oil 5 L jar, carton of 4", "SUNF", "CTN", 4, 5, "L", 810, 700, 100),
    "HH-RBO-1LB": ("HH Heart Rice Bran Oil 1 L bottle, carton of 12", "RICE", "CTN", 12, 1, "L", 175, 150, 200),
    "HH-RBO-5LJ": ("HH Heart Rice Bran Oil 5 L jar, carton of 4", "RICE", "CTN", 4, 5, "L", 860, 745, 85),
    "HH-VAN-1KP": ("HH Vanaspati 1 kg pouch, carton of 12", "VANA", "CTN", 12, 1, "kg", 155, 133, 240),
    "HH-VAN-15T": ("HH Vanaspati 15 kg tin", "VANA", "TIN", 1, 15, "kg", 2240, 2010, 300),
}  # fmt: skip
RAW = {
    "MUST": ("Mustard seed, crushed in our own expeller unit", "Rajasthan, Haryana, Madhya Pradesh (domestic)", "Bought at harvest, Mar-May, for most of the year"),
    "SOYA": ("Crude soybean oil", "Argentina and Brazil (about 70%, via Kandla port); Madhya Pradesh (30%)", "Imported cargo booked 6-8 weeks ahead"),
    "PALM": ("Crude palm oil", "Indonesia and Malaysia (via Kandla port)", "Imported cargo booked 6-8 weeks ahead"),
    "SUNF": ("Crude sunflower oil", "Russia and Ukraine (Black Sea, via Kandla and Mumbai)", "Imported cargo booked 8-10 weeks ahead"),
    "RICE": ("Crude rice bran oil", "Rice mills in Punjab and Haryana (domestic)", "Bought monthly"),
    "VANA": ("Palm oil and palm stearin, hydrogenated in-house", "Indonesia and Malaysia (via Kandla port)", "Imported cargo booked 6-8 weeks ahead"),
}  # fmt: skip
PACKAGING = {"1LP": "Laminated pouch", "1LB": "PET bottle", "5LJ": "HDPE jar", "15T": "Tinplate tin", "1KP": "Laminated pouch"}  # fmt: skip
IMPORTED = {"SOYA", "PALM", "SUNF", "VANA"}
RETAIL = [code for code in ITEMS if not code.endswith("15T")]
TINS = [code for code in ITEMS if code.endswith("15T")]

# customer: (channel, city, state, size, yearly growth, items, since)
CUSTOMERS = {
    "SHEKHAWATI TRADERS": ("Distributor", "Jaipur", "Rajasthan", 1.0, 0.05, RETAIL + TINS, 2004),
    "YAMUNA FOOD DISTRIBUTORS": ("Distributor", "New Delhi", "Delhi NCR", 1.2, 0.06, RETAIL + TINS, 2009),
    "AWADH PROVISIONS": ("Distributor", "Lucknow", "Uttar Pradesh", 0.9, 0.07, RETAIL + TINS, 2013),
    "BAZAARKART SUPERMARKETS": ("Modern trade", "Gurugram", "Haryana / Delhi NCR", 0.45, 0.16, [c for c in RETAIL if not c.startswith("HH-VAN")], 2019),
    "HALWAI AND CATERERS DESK": ("Institutional (sweet shops, caterers, namkeen makers)", "Jaipur, Agra, Delhi", "Rajasthan / UP / Delhi", 1.6, 0.04, TINS, 2001),
}  # fmt: skip

# Typical month-of-year pattern before festivals, weather and prices.
SEASON = {
    "MUST": [1.22, 1.10, 0.98, 0.95, 0.86, 0.80, 0.82, 0.88, 0.95, 1.04, 1.14, 1.22],
    "SOFT": [1.00, 0.97, 1.00, 0.97, 0.95, 0.92, 0.92, 0.95, 0.98, 1.03, 1.03, 1.00],
}
# Caterer tins follow the wedding season (pause mid-Dec to mid-Jan and in the monsoon).
WEDDINGS = [0.92, 1.16, 1.06, 1.10, 1.12, 0.94, 0.80, 0.82, 0.90, 1.00, 1.16, 1.02]
# (festival name in `holidays`, days of buying before it, uplift retail, uplift tins)
FESTIVALS = [("Diwali", 21, 0.16, 0.34), ("Dussehra", 10, 0.04, 0.08), ("Holi", 12, 0.07, 0.18)]
# Retail price index of imported oils after the real duty changes (1.00 = before 14 Sep 2024).
PRICE = {
    "2024-09": 1.03, "2024-10": 1.10, "2024-11": 1.13, "2024-12": 1.15, "2025-01": 1.15,
    "2025-02": 1.14, "2025-03": 1.13, "2025-04": 1.12, "2025-05": 1.11, "2025-06": 1.06,
    "2025-07": 1.04,
}  # fmt: skip
PRICE_AFTER = ("2025-08", 1.03)


def months() -> list[date]:
    return [
        date(FIRST.year + (FIRST.month - 1 + i) // 12, (FIRST.month - 1 + i) % 12 + 1, 1)
        for i in range(MONTHS)
    ]


def load_weather() -> tuple[dict[str, float], dict[int, float]]:
    """Monthly mean temperature, and the 2013-2022 normal for each calendar month."""
    path = FOLDER / "weather_delhi_monthly.csv"
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ]
    temps = {row["month"]: float(row["mean_temp_c"]) for row in csv.DictReader(lines)}
    normal: dict[int, list[float]] = defaultdict(list)
    for month, value in temps.items():
        if 2013 <= int(month[:4]) <= 2022:
            normal[int(month[5:])].append(value)
    return temps, {m: sum(v) / len(v) for m, v in normal.items()}


def festival_shares() -> dict[tuple[str, date], float]:
    """Share of each festival's buying window that falls in each month."""
    calendar = holidays.India(years=range(FIRST.year - 1, FIRST.year + 5))
    shares: dict[tuple[str, date], float] = defaultdict(float)
    for name, window, _, _ in FESTIVALS:
        for day, label in calendar.items():
            if name.lower() not in label.lower():
                continue
            for back in range(1, window + 1):
                buying = day - timedelta(days=back)
                shares[name, buying.replace(day=1)] += 1 / window
    return shares


def price_index(month: date) -> float:
    key = f"{month:%Y-%m}"
    if key in PRICE:
        return PRICE[key]
    return PRICE_AFTER[1] if key >= PRICE_AFTER[0] else 1.0


def multiplier(code: str, month: date, temps, normal, shares) -> float:
    group, tin = ITEMS[code][1], code.endswith("15T")
    factor = SEASON["MUST" if group == "MUST" else "SOFT"][month.month - 1]
    if tin:
        factor *= WEDDINGS[month.month - 1]
    for name, _, retail, tins in FESTIVALS:
        factor *= 1 + (tins if tin else retail) * shares.get((name, month), 0.0)
    anomaly = temps.get(f"{month:%Y-%m}", normal[month.month]) - normal[month.month]
    if group == "MUST" and month.month in (11, 12, 1, 2):
        factor *= 1 - 0.035 * anomaly  # colder than normal → more mustard oil
    if tin and month.month in (4, 5, 6):
        factor *= 1 - 0.015 * max(0.0, anomaly)  # extreme heat → fewer sweets and fried snacks
    price = price_index(month)
    if group in IMPORTED:
        factor *= price ** (-0.8 if tin else -0.6)
        if f"{month:%Y-%m}" == "2024-09":
            factor *= 1.07  # distributors bought ahead of the price list revision
        if f"{month:%Y-%m}" == "2024-10":
            factor *= 0.97  # and ran that stock down
    elif group == "MUST":
        factor *= price**0.35  # some buyers switch to mustard when imported oils get dearer
    return factor


def main() -> None:
    rng = random.Random(1998)
    temps, normal = load_weather()
    shares = festival_shares()
    rows = []
    for customer, (_, _, _, size, growth, items, _) in CUSTOMERS.items():
        short = "".join(word[0] for word in customer.split()[:3])
        for code in items:
            description, group, uom, *_rest, base = ITEMS[code]
            for index, month in enumerate(months()):
                trend = (1 + growth) ** (index / 12)
                noise = math.exp(rng.gauss(0, 0.09 if code.endswith("15T") else 0.06))
                effect = multiplier(code, month, temps, normal, shares)
                quantity = base * size * trend * noise * effect
                rows.append(
                    {
                        "series_id": f"{short}-{code}",
                        "customer": customer,
                        "item_code": code,
                        "description": description,
                        "item_group": group,
                        "period": month,
                        "quantity": float(round(quantity)),
                        "uom": uom,
                        "source_sheet": "sales",
                    }
                )
    FOLDER.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows)
    frame.write_csv(FOLDER / "sales.csv")

    with open(FOLDER / "item_master.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["item_code", "description", "item_group", "sales_unit", "units_per_pack", "unit_size",
             "size_unit", "mrp_inr_per_unit", "distributor_price_inr_per_unit", "packaging",
             "main_raw_material", "raw_material_origin", "buying_pattern"]
        )  # fmt: skip
        for code, (description, group, uom, units, size, unit, mrp, dp, _) in ITEMS.items():
            writer.writerow(
                [code, description, group, uom, units, size, unit, mrp, dp,
                 PACKAGING[code.split("-")[-1]], *RAW[group]]
            )  # fmt: skip
    with open(FOLDER / "customer_master.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer", "channel", "city", "state", "customer_since", "items_bought"])
        for customer, (channel, city, state, _, _, items, since) in CUSTOMERS.items():
            writer.writerow([customer, channel, city, state, since, len(items)])

    last_year = frame.filter(pl.col("period") >= date(2025, 11, 1))
    litres = rupees = 0.0
    for code, quantity in last_year.group_by("item_code").agg(pl.col("quantity").sum()).iter_rows():
        _, _, _, units, size, _, _, dp, _ = ITEMS[code]
        litres += quantity * units * size
        rupees += quantity * units * dp
    print(
        f"Wrote {frame.height} rows for {frame['series_id'].n_unique()} series to {FOLDER}\n"
        f"Last 12 months: about {litres / 1000:,.0f} thousand litres/kg, "
        f"sales about INR {rupees / 1e7:,.0f} crore at distributor prices"
    )


if __name__ == "__main__":
    main()
