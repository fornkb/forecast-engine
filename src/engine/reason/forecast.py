"""Pin one saved monthly forecast and roll it up into product families.

The model reasons per family (a handful) instead of per SKU (hundreds), which is
the largest single saving in API cost. Per-SKU numbers come back in adjust.py.
Nothing here retrains or changes the stored forecast.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from engine.config.schema import ClientProfile
from engine.reason.schemas import (
    CompanyProfile,
    FamilyMonth,
    FamilySummary,
    ReasoningError,
    digest,
)
from engine.store.duckdb_store import EngineStore

UNASSIGNED = "unassigned"


def add_months(period: date, offset: int) -> date:
    year, month = divmod(period.year * 12 + period.month - 1 + offset, 12)
    return date(year, month + 1, 1)


@dataclass(frozen=True)
class SeriesRow:
    series_id: str
    family_id: str
    period: date
    horizon: int
    model: str
    baseline: float
    item_code: str = ""
    customer: str = ""
    unit: str = ""


@dataclass(frozen=True)
class ForecastView:
    client_id: str
    run_id: int
    cutoff_period: date
    months: list[date]
    rows: list[SeriesRow]
    families: list[FamilySummary]
    unassigned: list[str]
    digest: str
    # Monthly actuals per family over the whole stored history (for measured effects).
    history: dict[str, dict[date, float]] = field(default_factory=dict)

    def family(self, family_id: str) -> FamilySummary | None:
        return next((item for item in self.families if item.family_id == family_id), None)


def assign_family(profile: CompanyProfile, series_id: str, item_code: str, item_group: str) -> str:
    """First family whose match rule fits the SKU; unmatched SKUs are reported, not dropped."""
    for family in profile.families:
        rule = family.match
        if (
            series_id in rule.series
            or (item_group and item_group in rule.item_group)
            or any(item_code.startswith(prefix) for prefix in rule.item_prefix if item_code)
        ):
            return family.id
    return UNASSIGNED


def latest_forecast_run(store: EngineStore) -> int:
    run_id = store.latest_run("forecast") if store.path.exists() else None
    if run_id is None:
        raise ReasoningError("No saved forecast. Run `engine forecast --client ...` first.")
    return run_id


def _quantile(quantiles: pl.DataFrame, target: float) -> dict[tuple[str, date], float]:
    if quantiles.is_empty():
        return {}
    levels = quantiles["level"].unique().to_list()
    level = min(levels, key=lambda value: abs(value - target))
    if abs(level - target) > 0.1:
        return {}
    chosen = quantiles.filter(pl.col("level") == level)
    return {
        (row["series_id"], row["period"]): row["value"]
        for row in chosen.iter_rows(named=True)
        if row["value"] is not None
    }


def load_forecast(
    client: ClientProfile, store: EngineStore, profile: CompanyProfile, run_id: int
) -> ForecastView:
    if client.calendar.granularity != "monthly":
        raise ReasoningError(
            f"Reasoning works on monthly forecasts; {client.client_id} is "
            f"{client.calendar.granularity}."
        )
    info = store.run_info(run_id) if store.path.exists() else None
    if info is None or info.get("client_id") != client.client_id:
        raise ReasoningError(f"Forecast run {run_id} was not found for {client.client_id}.")
    if info.get("run_type") != "forecast":
        raise ReasoningError(f"Run {run_id} is a {info.get('run_type')} run, not a forecast.")
    points = store.load_forecasts(run_id)
    if points.is_empty():
        raise ReasoningError(f"Forecast run {run_id} has no stored rows.")
    if any(period.day != 1 for period in points["period"].unique().to_list()):
        raise ReasoningError("Forecast periods must be month starts.")

    dim = store.load_dim()
    meta = {
        row["series_id"]: row
        for row in dim.select("series_id", "item_code", "item_group", "customer", "uom").iter_rows(
            named=True
        )
    }
    family_of = {
        series: assign_family(
            profile, series, (values["item_code"] or ""), (values["item_group"] or "")
        )
        for series, values in meta.items()
    }
    p10 = _quantile(store.load_forecast_quantiles(run_id), 0.1)
    p90 = _quantile(store.load_forecast_quantiles(run_id), 0.9)

    rows = []
    for record in points.iter_rows(named=True):
        series = record["series_id"]
        values = meta.get(series, {})
        rows.append(
            SeriesRow(
                series_id=series,
                family_id=family_of.get(series, UNASSIGNED),
                period=record["period"],
                horizon=int(record["horizon"]),
                model=record["model"],
                baseline=float(record["forecast"] or 0.0),
                item_code=values.get("item_code") or "",
                customer=values.get("customer") or "",
                unit=values.get("uom") or "",
            )
        )
    months = sorted({row.period for row in rows})
    cutoff = points["cutoff_period"].max()

    history: dict[str, dict[date, float]] = defaultdict(lambda: defaultdict(float))
    fact = store.load_fact()
    for record in fact.select("series_id", "period", "quantity").iter_rows(named=True):
        history[family_of.get(record["series_id"], UNASSIGNED)][record["period"]] += float(
            record["quantity"]
        )
    history = {key: dict(value) for key, value in history.items()}

    families = []
    names = {family.id: family.name for family in profile.families}
    for family_id in [*names, UNASSIGNED]:
        members = [row for row in rows if row.family_id == family_id]
        if not members:
            continue
        past = history.get(family_id, {})
        by_month = []
        for month in months:
            current = [row for row in members if row.period == month]
            low = [p10.get((row.series_id, month)) for row in current]
            high = [p90.get((row.series_id, month)) for row in current]
            by_month.append(
                FamilyMonth(
                    period=month,
                    baseline=round(sum(row.baseline for row in current), 2),
                    p10=round(sum(low), 2) if low and None not in low else None,
                    p90=round(sum(high), 2) if high and None not in high else None,
                    last_year=past.get(add_months(month, -12)),
                )
            )
        recent = [(add_months(cutoff, -offset), 0.0) for offset in range(11, -1, -1)]
        families.append(
            FamilySummary(
                family_id=family_id,
                name=names.get(family_id, "Not assigned to a family"),
                series_count=len({row.series_id for row in members}),
                months=by_month,
                history=[(month, round(past.get(month, value), 2)) for month, value in recent],
            )
        )
    unassigned = sorted({row.series_id for row in rows if row.family_id == UNASSIGNED})
    return ForecastView(
        client_id=client.client_id,
        run_id=run_id,
        cutoff_period=cutoff,
        months=months,
        rows=rows,
        families=families,
        unassigned=unassigned,
        digest=digest([run_id, [(r.series_id, r.period, r.baseline, r.family_id) for r in rows]]),
        history=history,
    )


def analog_uplift(
    history: dict[date, float], analog_months: list[date], exclude: set[date] | None = None
) -> tuple[float, int] | None:
    """How much higher the given past months ran than the same months in other years.

    Each year is normalized by its own monthly mean first, so growth or decline
    across years does not count as an effect. The same calendar months are left out
    of every year's mean, so the effect being measured neither dilutes its own
    baseline nor makes one year's baseline differ in season mix from another's.
    Returns (uplift %, months used).
    """
    exclude = (exclude or set()) | set(analog_months)
    calendar_months = {month.month for month in exclude}
    by_year: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    for month, quantity in history.items():
        counts[month.year] += 1
        if month.month not in calendar_months:
            by_year[month.year].append(quantity)
    means = {
        year: sum(v) / len(v)
        for year, v in by_year.items()
        if counts[year] >= 10 and v and sum(v) > 0
    }

    def index(month: date) -> float | None:
        if month.year not in means or month not in history:
            return None
        return history[month] / means[month.year]

    uplifts = []
    for month in analog_months:
        own = index(month)
        others = [
            value
            for year in means
            if year != month.year
            and date(year, month.month, 1) not in exclude
            and (value := index(date(year, month.month, 1))) is not None
        ]
        if own is None or not others or sum(others) <= 0:
            continue
        uplifts.append(own / (sum(others) / len(others)) - 1)
    if not uplifts:
        return None
    return round(100 * sum(uplifts) / len(uplifts), 1), len(uplifts)
