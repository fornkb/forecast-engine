"""Numbers for the manager pages: forecast totals, accuracy, review impact, attention items.

Read-only views over the client's store. Nothing here calls a model or writes.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from engine.reason import store as records
from engine.reason.adjust import active
from engine.reason.forecast import UNASSIGNED, add_months, assign_family
from engine.reason.outcome import Outcome, outcome
from engine.reason.report import finding_title
from engine.reason.schemas import CompanyProfile, ReasonRun
from engine.store.duckdb_store import EngineStore

DECISION_PREFIXES = ("accepted_by_", "edited_by_")


# --------------------------------------------------------------------- forecast


@dataclass
class MonthPoint:
    period: date
    forecast: float
    p10: float | None = None
    p90: float | None = None
    last_year: float | None = None
    low: float | None = None  # what-if band, when a review exists for this forecast
    high: float | None = None


@dataclass
class GroupRow:
    id: str
    name: str
    series: int
    total: float
    last_year: float | None
    history: list[tuple[date, float]]
    months: list[MonthPoint]

    @property
    def change_pct(self) -> float | None:
        if not self.last_year:
            return None
        return round(100 * (self.total / self.last_year - 1), 1)


@dataclass
class SeriesRow:
    series_id: str
    item_code: str
    description: str
    customer: str
    unit: str
    group: str
    model: str
    total: float
    last_year: float | None

    @property
    def change_pct(self) -> float | None:
        if not self.last_year:
            return None
        return round(100 * (self.total / self.last_year - 1), 1)


@dataclass
class ForecastOverview:
    run_id: int
    created_at: datetime | None
    cutoff: date
    data_until: date | None
    months: list[date]
    total: list[MonthPoint]
    history: list[tuple[date, float]]
    groups: list[GroupRow]
    top_series: list[SeriesRow]
    models: dict[str, int]
    series: int
    group_label: str

    @property
    def horizon_total(self) -> float:
        return sum(m.forecast for m in self.total)

    @property
    def last_year_total(self) -> float | None:
        values = [m.last_year for m in self.total]
        return None if any(v is None for v in values) else sum(values)

    @property
    def change_pct(self) -> float | None:
        ly = self.last_year_total
        return round(100 * (self.horizon_total / ly - 1), 1) if ly else None

    @property
    def stale(self) -> bool:
        """New sales data arrived after this forecast was made."""
        return self.data_until is not None and self.data_until > self.cutoff


def _quantile_map(frame, target: float) -> dict[tuple[str, date], float]:
    if frame.is_empty():
        return {}
    levels = frame["level"].unique().to_list()
    level = min(levels, key=lambda value: abs(value - target))
    if abs(level - target) > 0.1:
        return {}
    return {
        (row["series_id"], row["period"]): row["value"]
        for row in frame.filter(frame["level"] == level).iter_rows(named=True)
        if row["value"] is not None
    }


def forecast_overview(
    store: EngineStore, profile: CompanyProfile | None = None, top: int = 10
) -> ForecastOverview | None:
    if not store.path.exists():
        return None
    run_id = store.latest_run("forecast")
    if run_id is None:
        return None
    points = store.load_forecasts(run_id)
    if points.is_empty():
        return None
    info = store.run_info(run_id) or {}
    dim = {row["series_id"]: row for row in store.load_dim().iter_rows(named=True)}
    fact = store.load_fact()
    quantiles = store.load_forecast_quantiles(run_id)
    p10, p90 = _quantile_map(quantiles, 0.1), _quantile_map(quantiles, 0.9)

    def group_of(series: str) -> str:
        meta = dim.get(series, {})
        if profile is not None and profile.families:
            return assign_family(
                profile, series, meta.get("item_code") or "", meta.get("item_group") or ""
            )
        return meta.get("item_group") or "all"

    names = {f.id: f.name for f in profile.families} if profile else {}
    names[UNASSIGNED] = "Not in a family"
    months = sorted(set(points["period"].to_list()))
    cutoff = points["cutoff_period"].max()

    actual: dict[tuple[str, date], float] = defaultdict(float)
    total_actual: dict[date, float] = defaultdict(float)
    group_actual: dict[tuple[str, date], float] = defaultdict(float)
    groups_by_series = {}
    for series, period, quantity in fact.select("series_id", "period", "quantity").iter_rows():
        group = groups_by_series.setdefault(series, group_of(series))
        actual[series, period] += quantity
        total_actual[period] += quantity
        group_actual[group, period] += quantity
    data_until = max(total_actual) if total_actual else None

    by_month: dict[date, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, True])
    by_group: dict[str, dict] = {}
    per_series: dict[str, dict] = {}
    models: Counter = Counter()
    for row in points.iter_rows(named=True):
        series, period, value = row["series_id"], row["period"], float(row["forecast"] or 0.0)
        group = groups_by_series.setdefault(series, group_of(series))
        low, high = p10.get((series, period)), p90.get((series, period))
        cell = by_month[period]
        cell[0] += value
        if low is None or high is None:
            cell[3] = False
        else:
            cell[1] += low
            cell[2] += high
        entry = by_group.setdefault(group, {"series": set(), "months": defaultdict(float)})
        entry["series"].add(series)
        entry["months"][period] += value
        item = per_series.setdefault(series, {"total": 0.0, "model": row["model"]})
        item["total"] += value
    for item in per_series.values():
        models[item["model"]] += 1

    def last_year(period: date, source: dict, key=None) -> float | None:
        past = add_months(period, -12)
        value = source.get((key, past) if key is not None else past)
        return value

    total = [
        MonthPoint(
            period=m,
            forecast=round(by_month[m][0], 2),
            p10=round(by_month[m][1], 2) if by_month[m][3] else None,
            p90=round(by_month[m][2], 2) if by_month[m][3] else None,
            last_year=last_year(m, total_actual),
        )
        for m in months
    ]
    recent = [add_months(cutoff, -offset) for offset in range(23, -1, -1)]
    history = [(m, round(total_actual.get(m, 0.0), 2)) for m in recent if m in total_actual]

    order = [*names] if profile else sorted(by_group)
    groups = []
    for group in [g for g in order if g in by_group] + [g for g in by_group if g not in order]:
        entry = by_group[group]
        group_months = [
            MonthPoint(
                period=m,
                forecast=round(entry["months"][m], 2),
                last_year=last_year(m, group_actual, group),
            )
            for m in months
        ]
        ly = [m.last_year for m in group_months]
        groups.append(
            GroupRow(
                id=group,
                name=names.get(group, group),
                series=len(entry["series"]),
                total=round(sum(entry["months"].values()), 2),
                last_year=None if any(v is None for v in ly) else sum(ly),
                history=[(m, round(group_actual.get((group, m), 0.0), 2)) for m in recent[-12:]],
                months=group_months,
            )
        )

    ranked = sorted(per_series.items(), key=lambda kv: kv[1]["total"], reverse=True)[:top]
    top_series = []
    for series, item in ranked:
        meta = dim.get(series, {})
        ly = [actual.get((series, add_months(m, -12))) for m in months]
        top_series.append(
            SeriesRow(
                series_id=series,
                item_code=meta.get("item_code") or "",
                description=meta.get("description") or "",
                customer=meta.get("customer") or "",
                unit=meta.get("uom") or "",
                group=names.get(groups_by_series.get(series, ""), groups_by_series.get(series, "")),
                model=item["model"],
                total=round(item["total"], 2),
                last_year=None if any(v is None for v in ly) else sum(ly),
            )
        )
    return ForecastOverview(
        run_id=run_id,
        created_at=info.get("created_at"),
        cutoff=cutoff,
        data_until=data_until,
        months=months,
        total=total,
        history=history,
        groups=groups,
        top_series=top_series,
        models=dict(models.most_common()),
        series=len(per_series),
        group_label="family" if profile and profile.families else "item group",
    )


# --------------------------------------------------------------------- accuracy


@dataclass
class Accuracy:
    run_id: int
    created_at: datetime | None
    model: str  # "portfolio" when per-series arbitration was scored
    baseline: str
    wape: float
    baseline_wape: float | None
    bias_pct: float | None
    coverage: float | None
    nominal: float | None
    observations: int
    scope: str  # holdout | overall
    segments: list[tuple[str, float, float | None]] = field(default_factory=list)

    @property
    def accuracy_pct(self) -> float:
        return round(100 * (1 - self.wape), 1)

    @property
    def gain_pct(self) -> float | None:
        """How much lower the error is than the simple baseline's."""
        if not self.baseline_wape:
            return None
        return round(100 * (1 - self.wape / self.baseline_wape), 1)


def accuracy(store: EngineStore) -> Accuracy | None:
    if not store.path.exists():
        return None
    run_id = store.latest_run("backtest")
    if run_id is None:
        return None
    info = store.run_info(run_id) or {}
    try:
        config = json.loads(info.get("config") or "{}")
    except ValueError:
        config = {}
    baseline = config.get("baseline") or "seasonal_naive"
    with store.connect() as con:
        rows = con.execute(
            "SELECT scope_type, scope_value, model, metric, value, n_obs FROM backtest_metrics"
            " WHERE run_id = ? AND scope_type IN ('holdout', 'overall', 'holdout_segment')",
            [run_id],
        ).fetchall()
    metrics: dict[tuple[str, str, str, str], tuple[float, int]] = {
        (scope, value, model, metric): (number, n)
        for scope, value, model, metric, number, n in rows
    }
    scope = "holdout" if any(k[0] == "holdout" for k in metrics) else "overall"
    candidates = {k[2] for k in metrics if k[0] == scope and k[3] == "wape"}
    if not candidates:
        return None
    if "portfolio" in candidates:
        model = "portfolio"
    else:
        model = min(candidates, key=lambda m: metrics[scope, "all", m, "wape"][0])

    def get(metric: str, which: str = model, value: str = "all", where: str = scope):
        found = metrics.get((where, value, which, metric))
        return found[0] if found else None

    wape, n = metrics[scope, "all", model, "wape"]
    mae, bias = get("mae"), get("bias")
    mean_actual = mae / wape if mae and wape else None
    segments = sorted(
        {
            (k[1], v[0], get("wape", baseline, k[1], "holdout_segment"))
            for k, v in metrics.items()
            if k[0] == "holdout_segment" and k[2] == model and k[3] == "wape"
        },
        key=lambda s: ["Head", "Mid", "Tail"].index(s[0]) if s[0] in ("Head", "Mid", "Tail") else 9,
    )
    return Accuracy(
        run_id=run_id,
        created_at=info.get("created_at"),
        model=model,
        baseline=baseline,
        wape=wape,
        baseline_wape=get("wape", baseline),
        bias_pct=round(100 * bias / mean_actual, 1) if bias is not None and mean_actual else None,
        coverage=get("interval_coverage"),
        nominal=get("nominal_coverage"),
        observations=int(n or 0),
        scope=scope,
        segments=segments,
    )


# ----------------------------------------------------------------------- review


def decided(flags: list[str]) -> bool:
    return "dismissed" in flags or any(f.startswith(DECISION_PREFIXES) for f in flags)


@dataclass
class ReviewDigest:
    run: ReasonRun
    open: int
    decided: int
    demand: int
    alerts: int
    baseline: float
    low: float
    high: float
    headline: str
    family_names: dict[str, str]
    outcome: Outcome

    @property
    def low_pct(self) -> float:
        return round(100 * (self.low / self.baseline - 1), 1) if self.baseline else 0.0

    @property
    def high_pct(self) -> float:
        return round(100 * (self.high / self.baseline - 1), 1) if self.baseline else 0.0

    @property
    def moves(self) -> bool:
        return abs(self.high - self.baseline) > 0.5 or abs(self.low - self.baseline) > 0.5

    @property
    def cost(self) -> float | None:
        known = [c.cost for c in self.run.calls if c.cost is not None]
        return sum(known) if len(known) == len(self.run.calls) else None


def _months_text(months: list[date]) -> str:
    if not months:
        return ""
    if len(months) == 1:
        return f"{months[0]:%b %Y}"
    return f"{months[0]:%b}–{months[-1]:%b %Y}"


def review_digest(run: ReasonRun) -> ReviewDigest:
    names = {f.family_id: f.name for f in run.families}
    live = [f for f in run.findings if "dismissed" not in f.flags]
    demand = [f for f in live if f.channel == "demand" and active(f)]
    result = outcome(run)
    top = max(demand, key=lambda f: max(abs(f.low_pct), abs(f.high_pct)), default=None)
    headline = result.headline
    if top is not None:
        headline += (
            f". Largest: {finding_title(top)} could move "
            f"{names.get(top.family_id, top.family_id)} {top.low_pct:+g}% to {top.high_pct:+g}% "
            f"in {_months_text(top.months)}"
        )
    return ReviewDigest(
        run=run,
        open=sum(1 for f in run.findings if not decided(f.flags)),
        decided=sum(1 for f in run.findings if decided(f.flags)),
        demand=len(demand),
        alerts=sum(1 for f in live if f.channel != "demand"),
        baseline=sum(b.baseline for b in run.bands),
        low=sum(b.low for b in run.bands),
        high=sum(b.high for b in run.bands),
        headline=headline,
        family_names=names,
        outcome=result,
    )


def family_impact(run: ReasonRun) -> list[dict]:
    """Per family: forecast over the horizon, what-if range, and the drivers behind it."""
    drivers: dict[str, list[str]] = defaultdict(list)
    for f in run.findings:
        if "dismissed" not in f.flags and f.driver_id not in drivers[f.family_id]:
            drivers[f.family_id].append(f.driver_id)
    rows = []
    for family in run.families:
        base = sum(m.baseline for m in family.months)
        low = sum(m.low if m.low is not None else m.baseline for m in family.months)
        high = sum(m.high if m.high is not None else m.baseline for m in family.months)
        rows.append(
            {
                "id": family.family_id,
                "name": family.name,
                "series": family.series_count,
                "baseline": base,
                "low": low,
                "high": high,
                "low_pct": round(100 * (low / base - 1), 1) if base else 0.0,
                "high_pct": round(100 * (high / base - 1), 1) if base else 0.0,
                "drivers": drivers.get(family.family_id, []),
            }
        )
    return rows


# --------------------------------------------------------------------- spending


def month_spend(store: EngineStore, month: date | None = None) -> tuple[float, int]:
    """Known AI cost and call count in one calendar month (UTC), default this month."""
    month = month or datetime.now(UTC).date().replace(day=1)
    cost, calls = 0.0, 0
    for row in records.call_ledger(store):
        started = row["started_at"]
        if started is not None and (started.year, started.month) == (month.year, month.month):
            calls += 1
            cost += row["cost"] or 0.0
    return round(cost, 4), calls


# -------------------------------------------------------------------- attention


@dataclass
class Attention:
    level: str  # bad | warn | info
    text: str
    link: str = ""


def attention(status, overview: ForecastOverview | None, digest: ReviewDigest | None) -> list:
    """What a manager should look at for one client, most serious first."""
    base = f"/c/{status.client_id}"
    items: list[Attention] = []
    if status.error:
        items.append(Attention("bad", status.error))
        return items
    if not status.enabled:
        return items
    if status.granularity != "monthly":
        items.append(
            Attention("info", f"{status.granularity.title()} forecasts: reasoning needs monthly")
        )
    if overview is None:
        items.append(Attention("bad", "No saved forecast yet", f"{base}/forecast"))
    elif overview.stale:
        items.append(
            Attention(
                "warn",
                f"Sales data to {overview.data_until:%b %Y} is newer than the forecast",
                f"{base}/forecast",
            )
        )
    if status.profile_status in ("missing", "invalid"):
        items.append(
            Attention("bad", f"Company profile {status.profile_status}", f"{base}/profile")
        )
    elif status.profile_status == "draft":
        items.append(Attention("warn", "Company profile is still a draft", f"{base}/profile"))
    if status.driver_problems:
        items.append(
            Attention(
                "warn", f"{len(status.driver_problems)} driver(s) need setup", f"{base}/profile"
            )
        )
    if status.provider not in ("replay", "") and not status.key_set:
        items.append(
            Attention(
                "warn", f"{status.key_env} not set: live reviews will fail", f"{base}/settings"
            )
        )
    if overview is not None and status.ready:
        if digest is None:
            items.append(Attention("warn", "Forecast not reviewed yet", f"{base}/reviews"))
        elif digest.run.forecast_run_id != overview.run_id:
            items.append(Attention("warn", "Latest forecast not reviewed yet", f"{base}/reviews"))
    if digest is not None:
        if digest.open:
            items.append(
                Attention(
                    "info",
                    f"{digest.open} finding{'s' if digest.open != 1 else ''} "
                    "waiting for a decision",
                    f"{base}/runs/{digest.run.reason_id}",
                )
            )
        if not digest.outcome.complete and digest.outcome.kind != "estimate":
            missing = digest.outcome.missing
            items.append(
                Attention(
                    "warn",
                    f"Last review incomplete: {missing} source{'s' if missing != 1 else ''} "
                    "could not be checked",
                    f"{base}/runs/{digest.run.reason_id}",
                )
            )
    order = {"bad": 0, "warn": 1, "info": 2}
    return sorted(items, key=lambda a: order[a.level])
