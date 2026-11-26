"""Collect signals for every driver and decide what changed since the last run.

No model calls happen here. A driver whose signals did not change keeps last
month's finding at zero cost, which is the second largest saving after
reasoning per family:
  * news     changed when an article URL appears that no earlier run has seen
  * enso     changed when the NOAA alert status changes
  * weather  changed when a month's temperature or rain anomaly moves a bucket
  * calendar changed when festival months in the horizon change
  * measured changed when a measured effect moves by a full percent
"""

from __future__ import annotations

from datetime import date

from engine.reason import store as records
from engine.reason.connectors import CONNECTORS, NETWORK, Context, ProviderError
from engine.reason.connectors.scenario import ScenarioPack
from engine.reason.forecast import analog_uplift
from engine.reason.profile import ResolvedDriver
from engine.reason.schemas import DriverState, ReasonRun, Signal, digest
from engine.store.duckdb_store import EngineStore

MAX_SEEN_URLS = 500


def _rain_bucket(entry: dict) -> str:
    if entry.get("rain_mm") is None or not entry.get("rain_normal_mm"):
        return "?"
    ratio = entry["rain_mm"] / entry["rain_normal_mm"]
    return next(
        label
        for limit, label in (
            (0.5, "vdry"),
            (0.8, "dry"),
            (1.2, "norm"),
            (1.5, "wet"),
            (99e9, "vwet"),
        )
        if ratio < limit
    )


def fingerprint(signals: list[Signal]) -> str:
    """What counts as "changed" for everything except news (news uses seen URLs)."""
    parts = []
    for signal in signals:
        if not signal.ok:
            parts.append(("error", signal.error))
        elif signal.kind == "enso":
            parts.append(("enso", str(signal.data.get("status", "")).lower()))
        elif signal.kind == "weather":
            parts.append(
                (
                    "weather",
                    [
                        (m["period"], round(2 * m.get("temp_anomaly_c", 0)) / 2, _rain_bucket(m))
                        for m in signal.data.get("months", [])
                        if m.get("coverage") != "none"
                    ],
                )
            )
        elif signal.kind == "calendar":
            parts.append(("calendar", signal.data.get("in_horizon")))
        elif signal.kind == "measured":
            parts.append(
                ("measured", signal.data.get("family_id"), round(signal.data.get("uplift_pct", 0)))
            )
    return digest(sorted(parts, key=str))


def measured_from_params(driver: ResolvedDriver, ctx: Context) -> list[Signal]:
    """`analog_periods: [2024-03, 2024-04]` on any driver = "measure what happened last time"."""
    periods = driver.params.get("analog_periods")
    if not periods:
        return []
    months = [
        date.fromisoformat(f"{value}-01")
        for value in ([periods] if isinstance(periods, str) else periods)
    ]
    label = driver.params.get("analog_label", "the analog months")
    signals = []
    for family_id in driver.families:
        measured = analog_uplift(ctx.history.get(family_id, {}), months)
        if measured is None:
            continue
        uplift, used = measured
        signals.append(
            Signal(
                driver_id=driver.id,
                kind="measured",
                summary=(
                    f"Measured in this company's sales: {family_id} ran {uplift:+.1f}% in "
                    f"{label} ({', '.join(m.strftime('%Y-%m') for m in months)}) vs the same "
                    f"months in other years, trend removed ({used} months)"
                ),
                data={"family_id": family_id, "uplift_pct": uplift, "months_used": used},
                source_ref="client sales history",
            )
        )
    return signals


def _fetch(
    name: str, driver: ResolvedDriver, ctx: Context, store: EngineStore, cache_hours: int
) -> tuple[list[Signal], bool]:
    key = digest(
        [
            name,
            driver.id,
            driver.query,
            driver.params,
            ctx.months,
            ctx.as_of.date(),
            ctx.timezone,
            ctx.sources.model_dump(mode="json"),
        ]
    )
    cached = records.cache_get(store, key, cache_hours) if name in NETWORK else None
    if cached is not None:
        return cached, True
    signals = CONNECTORS[name](driver, ctx)
    if name in NETWORK:
        records.cache_put(store, key, signals)
    return signals, False


def collect(
    drivers: list[ResolvedDriver],
    ctx: Context,
    store: EngineStore,
    previous: ReasonRun | None,
    *,
    scenario: ScenarioPack | None = None,
    cache_hours: int = 12,
) -> tuple[list[DriverState], dict]:
    """Return one DriverState per driver plus counts of live fetches and cache hits."""
    before = {state.driver_id: state for state in (previous.drivers if previous else [])}
    states, stats = [], {"fetched": 0, "cached": 0, "scenario": 0, "errors": 0}
    for driver in drivers:
        signals: list[Signal] = []
        if driver.error:
            signals.append(
                Signal(
                    driver_id=driver.id,
                    kind="news",
                    summary="driver not usable",
                    ok=False,
                    error=driver.error,
                )
            )
        else:
            for name in driver.template.connectors:
                if scenario is not None and name in NETWORK:
                    signals += scenario.signals(driver.id, name)
                    stats["scenario"] += 1
                    continue
                try:
                    found, hit = _fetch(name, driver, ctx, store, cache_hours)
                    signals += found
                    stats["cached" if hit else "fetched"] += 1
                except (ProviderError, ValueError, KeyError) as exc:
                    stats["errors"] += 1
                    signals.append(
                        Signal(
                            driver_id=driver.id,
                            kind=name,
                            summary=f"{name} failed",
                            ok=False,
                            error=str(exc)[:200],
                        )
                    )
            signals += measured_from_params(driver, ctx)
        old = before.get(driver.id)
        seen_before = set(old.seen_urls) if old else set()
        urls = [a.url for s in signals if s.ok for a in s.articles]
        new_urls = [url for url in urls if url not in seen_before]
        mark = fingerprint(signals)
        usable = [s for s in signals if s.ok and (s.kind != "news" or s.articles)]
        if not any(s.ok for s in signals):
            status = "error"
        elif not usable:
            status = "no_data"
        elif old is None or new_urls or mark != old.fingerprint:
            status = "changed"
        else:
            status = "unchanged"
        seen = list(dict.fromkeys([*urls, *(old.seen_urls if old else [])]))[:MAX_SEEN_URLS]
        states.append(
            DriverState(
                driver_id=driver.id,
                status=status,
                fingerprint=mark,
                signals=signals,
                seen_urls=seen,
            )
        )
    return states, stats


def new_articles(state: DriverState, previous: ReasonRun | None):
    """Articles a model has not seen in an earlier run (old ones were already reasoned about)."""
    old = next(
        (s for s in (previous.drivers if previous else []) if s.driver_id == state.driver_id), None
    )
    seen = set(old.seen_urls) if old else set()
    return [a for s in state.signals if s.ok for a in s.articles if a.url not in seen]


__all__ = ["collect", "fingerprint", "new_articles"]
