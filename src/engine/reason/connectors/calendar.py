"""Festival and holiday dates (offline, from the `holidays` package or explicit dates).

Two things a seasonal forecast can get wrong: a festival that moves to a
different month than last year, and how big the festival effect really is. The
second is measured from the client's own history, so the model can cite a
number that came from the company's data rather than guess one.
"""

from __future__ import annotations

from datetime import date

from engine.reason.connectors.http import ProviderError
from engine.reason.forecast import analog_uplift
from engine.reason.schemas import Signal

MIN_FESTIVAL_MONTHS = 3


def festival_dates(name: str, country: str, years: range, explicit: list[str]) -> list[date]:
    if explicit:
        return sorted(date.fromisoformat(value) for value in explicit)
    try:
        import holidays
    except ImportError:
        raise ProviderError("the `holidays` package is not installed") from None
    if not country:
        raise ProviderError("festival driver needs a country (region country or `country` param)")
    found = holidays.country_holidays(country.upper(), years=list(years))
    needle = name.lower()
    return sorted(day for day, label in found.items() if needle in label.lower())


def fetch(driver, ctx) -> list[Signal]:
    params = driver.params
    name = str(params["festival"])
    explicit = params.get("dates") or []
    explicit = [explicit] if isinstance(explicit, str) else list(explicit)
    history_years = [m.year for series in ctx.history.values() for m in series]
    first = min([*history_years, ctx.months[0].year - 1])
    years = range(first, ctx.months[-1].year + 1)
    dates = festival_dates(name, str(params.get("country") or ""), years, explicit)
    if not dates:
        raise ProviderError(f"no dates found for festival '{name}'")
    by_year = {day.year: day for day in dates}
    lines, shifts = [], []
    for month in ctx.months:
        day = by_year.get(month.year)
        if day is None or day.month != month.month:
            continue
        before = by_year.get(month.year - 1)
        moved = before is not None and before.month != month.month
        shifts.append(
            {
                "month": month.strftime("%Y-%m"),
                "date": day.isoformat(),
                "last_year": before.isoformat() if before else None,
                "moved": moved,
            }
        )
        lines.append(
            f"{name} {day.isoformat()} falls in {month:%b %Y}"
            + (f"; last year it was {before.isoformat()} ({before:%b})" if before else "")
            + (" — moved month vs last year" if moved else "")
        )
    signals = [
        Signal(
            driver_id=driver.id,
            kind="calendar",
            summary="; ".join(lines) or f"{name} does not fall in the forecast months",
            data={"festival": name, "in_horizon": shifts},
            source_ref="holidays" if not explicit else "profile dates",
        )
    ]
    festival_months = [day.replace(day=1) for day in dates]
    for family_id in driver.families:
        history = ctx.history.get(family_id, {})
        past = [month for month in festival_months if month in history]
        measured = analog_uplift(history, past) if past else None
        if measured is None:
            continue
        uplift, used = measured
        if used < MIN_FESTIVAL_MONTHS:
            continue  # two festival months of history measure noise, not the festival
        signals.append(
            Signal(
                driver_id=driver.id,
                kind="measured",
                summary=(
                    f"Measured in this company's sales: {family_id} ran {uplift:+.1f}% in "
                    f"{name} months vs the same months without {name} ({used} months of history)"
                ),
                data={"family_id": family_id, "uplift_pct": uplift, "months_used": used},
                source_ref="client sales history",
            )
        )
    return signals
