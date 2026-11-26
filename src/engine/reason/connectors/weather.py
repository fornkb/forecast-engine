"""Monthly seasonal outlook vs a 10-year normal, from Open-Meteo (ECMWF SEAS5 ensemble mean).

Daily values in the client's timezone are rolled up to calendar months. The
normal comes from Open-Meteo's historical archive for the same point. The
anomaly is indicative only: regional grid, no bias correction, model vs
reanalysis. Months beyond the outlook are reported as "no outlook".
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date

from engine.reason.connectors.http import HttpRequest, ProviderError, get_ok
from engine.reason.schemas import Signal

SEASONAL_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_DAILY = "temperature_2m_mean,precipitation_sum"


def _days_in_month(month: date) -> int:
    following = date(month.year + month.month // 12, month.month % 12 + 1, 1)
    return (following - month).days


def _monthly(payload: dict) -> dict[date, dict]:
    """{month: {"days": n, "temp": mean °C, "rain": total mm}} from a daily payload."""
    daily = payload.get("daily") or {}
    times, temps, rains = (
        daily.get("time", []),
        daily.get("temperature_2m_mean", []),
        daily.get("precipitation_sum", []),
    )
    buckets: dict[date, dict] = defaultdict(lambda: {"days": 0, "t": 0.0, "r": 0.0})
    for day, temp, rain in zip(times, temps, rains, strict=False):
        if temp is None or rain is None:
            continue
        month = date.fromisoformat(day).replace(day=1)
        bucket = buckets[month]
        bucket["days"] += 1
        bucket["t"] += float(temp)
        bucket["r"] += float(rain)
    return {
        month: {"days": b["days"], "temp": b["t"] / b["days"], "rain": b["r"]}
        for month, b in buckets.items()
        if b["days"]
    }


def fetch(driver, ctx) -> list[Signal]:
    source = ctx.sources.weather
    region = driver.region
    if not source.enabled:
        return []
    if region is None or region.latitude is None or region.longitude is None:
        raise ProviderError("the driver's region has no latitude/longitude in the profile")
    point = {
        "latitude": f"{region.latitude:.3f}",
        "longitude": f"{region.longitude:.3f}",
        "daily": _DAILY,
        "timezone": ctx.timezone,
    }
    seasonal = {**point, "models": "ecmwf_seas5_ensemble_mean", "forecast_days": "215"}
    url = SEASONAL_URL
    if source.access == "commercial":
        key = os.environ.get(source.api_key_env)
        if not key:
            raise ProviderError(f"missing Open-Meteo key: set {source.api_key_env}")
        url, seasonal["apikey"] = url.replace("://", "://customer-", 1), key
    outlook = _monthly(get_ok(ctx.http, HttpRequest(url, seasonal), source.timeout_seconds).json())
    last_year = ctx.as_of.year - 1
    archive = {
        **point,
        "start_date": f"{last_year - source.normal_years + 1}-01-01",
        "end_date": f"{last_year}-12-31",
    }
    history = _monthly(
        get_ok(ctx.http, HttpRequest(ARCHIVE_URL, archive), source.timeout_seconds).json()
    )
    normals: dict[int, dict] = {}
    for calendar_month in range(1, 13):
        past = [v for m, v in history.items() if m.month == calendar_month and v["days"] >= 28]
        if past:
            normals[calendar_month] = {
                "temp": sum(v["temp"] for v in past) / len(past),
                "rain": sum(v["rain"] for v in past) / len(past),
            }
    months, parts = [], []
    for month in ctx.months:
        label = month.strftime("%Y-%m")
        value, normal = outlook.get(month), normals.get(month.month)
        if value is None or normal is None:
            months.append({"period": label, "coverage": "none"})
            parts.append(f"{label}: no outlook")
            continue
        complete = value["days"] >= _days_in_month(month) - 1
        entry = {
            "period": label,
            "coverage": "complete" if complete else "partial",
            "temp_c": round(value["temp"], 1),
            "temp_anomaly_c": round(value["temp"] - normal["temp"], 1),
            "rain_mm": round(value["rain"], 1) if complete else None,
            "rain_normal_mm": round(normal["rain"], 1),
        }
        months.append(entry)
        rain = (
            f"rain {entry['rain_mm']:.0f} mm vs normal {entry['rain_normal_mm']:.0f} mm"
            if complete
            else f"partial month ({value['days']} days)"
        )
        parts.append(
            f"{label}: {entry['temp_c']}°C ({entry['temp_anomaly_c']:+.1f} vs normal), {rain}"
        )
    return [
        Signal(
            driver_id=driver.id,
            kind="weather",
            summary=(
                f"{region.name} seasonal outlook (ECMWF SEAS5 mean, regional, not bias-corrected; "
                f"normal = {source.normal_years}-yr archive): " + "; ".join(parts)
            ),
            data={"region": region.id, "months": months},
            source_ref=SEASONAL_URL,
        )
    ]
