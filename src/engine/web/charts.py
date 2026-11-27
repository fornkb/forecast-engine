"""Charts as server-rendered SVG (no chart library, nothing fetched from the internet).

`trend_chart`: actuals, then the forecast with a shaded range. The range is either
the forecasting model's p10–p90 or the review's what-if band, and the legend and
tooltip say which. Hover values come from a JSON payload that app.js reads; the
same numbers are always in a table on the page too.
"""

from __future__ import annotations

import json
import math
from datetime import date
from html import escape

from markupsafe import Markup

W, H = 760, 260
LEFT, RIGHT, TOP, BOTTOM = 52, 16, 16, 28
BAND_LABEL = {"model": "Model range (p10–p90)", "whatif": "What-if band"}


def compact(value: float | None) -> str:
    if value is None:
        return "–"
    size = abs(value)
    if size >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if size >= 10_000:
        return f"{value / 1_000:.0f}k"
    if size >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _nice_step(raw: float) -> float:
    """A round grid step (1, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 × a power of ten) at least `raw`."""
    if raw <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(raw))
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if step * power >= raw:
            return step * power
    return 10 * power


def _get(item, name: str):
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def trend_chart(
    history: list[tuple[date, float]], months: list, band: str = "model", label: str = ""
) -> dict:
    """`months`: items with period, forecast (or baseline), p10, p90, low, high."""
    points: list[dict] = [
        {"label": f"{m:%b %Y}", "actual": round(v, 1)} for m, v in history if v is not None
    ]
    first = len(points)
    for item in months:
        forecast = _get(item, "forecast")
        if forecast is None:
            forecast = _get(item, "baseline")
        if band == "model":
            low, high = _get(item, "p10"), _get(item, "p90")
        else:
            low, high = _get(item, "low"), _get(item, "high")
        low = forecast if low is None else low
        high = forecast if high is None else high
        point = {
            "label": f"{_get(item, 'period'):%b %Y}",
            "forecast": round(forecast, 1),
            "low": round(low, 1),
            "high": round(high, 1),
        }
        if band == "whatif" and _get(item, "p10") is not None:
            point["p10"], point["p90"] = _get(item, "p10"), _get(item, "p90")
        points.append(point)
    values = [
        v
        for p in points
        for k in ("actual", "forecast", "low", "high")
        if (v := p.get(k)) is not None
    ]
    tick = _nice_step(max(values, default=1.0) * 1.05 / 4)
    top = tick * 4
    inner_w, inner_h = W - LEFT - RIGHT, H - TOP - BOTTOM
    step = inner_w / max(1, len(points) - 1)

    def x(index: int) -> float:
        return LEFT + index * step

    def y(value: float) -> float:
        return TOP + inner_h - (value / top) * inner_h

    parts = []
    for level in (tick * i for i in range(5)):
        parts.append(
            f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(level):.1f}" y2="{y(level):.1f}"/>'
            f'<text class="axis" x="{LEFT - 8}" y="{y(level) + 4:.1f}" text-anchor="end">'
            f"{compact(level)}</text>"
        )
    every = 3 if len(points) <= 24 else 6
    for index, point in enumerate(points):
        if index % every == 0 or index == first:
            text = point["label"][:3] + " " + point["label"][-2:]
            parts.append(
                f'<text class="axis" x="{x(index):.1f}" y="{H - 8}" text-anchor="middle">'
                f"{escape(text)}</text>"
            )
    if first < len(points):
        divider = x(first) - step / 2 if first else x(0)
        parts.append(
            f'<rect class="future" x="{divider:.1f}" y="{TOP}" width="{W - RIGHT - divider:.1f}" '
            f'height="{inner_h}"/>'
            f'<text class="axis future-label" x="{divider + 6:.1f}" y="{TOP + 12}">Forecast</text>'
        )
        forecast = [(i, p) for i, p in enumerate(points) if "forecast" in p]
        upper = " ".join(f"{x(i):.1f},{y(p['high']):.1f}" for i, p in forecast)
        lower = " ".join(f"{x(i):.1f},{y(p['low']):.1f}" for i, p in reversed(forecast))
        parts.append(f'<polygon class="band {band}" points="{upper} {lower}"/>')
        line = [f"{x(i):.1f},{y(p['forecast']):.1f}" for i, p in forecast]
        if first:
            line.insert(0, f"{x(first - 1):.1f},{y(points[first - 1]['actual']):.1f}")
        parts.append(f'<polyline class="forecast" points="{" ".join(line)}"/>')
        for i, p in forecast:
            parts.append(f'<circle class="dot" cx="{x(i):.1f}" cy="{y(p["forecast"]):.1f}" r="3"/>')
    actual = " ".join(
        f"{x(i):.1f},{y(p['actual']):.1f}" for i, p in enumerate(points) if "actual" in p
    )
    if actual:
        parts.append(f'<polyline class="actual" points="{actual}"/>')
    parts.append(
        f'<line class="cross" x1="0" x2="0" y1="{TOP}" y2="{TOP + inner_h}" visibility="hidden"/>'
        f'<rect class="hit" x="{LEFT}" y="{TOP}" width="{inner_w}" height="{inner_h}"/>'
    )
    title = escape(label or "Demand")
    svg = (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}: actual sales and the '
        f'forecast with its {escape(BAND_LABEL[band].lower())}">{"".join(parts)}</svg>'
    )
    payload = {
        "x": [round(x(i), 1) for i in range(len(points))],
        "w": W,
        "band": BAND_LABEL[band],
        "points": points,
    }
    return {"svg": Markup(svg), "data": json.dumps(payload), "band": band}


def sparkline(history: list[tuple[date, float]], forecast: list[float]) -> Markup:
    """Tiny trend for table rows: actuals solid, forecast dashed. Decorative only."""
    values = [v for _, v in history] + list(forecast)
    if not values or max(values) <= 0:
        return Markup("")
    w, h, pad = 120, 30, 3
    top, count = max(values), len(values)
    step = (w - 2 * pad) / max(1, count - 1)

    def point(index: int, value: float) -> str:
        return f"{pad + index * step:.1f},{h - pad - (value / top) * (h - 2 * pad):.1f}"

    past = " ".join(point(i, v) for i, (_, v) in enumerate(history))
    start = len(history) - 1 if history else 0
    ahead = " ".join(
        point(start + i, v)
        for i, v in enumerate(([history[-1][1]] if history else []) + list(forecast))
    )
    return Markup(
        f'<svg class="spark" viewBox="0 0 {w} {h}" aria-hidden="true">'
        f'<polyline class="spark-past" points="{past}"/>'
        f'<polyline class="spark-ahead" points="{ahead}"/></svg>'
    )
