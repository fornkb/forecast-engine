"""Render a saved reasoning run: one self-contained HTML page for managers, plus Markdown/JSON.

Reads only the stored run document, so a report reopens offline at zero cost.
All text from sources is escaped (article titles are untrusted).
"""

from __future__ import annotations

from collections import Counter
from html import escape

from engine.reason.adjust import active
from engine.reason.outcome import outcome
from engine.reason.schemas import FamilySummary, Finding, ReasonRun

TIER_LABEL = {
    "measured": "Measured in your sales history",
    "sourced": "Quoted from a source",
    "judgment": "AI judgment (capped)",
    "none": "Direction only",
}
CHANNEL_LABEL = {"demand": "Demand", "supply_risk": "Supply risk", "cost": "Cost"}


def _n(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f}"


def _pct(low: float, high: float) -> str:
    if low == high == 0:
        return "no quantity change"
    return f"{low:+g}% to {high:+g}%"


def totals(run: ReasonRun) -> dict:
    calls = run.calls
    known = [c.cost for c in calls if c.cost is not None]
    return {
        "calls": len(calls),
        "input_tokens": sum(c.usage.input_tokens for c in calls),
        "output_tokens": sum(c.usage.output_tokens for c in calls),
        "cost": round(sum(known), 4) if len(known) == len(calls) else None,
        "currency": calls[0].currency if calls else "USD",
    }


def _sparkline(family: FamilySummary, width: int = 420, height: int = 120) -> str:
    """Last 12 months actuals, then the forecast with its ± band, as inline SVG."""
    history = [value for _, value in family.history]
    base = [m.baseline for m in family.months]
    lows = [m.low if m.low is not None else m.baseline for m in family.months]
    highs = [m.high if m.high is not None else m.baseline for m in family.months]
    values = [*history, *base, *lows, *highs]
    top = max(values) if values and max(values) > 0 else 1.0
    count = len(history) + len(base)
    step = width / max(1, count - 1)

    def point(index: int, value: float) -> str:
        return f"{index * step:.1f},{height - 8 - (value / top) * (height - 16):.1f}"

    actual = " ".join(point(i, v) for i, v in enumerate(history))
    offset = len(history)
    forecast = " ".join(point(offset + i, v) for i, v in enumerate(base))
    band = (
        " ".join(point(offset + i, v) for i, v in enumerate(highs))
        + " "
        + " ".join(point(offset + i, v) for i, v in reversed(list(enumerate(lows))))
    )
    joint = (
        point(offset - 1, history[-1]) + " " + point(offset, base[0]) if history and base else ""
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark" role="img" '
        f'aria-label="Last 12 months and forecast with band">'
        f'<polygon points="{band}" class="band"/>'
        f'<polyline points="{actual}" class="actual"/>'
        f'<polyline points="{joint}" class="joint"/>'
        f'<polyline points="{forecast}" class="forecast"/></svg>'
    )


def finding_title(finding: Finding) -> str:
    """The cause a finding is about: a watched driver, or a new factor from the open scan."""
    if finding.new_factor:
        return f"New factor: {finding.new_factor}"
    return finding.driver_id.replace("_", " ").title()


def _finding_card(finding: Finding, refs: dict) -> str:
    links = []
    for key in finding.citations:
        ref = refs.get(key)
        if ref is None:
            continue
        label = escape(f"{key}: {ref.title[:110]}")
        if ref.url.startswith(("http://", "https://")):
            links.append(
                f'<li><a href="{escape(ref.url)}" rel="noopener noreferrer">{label}</a>'
                f"{' · ' + escape(ref.source) if ref.source else ''}"
                f"{' · ' + escape(ref.published) if ref.published else ''}</li>"
            )
        else:
            links.append(f"<li>{label}</li>")
    months = ", ".join(m.strftime("%b %Y") for m in finding.months)
    flags = [f for f in finding.flags if f]
    state = "dismissed" if "dismissed" in finding.flags else ""
    quote = f"<blockquote>“{escape(finding.quote)}”</blockquote>" if finding.quote else ""
    carried = (
        f'<span class="tag muted">unchanged since {escape(finding.carried_from)}</span>'
        if finding.carried_from
        else ""
    )
    return f"""
    <div class="finding {finding.channel} {state}">
      <div class="row"><b>{escape(finding.finding_id)} · {escape(finding_title(finding))}</b>
        <span class="tag {finding.channel}">{CHANNEL_LABEL[finding.channel]}</span>
        <span class="tag">{TIER_LABEL[finding.tier]}</span>
        <span class="tag">confidence {finding.confidence}</span>{carried}</div>
      <div class="effect">{_pct(finding.low_pct, finding.high_pct)} · {months}</div>
      <p>{escape(finding.reasoning)}</p>{quote}
      <ul class="cites">{"".join(links)}</ul>
      {f'<div class="flags">checks: {escape(", ".join(flags))}</div>' if flags else ""}
    </div>"""


CSS = """
:root{--ink:#1f2933;--muted:#6b7785;--line:#e3e7ec;--bg:#f7f8fa;--accent:#2563eb;
--up:#15803d;--down:#b91c1c;--risk:#b45309}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
color:var(--ink);background:var(--bg)}main{max-width:1060px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:28px 0 10px}h3{font-size:16px;margin:0}
.sub{color:var(--muted)}.banner{background:#fff7e6;border:1px solid #f5d38f;padding:10px 14px;
border-radius:8px;margin:14px 0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:10px;margin:16px 0}.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}
.card b{display:block;font-size:22px}.summary{background:#fff;border-left:4px solid var(--accent);
padding:12px 16px;border-radius:6px}.family{background:#fff;border:1px solid var(--line);border-radius:12px;
padding:16px;margin:14px 0}.family-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}th,td{text-align:right;
padding:5px 8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
.spark{width:100%;max-width:420px;height:120px}.spark .band{fill:#2563eb22}.spark polyline{fill:none;
stroke-width:2}.spark .actual{stroke:#475569}.spark .forecast,.spark .joint{stroke:var(--accent);
stroke-dasharray:5 4}.finding{border:1px solid var(--line);border-left:4px solid var(--up);border-radius:8px;
padding:10px 12px;margin:10px 0}.finding.supply_risk,.finding.cost{border-left-color:var(--risk)}
.finding.dismissed{opacity:.5}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.tag{font-size:12px;background:#eef2f7;border-radius:999px;padding:2px 8px}.tag.muted{color:var(--muted)}
.effect{font-weight:600;margin:6px 0}.cites{margin:6px 0 0;padding-left:18px;font-size:13px;color:var(--muted)}
.flags{font-size:12px;color:var(--muted)}blockquote{margin:6px 0;padding-left:10px;border-left:3px solid var(--line);
color:var(--muted)}.quiet li,.gaps li{margin:3px 0}footer{margin-top:30px;color:var(--muted);font-size:13px}
"""


def _sources(run: ReasonRun) -> str:
    items = []
    for ref in run.references:
        if ref.kind == "previous":
            continue
        label = escape(f"{ref.id} [{ref.driver_id}] {ref.title[:140]}")
        meta = " · ".join(escape(part) for part in (ref.source, ref.published) if part)
        if ref.url.startswith(("http://", "https://")):
            label = f'<a href="{escape(ref.url)}" rel="noopener noreferrer">{label}</a>'
        items.append(f"<li>{label}{' · ' + meta if meta else ''}</li>")
    return "".join(items)


def render_html(run: ReasonRun) -> str:
    refs = {ref.id: ref for ref in run.references}
    used = totals(run)
    findings_by_family: dict[str, list[Finding]] = {}
    for finding in run.findings:
        findings_by_family.setdefault(finding.family_id, []).append(finding)
    counts = Counter(f.channel for f in run.findings if "dismissed" not in f.flags)
    result = outcome(run)
    cost_text = (
        "unknown (set reasoning.llm.rates)"
        if used["cost"] is None
        else f"{used['cost']:.4f} {used['currency']}"
    )
    sections = []
    for family in run.families:
        items = findings_by_family.get(family.family_id, [])
        rows = "".join(
            f"<tr><td>{m.period:%b %Y}</td><td>{_n(m.last_year)}</td><td>{_n(m.baseline)}</td>"
            f"<td>{_n(m.p10)} – {_n(m.p90)}</td><td><b>{_n(m.low)} – {_n(m.high)}</b></td></tr>"
            for m in family.months
        )
        cards = "".join(_finding_card(f, refs) for f in items) or (
            '<p class="sub">No external driver changes this family\'s forecast this month.</p>'
        )
        moved = any(active(f) for f in items)
        sections.append(f"""
  <section class="family">
    <div class="family-head"><div><h3>{escape(family.name)}</h3>
      <div class="sub">{family.series_count} SKUs · {"band applied" if moved else "forecast unchanged"}</div></div>
      {_sparkline(family)}</div>
    <table><tr><th>Month</th><th>Same month last year</th><th>Forecast</th>
      <th>Model range (p10–p90)</th><th>What-if band</th></tr>{rows}</table>
    {cards}
  </section>""")
    quiet = "".join(f"<li><b>{escape(q.driver_id)}</b>: {escape(q.reason)}</li>" for q in run.quiet)
    unchanged = [d.driver_id for d in run.drivers if d.status == "unchanged"]
    if unchanged:
        quiet += (
            f"<li><b>{escape(', '.join(unchanged))}</b>: no new information since the last "
            "run; earlier findings carried forward at no cost.</li>"
        )
    gaps = "".join(f"<li>{escape(g)}</li>" for g in [*run.not_assessed, *run.gaps])
    failed = "".join(
        f"<li><b>{escape(c.label)}</b>: {escape(c.detail)}</li>" for c in result.failed
    ) + "".join(f"<li>{escape(item)}</li>" for item in run.not_assessed)
    banner = (
        f'<div class="banner"><b>Scenario data.</b> {escape(run.scenario)}. Signals in this '
        "report are illustrative, prepared for a demonstration, not live news.</div>"
        if run.scenario
        else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forecast reasoning · {escape(run.client_id)} · {run.as_of:%b %Y}</title><style>{CSS}</style></head>
<body><main>
  <h1>What changed that could affect the forecast</h1>
  <div class="sub">{escape(run.client_id)} · forecast run {run.forecast_run_id} · reviewed {run.as_of:%d %b %Y}
   · months {run.months[0]:%b %Y} – {run.months[-1]:%b %Y}</div>
  {banner}
  <div class="summary"><b>{escape(result.headline)}.</b>
   {escape(run.summary) if run.summary else ""}</div>
  {f'<div class="banner"><b>Could not be checked</b><ul>{failed}</ul></div>' if failed else ""}
  <div class="cards">
    <div class="card"><b>{result.checked}</b>sources checked</div>
    <div class="card"><b>{counts.get("demand", 0)}</b>demand findings</div>
    <div class="card"><b>{result.new_factors}</b>new factors</div>
    <div class="card"><b>{counts.get("supply_risk", 0) + counts.get("cost", 0)}</b>supply / cost alerts</div>
  </div>
  <h2>By product family</h2>
  <p class="sub">The what-if band shows the forecast under the stated conditions. It is not a
  probability range; the model range (p10–p90) is the forecasting model's own uncertainty.</p>
  {"".join(sections)}
  <h2>Checked, no change</h2><ul class="quiet">{quiet or "<li>–</li>"}</ul>
  <h2>Gaps and limits</h2><ul class="gaps">{gaps or "<li>None</li>"}</ul>
  <h2>Sources checked</h2><ul class="cites">{_sources(run) or "<li>–</li>"}</ul>
  <footer>Run {escape(run.reason_id)} · {escape(run.provider)} {escape(run.model or "")} ·
   prompt {escape(run.prompt_version)} · cost {escape(cost_text)} · AI proposes, Python checks every
   number against its evidence tier and computes the bands.</footer>
</main></body></html>"""


def render_markdown(run: ReasonRun) -> str:
    used = totals(run)
    result = outcome(run)
    lines = [
        f"# External factors · {run.client_id} · {run.as_of:%d %b %Y}",
        "",
        f"Forecast run {run.forecast_run_id} · run `{run.reason_id}` · {used['calls']} AI calls · "
        f"cost {'unknown' if used['cost'] is None else used['cost']}",
        "",
        f"**{result.headline}.**",
    ]
    if result.failed:
        lines += ["", "Could not be checked:"] + [f"- {c.label}: {c.detail}" for c in result.failed]
    if run.scenario:
        lines += ["", f"> Scenario data: {run.scenario}. Illustrative, not live news."]
    if run.summary:
        lines += ["", run.summary]
    for family in run.families:
        lines += [
            "",
            f"## {family.name} ({family.series_count} SKUs)",
            "",
            "| Month | Last year | Forecast | What-if band |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| {m.period:%b %Y} | {_n(m.last_year)} | {_n(m.baseline)} | "
            f"{_n(m.low)} – {_n(m.high)} |"
            for m in family.months
        ]
        for finding in [f for f in run.findings if f.family_id == family.family_id]:
            lines += [
                "",
                f"- **{finding.finding_id} {finding_title(finding)}** · "
                f"{CHANNEL_LABEL[finding.channel]} · {_pct(finding.low_pct, finding.high_pct)}"
                f" · {TIER_LABEL[finding.tier]} · {finding.reasoning} "
                f"(cites {', '.join(finding.citations) or 'none'})",
            ]
    lines += ["", "## Checked, no change", ""]
    lines += [f"- {q.driver_id}: {q.reason}" for q in run.quiet] or ["- none"]
    if run.not_assessed or run.gaps:
        lines += ["", "## Gaps", ""] + [f"- {g}" for g in [*run.not_assessed, *run.gaps]]
    lines += ["", "## Sources", ""]
    lines += [f"- {r.id}: {r.title} {r.url}".rstrip() for r in run.references]
    return "\n".join(lines) + "\n"
