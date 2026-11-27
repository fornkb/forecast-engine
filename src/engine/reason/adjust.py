"""Python checks the model's answer and computes the numbers. The model never does arithmetic.

Checks per finding (anything that fails is fixed visibly, never silently):
  * driver, family and months must exist in this run          → dropped otherwise
  * citations must be packet ids                               → unknown ids removed
  * sourced: the quote must really appear in a cited article   → else downgraded to judgment
  * measured: must cite a measured effect for that family      → else downgraded to judgment
  * judgment: capped at ±max_judgment_pct
  * supply_risk / cost / already-in-forecast / tier none       → no quantity change
New factors (from the open news scan) are stricter, because no driver vouches for them:
  * the quote must appear in a cited article or earlier finding → dropped otherwise
  * they cannot be "measured"                                   → downgraded to judgment
  * judgment is capped at ±max_new_factor_pct
Bands: baseline × (1 + Σ low%) to baseline × (1 + Σ high%) per SKU-month, never below 0.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from engine.reason.forecast import ForecastView
from engine.reason.schemas import (
    Band,
    FamilySummary,
    Finding,
    NewFactorOut,
    Override,
    QuietOut,
    ReasoningOut,
    Reference,
)

NEW_PREFIX = "new:"


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9%]+", text.lower()))


def _month(value: str) -> date | None:
    try:
        year, month = value.strip()[:7].split("-")
        return date(int(year), int(month), 1)
    except ValueError:
        return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "factor"


def _quoted(quote: str, keys: list[str], references: dict[str, Reference]) -> bool:
    text = _norm(quote)
    return len(text) >= 12 and any(
        text in _norm(references[key].text) for key in keys if key[0] in "ASP"
    )


def _validate(
    raw,
    driver_id: str,
    references: dict[str, Reference],
    months: list[date],
    cap: float,
    measured: dict[str, dict],
    notes: list[str],
    label: str,
) -> Finding | None:
    when = sorted({m for value in raw.months if (m := _month(value)) in months})
    if not when:
        notes.append(f"dropped {label}/{raw.family_id}: no forecast month")
        return None
    flags = []
    cited = [key for key in raw.citations if key in references]
    if len(cited) < len(raw.citations):
        flags.append("unknown_citation_removed")
    low, high, tier = float(raw.low_pct), float(raw.high_pct), raw.evidence_tier
    if low > high:
        low, high = high, low
        flags.append("low_high_swapped")
    if tier == "sourced" and not _quoted(raw.quote, [k for k in cited if k[0] in "AS"], references):
        tier = "judgment"
        flags.append("quote_not_found_downgraded")
    if tier == "measured":
        match = [
            measured[key]
            for key in cited
            if key in measured and measured[key].get("family_id") == raw.family_id
        ]
        if not match:
            tier = "judgment"
            flags.append("measured_without_matching_measure_downgraded")
        else:
            limit = max(1.0, 1.5 * max(abs(m["uplift_pct"]) for m in match))
            if max(abs(low), abs(high)) > limit:
                low, high = max(low, -limit), min(high, limit)
                flags.append(f"capped_to_measured_{limit:g}pct")
    if tier == "judgment" and max(abs(low), abs(high)) > cap:
        low, high = max(low, -cap), min(high, cap)
        flags.append(f"judgment_capped_{cap:g}pct")
    if raw.channel != "demand" or raw.already_in_forecast or tier == "none":
        if low or high:
            flags.append("no_quantity_change")
        low = high = 0.0
    if not cited:
        flags.append("no_valid_citation")
    return Finding(
        finding_id="",
        driver_id=driver_id,
        family_id=raw.family_id,
        channel=raw.channel,
        direction=raw.direction,
        months=when,
        low_pct=round(max(low, -100.0), 2),
        high_pct=round(max(high, -100.0), 2),
        tier=tier,
        quote=raw.quote if tier == "sourced" else "",
        citations=cited,
        reasoning=" ".join(raw.reasoning.split())[:600],
        confidence=raw.confidence,
        already_in_forecast=raw.already_in_forecast,
        flags=flags,
    )


def check_new(
    items: list[NewFactorOut],
    references: dict[str, Reference],
    family_ids: list[str],
    months: list[date],
    new_cap: float,
    watched: set[str],
) -> tuple[list[Finding], list[str]]:
    """New factors from the open scan: kept only when a cited item really says it."""
    findings, notes = [], []
    for raw in items:
        name = " ".join(raw.name.split())[:80] or "Unnamed factor"
        if raw.family_id not in family_ids:
            notes.append(f"dropped new factor '{name}': unknown family {raw.family_id}")
            continue
        if _slug(name) in watched:
            notes.append(f"dropped new factor '{name}': already a watched driver")
            continue
        cited = [key for key in raw.citations if key in references and key[0] in "ASP"]
        if not _quoted(raw.quote, cited, references):
            notes.append(f"dropped new factor '{name}': its quote is not in any cited item")
            continue
        flags = []
        tier = raw.evidence_tier
        if tier == "measured":
            tier = "judgment"
            flags.append("new_factor_cannot_be_measured")
        candidate = raw.model_copy(update={"evidence_tier": tier})
        finding = _validate(
            candidate, NEW_PREFIX + _slug(name), references, months, new_cap, {}, notes, name
        )
        if finding is None:
            continue
        findings.append(
            finding.model_copy(
                update={
                    "flags": [*flags, *finding.flags],
                    "quote": " ".join(raw.quote.split())[:300],
                    "new_factor": name,
                    "watch_query": " ".join(raw.watch_query.split())[:80],
                }
            )
        )
    return findings, notes


def check(
    out: ReasoningOut,
    references: dict[str, Reference],
    families_by_driver: dict[str, list[str]],
    months: list[date],
    cap: float,
    measured: dict[str, dict],
) -> tuple[list[Finding], list[QuietOut], list[str]]:
    """Validated watched-driver findings, quiet drivers, and notes about anything dropped."""
    findings, notes = [], []
    for raw in out.findings:
        if raw.driver_id not in families_by_driver:
            notes.append(f"dropped finding for unknown driver {raw.driver_id}")
            continue
        if raw.family_id not in families_by_driver[raw.driver_id]:
            notes.append(
                f"dropped {raw.driver_id} finding for family {raw.family_id} it does not cover"
            )
            continue
        finding = _validate(
            raw, raw.driver_id, references, months, cap, measured, notes, raw.driver_id
        )
        if finding is not None:
            findings.append(finding)
    quiet = [q for q in out.no_change if q.driver_id in families_by_driver]
    return findings, quiet, notes


def number(findings: list[Finding]) -> list[Finding]:
    return [f.model_copy(update={"finding_id": f"F{i}"}) for i, f in enumerate(findings, 1)]


def effective(findings: list[Finding], overrides: list[Override]) -> list[Finding]:
    """Apply the planner's accept / edit / dismiss decisions (latest per finding)."""
    by_id = {o.finding_id: o for o in overrides}
    result = []
    for finding in findings:
        decision = by_id.get(finding.finding_id)
        if decision is None:
            result.append(finding)
        elif decision.action == "dismiss":
            result.append(finding.model_copy(update={"flags": [*finding.flags, "dismissed"]}))
        elif decision.action == "edit":
            low, high = sorted((decision.low_pct or 0.0, decision.high_pct or 0.0))
            result.append(
                finding.model_copy(
                    update={
                        "low_pct": low,
                        "high_pct": high,
                        "flags": [*finding.flags, f"edited_by_{decision.actor}"],
                    }
                )
            )
        else:
            result.append(
                finding.model_copy(
                    update={"flags": [*finding.flags, f"accepted_by_{decision.actor}"]}
                )
            )
    return result


def active(finding: Finding) -> bool:
    return bool(
        finding.channel == "demand"
        and "dismissed" not in finding.flags
        and (finding.low_pct or finding.high_pct)
    )


def bands(
    view: ForecastView, findings: list[Finding]
) -> tuple[list[Band], list[FamilySummary], list[str]]:
    effects: dict[tuple[str, date], list[Finding]] = defaultdict(list)
    for finding in findings:
        if active(finding):
            for month in finding.months:
                effects[finding.family_id, month].append(finding)
    rows = []
    for row in view.rows:
        applied = effects.get((row.family_id, row.period), [])
        low = sum(f.low_pct for f in applied)
        high = sum(f.high_pct for f in applied)
        rows.append(
            Band(
                series_id=row.series_id,
                family_id=row.family_id,
                period=row.period,
                baseline=row.baseline,
                low=max(0.0, row.baseline * (1 + low / 100)),
                high=max(0.0, row.baseline * (1 + high / 100)),
                finding_ids=[f.finding_id for f in applied],
            )
        )
    totals: dict[tuple[str, date], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for band in rows:
        totals[band.family_id, band.period][0] += band.low
        totals[band.family_id, band.period][1] += band.high
    families = []
    for family in view.families:
        months = [
            m.model_copy(
                update={
                    "low": round(totals[family.family_id, m.period][0], 2),
                    "high": round(totals[family.family_id, m.period][1], 2),
                }
            )
            for m in family.months
        ]
        families.append(family.model_copy(update={"months": months}))
    stacked = [
        f"{len(items)} drivers stack on {family} in {month:%Y-%m}; "
        "check they are not the same effect"
        for (family, month), items in effects.items()
        if len({f.driver_id for f in items}) >= 3
    ]
    return rows, families, stacked
