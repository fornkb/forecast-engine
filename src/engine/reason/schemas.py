"""Records shared by the reasoning layer.

Three groups:
  * company profile + driver templates (what a company is and what to watch)
  * signals collected for each driver (what changed outside)
  * model answers, checked findings and per-SKU bands (what it means for the forecast)

Model-facing answer schemas (`*Out`) use only required fields, enums, lists and
plain objects, because that is the subset OpenAI, Anthropic and Gemini all accept
for strict JSON output. Value checks happen in Python (adjust.py), not in the schema.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Literal

from pydantic import AwareDatetime, Field

from engine.config.base import StrictModel


class ReasoningError(RuntimeError):
    """A problem the operator can fix: missing setup, bad input or an unusable answer."""


def digest(value: object) -> str:
    """Stable short hash of any JSON-serializable value."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- profile

ParamValue = str | int | float | bool | list[str]


class Region(StrictModel):
    id: str
    name: str
    country: str = ""  # ISO 3166 alpha-2, e.g. IN
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class FamilyMatch(StrictModel):
    """How SKUs (forecast series) are assigned to a family. Any rule may match."""

    item_group: list[str] = Field(default_factory=list)
    item_prefix: list[str] = Field(default_factory=list)
    series: list[str] = Field(default_factory=list)


class Family(StrictModel):
    id: str
    name: str
    description: str = ""
    match: FamilyMatch = Field(default_factory=FamilyMatch)
    inputs: list[str] = Field(default_factory=list)  # raw materials / key inputs
    regions: list[str] = Field(default_factory=list)  # Region ids where it sells
    customers: str = ""


class Driver(StrictModel):
    """One external factor to watch. `template` names a catalogue entry."""

    id: str
    template: str
    families: list[str] = Field(default_factory=lambda: ["*"])
    mechanism: str = ""  # plain-words cause → effect, written or confirmed by the admin
    params: dict[str, ParamValue] = Field(default_factory=dict)
    enabled: bool = True


class Company(StrictModel):
    name: str
    summary: str = ""
    regions: list[Region] = Field(default_factory=list)


class CompanyProfile(StrictModel):
    status: Literal["draft", "confirmed"] = "draft"
    company: Company
    families: list[Family]
    drivers: list[Driver] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    # Broad news searches run every review to catch developments no driver covers.
    # Empty = built from the family names and the main country.
    scan: list[str] = Field(default_factory=list)
    notes: str = ""

    def family(self, family_id: str) -> Family | None:
        return next((item for item in self.families if item.id == family_id), None)

    def region(self, region_id: str) -> Region | None:
        return next((item for item in self.company.regions if item.id == region_id), None)


class DriverTemplate(StrictModel):
    """A reusable kind of external factor, shipped in reason/catalog/*.yaml."""

    id: str
    kind: Literal[
        "climate", "weather", "commodity", "calendar", "policy", "geopolitics", "market", "news"
    ]
    description: str
    connectors: list[Literal["news", "weather", "enso", "calendar", "scan"]]
    # Unlisted templates still work in existing profiles but are not offered to onboarding.
    listed: bool = True
    news_query: str = ""  # may use {param} placeholders filled from Driver.params
    mechanism_hint: str = ""
    required_params: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- signals


class Article(StrictModel):
    url: str
    title: str
    source: str
    published: AwareDatetime | None = None
    snippet: str = ""
    country: str = ""


class Signal(StrictModel):
    """One observation for one driver, kept small enough to show the model directly."""

    driver_id: str
    kind: Literal["news", "weather", "enso", "calendar", "measured"]
    summary: str
    data: dict = Field(default_factory=dict)
    articles: list[Article] = Field(default_factory=list)
    source_ref: str = ""
    ok: bool = True
    error: str | None = None


class DriverState(StrictModel):
    driver_id: str
    status: Literal["changed", "unchanged", "no_data", "error"]
    fingerprint: str
    signals: list[Signal] = Field(default_factory=list)
    # News URLs already reasoned about in an earlier run (not sent again).
    seen_urls: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- model answers


class TriageKeepOut(StrictModel):
    id: str
    relevance: int
    why: str


class TriageOut(StrictModel):
    keep: list[TriageKeepOut]


Channel = Literal["demand", "supply_risk", "cost"]
Tier = Literal["measured", "sourced", "judgment", "none"]


class FindingOut(StrictModel):
    driver_id: str
    family_id: str
    channel: Channel
    direction: Literal["up", "down", "mixed", "none"]
    months: list[str]
    low_pct: float
    high_pct: float
    evidence_tier: Tier
    quote: str
    citations: list[str]
    already_in_forecast: bool
    confidence: Literal["low", "medium", "high"]
    reasoning: str


class QuietOut(StrictModel):
    driver_id: str
    reason: str


class NewFactorOut(StrictModel):
    """A development found in the open news scan that no watched driver covers."""

    name: str
    family_id: str
    channel: Channel
    direction: Literal["up", "down", "mixed", "none"]
    months: list[str]
    low_pct: float
    high_pct: float
    evidence_tier: Tier
    quote: str
    citations: list[str]
    already_in_forecast: bool
    confidence: Literal["low", "medium", "high"]
    reasoning: str
    watch_query: str


class ReasoningOut(StrictModel):
    summary: str
    findings: list[FindingOut]
    new_factors: list[NewFactorOut]
    no_change: list[QuietOut]


class ParamOut(StrictModel):
    name: str
    value: str


class RegionOut(StrictModel):
    id: str
    name: str
    country: str
    latitude: float
    longitude: float


class FamilyOut(StrictModel):
    id: str
    name: str
    description: str
    item_groups: list[str]
    inputs: list[str]
    regions: list[str]
    customers: str


class DriverOut(StrictModel):
    id: str
    template: str
    families: list[str]
    mechanism: str
    params: list[ParamOut]


class ProfileDraftOut(StrictModel):
    company_name: str
    summary: str
    regions: list[RegionOut]
    families: list[FamilyOut]
    drivers: list[DriverOut]
    scan_topics: list[str]
    questions: list[str]


# --------------------------------------------------------------------------- results


class Finding(StrictModel):
    finding_id: str
    driver_id: str
    family_id: str
    channel: Channel
    direction: Literal["up", "down", "mixed", "none"]
    months: list[date]
    low_pct: float
    high_pct: float
    tier: Tier
    quote: str = ""
    citations: list[str] = Field(default_factory=list)  # packet ids, e.g. A3, S1, M2
    reasoning: str
    confidence: Literal["low", "medium", "high"]
    already_in_forecast: bool = False
    flags: list[str] = Field(default_factory=list)
    carried_from: str | None = None  # reason_id of the run that produced it
    # Set when the finding came from the open news scan rather than a watched driver.
    new_factor: str = ""
    watch_query: str = ""


class Band(StrictModel):
    series_id: str
    family_id: str
    period: date
    baseline: float
    low: float
    high: float
    finding_ids: list[str] = Field(default_factory=list)


class FamilyMonth(StrictModel):
    period: date
    baseline: float
    p10: float | None = None
    p90: float | None = None
    last_year: float | None = None
    low: float | None = None
    high: float | None = None


class FamilySummary(StrictModel):
    family_id: str
    name: str
    series_count: int
    months: list[FamilyMonth]
    history: list[tuple[date, float]] = Field(default_factory=list)  # last 12 months actuals


class Usage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


class LLMCall(StrictModel):
    call_id: str
    stage: Literal["onboard", "onboard_answers", "triage", "reason"]
    provider: str
    model: str
    started_at: AwareDatetime
    seconds: float
    usage: Usage
    cost: float | None = None
    currency: str = "USD"
    outcome: str  # ok | replayed | error:<reason>
    request_hash: str


class Reference(StrictModel):
    """What a short packet id (A1, S2, M3) points to, kept for the report."""

    id: str
    driver_id: str
    kind: str
    title: str
    url: str = ""
    source: str = ""
    published: str = ""
    text: str = ""


class ReasonRun(StrictModel):
    reason_id: str
    client_id: str
    forecast_run_id: int
    created_at: AwareDatetime
    as_of: date
    status: Literal["complete", "partial", "dry_run"]
    inputs_hash: str
    profile_hash: str
    prompt_version: str
    provider: str
    model: str | None
    scenario: str | None = None
    months: list[date]
    summary: str = ""
    families: list[FamilySummary]
    drivers: list[DriverState]
    findings: list[Finding]
    quiet: list[QuietOut] = Field(default_factory=list)
    not_assessed: list[str] = Field(default_factory=list)  # "driver_id: reason"
    references: list[Reference] = Field(default_factory=list)
    bands: list[Band] = Field(default_factory=list)
    calls: list[LLMCall] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    unassigned_series: int = 0
    estimate: dict = Field(default_factory=dict)  # dry-run token/cost estimate


class Override(StrictModel):
    reason_id: str
    finding_id: str
    action: Literal["accept", "dismiss", "edit"]
    low_pct: float | None = None
    high_pct: float | None = None
    actor: str
    note: str = ""
    at: datetime
