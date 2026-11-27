"""What a review found, in the terms of its one question:

    What changed that could affect this company's forecast, and what evidence do we have?

Three answers that are never confused with each other:
  * changes found (each with its evidence);
  * nothing important changed, and every source was checked;
  * incomplete: a source failed, so "nothing changed" cannot be claimed for what it covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.reason.profile import SCAN_ID
from engine.reason.schemas import Finding, ReasonRun, Signal

KIND_LABEL = {
    "news": "news",
    "weather": "weather outlook",
    "enso": "NOAA ENSO status",
    "calendar": "holiday calendar",
    "measured": "sales history",
}


def source_label(driver_id: str, signal: Signal) -> str:
    if driver_id == SCAN_ID:
        query = signal.data.get("query", "")
        return f"open news search '{query}'" if query else "open news search"
    name = driver_id.replace("_", " ")
    return f"{name}: {KIND_LABEL.get(signal.kind, signal.kind)}"


@dataclass(frozen=True)
class SourceCheck:
    driver_id: str
    label: str
    ok: bool
    detail: str


def checks(run: ReasonRun) -> list[SourceCheck]:
    """One line per source queried in this review, failed or not."""
    return [
        SourceCheck(
            state.driver_id,
            source_label(state.driver_id, signal),
            signal.ok,
            (signal.error or "failed") if not signal.ok else signal.summary,
        )
        for state in run.drivers
        for signal in state.signals
    ]


def is_change(finding: Finding) -> bool:
    return (
        "dismissed" not in finding.flags
        and not finding.already_in_forecast
        and finding.direction != "none"
    )


@dataclass
class Outcome:
    kind: str  # changes | nothing | estimate
    complete: bool
    changes: int
    new_factors: int
    checked: int
    failed: list[SourceCheck] = field(default_factory=list)
    not_assessed: list[str] = field(default_factory=list)

    @property
    def missing(self) -> int:
        return len(self.failed) + len(self.not_assessed)

    @property
    def failed_groups(self) -> list[tuple[str, list[str]]]:
        """Failures that share one reason (a rate-limited news provider) listed once."""
        groups: dict[str, list[str]] = {}
        for check in self.failed:
            label = check.label[:1].upper() + check.label[1:]
            groups.setdefault(check.detail, []).append(label)
        return list(groups.items())

    @property
    def headline(self) -> str:
        if self.kind == "estimate":
            return "Estimate only: nothing was assessed"
        plural = "" if self.missing == 1 else "s"
        if self.kind == "changes":
            text = (
                f"{self.changes} change{'' if self.changes == 1 else 's'} could affect the forecast"
            )
            if self.new_factors:
                text += (
                    f", including {self.new_factors} new factor"
                    f"{'' if self.new_factors == 1 else 's'} not on the watch list"
                )
            if not self.complete:
                text += f". Incomplete: {self.missing} source{plural} could not be checked"
            return text
        if self.complete:
            return f"Nothing important changed. {self.checked} sources checked"
        return (
            "No important change in the sources that worked, but "
            f"{self.missing} source{plural} could not be checked"
        )


def outcome(run: ReasonRun) -> Outcome:
    found = checks(run)
    failed = [c for c in found if not c.ok]
    changes = [f for f in run.findings if is_change(f)]
    kind = "changes" if changes else "nothing"
    if run.status == "dry_run":
        kind = "estimate"
    return Outcome(
        kind=kind,
        complete=not failed and not run.not_assessed,
        changes=len(changes),
        new_factors=sum(1 for f in changes if f.new_factor),
        checked=sum(1 for c in found if c.ok),
        failed=failed,
        not_assessed=list(run.not_assessed),
    )


__all__ = ["Outcome", "SourceCheck", "checks", "is_change", "outcome", "source_label"]
