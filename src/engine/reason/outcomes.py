"""Versioned outcome and independent label inputs for advisory evaluation."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from engine.config.base import StrictModel
from engine.reason.config import Text
from engine.reason.facts import local_midnight
from engine.reason.schemas import ContextError


class Outcome(StrictModel):
    client_id: Text
    series_id: Text
    period: date
    target: Text
    unit: Text
    actual: float = Field(ge=0, allow_inf_nan=False)
    observed_at: AwareDatetime
    source_ref: Text
    revision: int = Field(default=1, ge=1)
    review_id: Text | None = None
    planner_point: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    selected_at: AwareDatetime | None = None
    scenario_id: Text | None = None
    condition_id: Text | None = None
    condition_observed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def shape(self):
        if self.period.day != 1:
            raise ValueError("period must be the first day of a client calendar month")
        if (self.planner_point is None) != (self.selected_at is None):
            raise ValueError("planner_point and selected_at must be supplied together")
        if (self.scenario_id is None) != (self.condition_id is None):
            raise ValueError("scenario_id and condition_id must be supplied together")
        if self.condition_observed_at and self.condition_id is None:
            raise ValueError("condition_observed_at requires a condition ID")
        return self

    @property
    def key(self):
        return self.series_id, self.period, self.target


class GroupLabel(StrictModel):
    client_id: Text
    review_id: Text
    group_id: Text
    label: Literal["affected", "unaffected", "unresolved"]


class TriageLabel(StrictModel):
    client_id: Text
    review_id: Text
    article_id: Text
    label: Literal["relevant", "irrelevant", "unresolved"]
    entity_type: Text | None = None
    entity_id: Text | None = None
    period: date | None = None
    series_id: Text | None = None

    @model_validator(mode="after")
    def shape(self):
        association = (self.entity_type, self.entity_id, self.period)
        if any(association) and (not all(association) or self.period.day != 1):
            raise ValueError("expected association needs entity type, ID and a month start")
        if self.series_id and not all(association):
            raise ValueError("a product label needs an expected entity and month")
        if any(association) and self.label != "relevant":
            raise ValueError("only relevant articles can have expected associations")
        return self


def _instant(value: str, zone: ZoneInfo) -> datetime:
    if len(value) == 10:
        day = date.fromisoformat(value)
        return local_midnight(day.fromordinal(day.toordinal() + 1), zone)
    value = datetime.fromisoformat(value)
    if value.utcoffset() is None:
        raise ValueError("timestamps require an offset; date-only values use the client calendar")
    return value.astimezone(UTC)


def read_csv(path: Path, schema, client_id: str, *, columns=None, zone=None):
    mapping = {key: key for key in schema.model_fields} | (columns or {})
    if mapping.keys() != schema.model_fields.keys() or len(set(mapping.values())) != len(mapping):
        raise ContextError("CSV column mapping references an unknown field or repeats a column")
    required = {mapping[name] for name, field in schema.model_fields.items() if field.is_required()}
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            headers = reader.fieldnames or []
            if len(set(headers)) != len(headers) or required - set(headers):
                raise ContextError(f"{path.name} needs unique columns including {sorted(required)}")
            rows = []
            for lineno, raw in enumerate(reader, 2):
                try:
                    if None in raw or any(value is None for value in raw.values()):
                        raise ValueError("incorrect number of CSV columns")
                    values = {
                        name: raw[column].strip()
                        for name, column in mapping.items()
                        if column in raw and raw[column].strip()
                    }
                    if zone is not None:
                        for name in ("observed_at", "selected_at", "condition_observed_at"):
                            if name in values:
                                values[name] = _instant(values[name], zone)
                    row = schema.model_validate(values)
                    if row.client_id != client_id:
                        raise ValueError("row belongs to a different client")
                except (ValidationError, ValueError, OverflowError) as exc:
                    raise ContextError(f"{path.name} row {lineno}: {exc}") from exc
                rows.append(row)
            return tuple(rows)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContextError(f"Could not read {path.name}: {exc}") from exc


class OutcomeIndex:
    """Keep actual revisions distinct from labels attached to a particular decision."""

    def __init__(self, outcomes: tuple[Outcome, ...], review_ids: set[str], as_of: datetime, zone):
        by_key = defaultdict(dict)
        decisions = {}
        recorded = set()
        for row in outcomes:
            if row.review_id and row.review_id not in review_ids:
                raise ContextError(
                    f"Outcome {row.key} references an unselected review {row.review_id}"
                )
            row_id = row.key, row.revision, row.review_id
            if row_id in recorded:
                raise ContextError(f"Duplicate outcome {row.key} revision {row.revision}")
            recorded.add(row_id)
            prior = by_key[row.key].setdefault(row.revision, row)
            if (prior.actual, prior.unit, prior.observed_at, prior.source_ref) != (
                row.actual,
                row.unit,
                row.observed_at,
                row.source_ref,
            ):
                raise ContextError(f"Conflicting outcome {row.key} revision {row.revision}")
            if row.observed_at < local_midnight(row.period, zone):
                raise ContextError(f"Outcome {row.key} was observed before its target month")
            if row.planner_point is not None or row.condition_id:
                decision_key = row.key, row.review_id
                if row.review_id is None and len(review_ids) > 1:
                    raise ContextError(
                        f"Decision label for {row.key} requires an explicit review ID"
                    )
                previous = decisions.setdefault(decision_key, row)
                fields = (
                    "planner_point",
                    "selected_at",
                    "condition_id",
                    "scenario_id",
                    "condition_observed_at",
                )
                if any(getattr(previous, key) != getattr(row, key) for key in fields):
                    raise ContextError(f"Conflicting decision labels for {row.key}")
        self.actuals, self.first_seen, self.decisions = {}, {}, decisions
        self.pending = set()
        for key, versions in by_key.items():
            ordered = [versions[number] for number in sorted(versions)]
            if [row.revision for row in ordered] != list(range(1, len(ordered) + 1)):
                raise ContextError(f"Outcome {key} revisions must start at 1 and be consecutive")
            if any(
                older.observed_at >= newer.observed_at or older.unit != newer.unit
                for older, newer in zip(ordered, ordered[1:], strict=False)
            ):
                raise ContextError(
                    f"Outcome {key} corrections must advance time and preserve units"
                )
            first = ordered[0]
            self.first_seen[key] = first.observed_at
            if eligible := [row for row in ordered if row.observed_at <= as_of]:
                self.actuals[key] = eligible[-1]
            else:
                self.pending.add(key)
        for (key, _), row in decisions.items():
            if row.selected_at and row.selected_at >= self.first_seen[key]:
                raise ContextError(
                    f"Planner selection for {key} must precede first outcome availability"
                )

    def decision(self, key, review_id) -> Outcome | None:
        return self.decisions.get((key, review_id), self.decisions.get((key, None)))
