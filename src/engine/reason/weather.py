"""Regional ECMWF seasonal series and issued NOAA CPC climate context."""

from __future__ import annotations

import calendar
import os
import re
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from math import isfinite
from statistics import fmean
from zoneinfo import ZoneInfo

from engine.reason.config import EnsoConfig, SeasonalWeatherConfig
from engine.reason.facts import local_midnight
from engine.reason.providers import (
    HttpRequest,
    ParsedEvidence,
    ProviderError,
    evidence_revision,
    next_month,
    optional_instant,
)
from engine.reason.schemas import (
    EnsoAdvisory,
    EnsoSeason,
    MonthCoverage,
    MonthlyWeather,
    QueryScope,
    VariableCoverage,
    WeatherMember,
)

SEASONAL_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"
NOAA_ROOT = "https://www.cpc.ncep.noaa.gov"
_STEP = timedelta(hours=6)
_VARIABLES = ("temperature_2m", "precipitation")
_MODEL = "ecmwf_seas5"
_NOAA_ZONE = ZoneInfo("America/New_York")


def seasonal_request(
    config: SeasonalWeatherConfig, scope: QueryScope, now: datetime
) -> HttpRequest:
    if scope.coordinates is None:
        raise ProviderError("missing_coordinates")
    lat, lon = scope.coordinates
    # Request UTC instants. Use a padding day for the backward precipitation interval.
    today = now.date()
    start = max(scope.valid_from.date(), today - timedelta(days=31))
    end = min(scope.valid_to.date(), today + timedelta(days=216))
    if end < start:
        raise ProviderError("outside_provider_horizon")
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "hourly": ",".join(_VARIABLES),
        "models": _MODEL if config.ensemble_members else _MODEL + "_ensemble_mean",
        "temperature_unit": config.temperature_unit,
        "precipitation_unit": config.precipitation_unit,
        "timezone": "GMT",
        "timeformat": "unixtime",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    url = SEASONAL_URL
    if config.access == "commercial":
        token = os.environ.get(config.api_key_env)
        if not token:
            raise ProviderError("missing_credentials")
        url = url.replace("://", "://customer-", 1)
        params["apikey"] = token
    return HttpRequest(url, params, {"Accept": "application/json"})


def _times(raw: object) -> tuple[datetime, ...]:
    if not isinstance(raw, list) or len(raw) > 1100:
        raise ValueError("Invalid seasonal time axis")
    timestamps = []
    for value in raw:
        if isinstance(value, int) and not isinstance(value, bool):
            instant = datetime.fromtimestamp(value, UTC)
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            instant = (
                parsed.replace(tzinfo=UTC) if parsed.utcoffset() is None else parsed.astimezone(UTC)
            )
        else:
            raise ValueError("Invalid seasonal timestamp")
        timestamps.append(instant)
    if any(end - start < _STEP for start, end in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("Seasonal intervals overlap or are out of order")
    return tuple(timestamps)


def _member_summary(
    member: str,
    values: object,
    times: tuple[datetime, ...],
    variable: str,
    unit: str,
    start: datetime,
    end: datetime,
) -> WeatherMember:
    total = 0.0
    covered = 0
    if not isinstance(values, list) or len(values) != len(times):
        return WeatherMember(member=member, quantity=None, covered_seconds=0)
    for timestamp, value in zip(times, values, strict=True):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
            continue
        precipitation = variable == "precipitation"
        if precipitation and value < 0:
            continue
        left, right = (
            (timestamp - _STEP, timestamp) if precipitation else (timestamp, timestamp + _STEP)
        )
        seconds = int((min(right, end) - max(left, start)).total_seconds())
        if seconds <= 0:
            continue
        if not precipitation:
            total += value * seconds
        elif unit.endswith("/h"):
            total += value * seconds / 3600
        else:
            # Partition a six-hour total at a client month boundary assuming a uniform rate.
            total += value * seconds / _STEP.total_seconds()
        covered += seconds
    complete = covered == (end - start).total_seconds()
    quantity = (total if variable == "precipitation" else total / covered) if complete else None
    return WeatherMember(member=member, quantity=quantity, covered_seconds=covered)


def parse_seasonal(
    payload: dict,
    config: SeasonalWeatherConfig,
    scope: QueryScope,
    captured_at: datetime,
) -> ParsedEvidence:
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Open-Meteo response must use UTC")
    hourly, units = payload["hourly"], payload["hourly_units"]
    times = _times(hourly["time"])
    if not isinstance(units, dict):
        raise ValueError("Missing seasonal units")
    latitude, longitude = float(payload["latitude"]), float(payload["longitude"])
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Invalid provider grid cell")
    issue = optional_instant(payload.get("issue_time") or payload.get("model_run"))
    zone = ZoneInfo(dict(scope.filters)["timezone"])
    expected_members = 51 if config.ensemble_members else 1
    location_id = scope.entity_ids[0].id
    revisions, coverage = [], []
    for month in scope.months:
        start, end = local_midnight(month, zone), local_midnight(next_month(month), zone)
        variables = []
        for variable in _VARIABLES:
            unit = "°F" if config.temperature_unit == "fahrenheit" else "°C"
            if variable == "precipitation":
                unit = config.precipitation_unit
            columns = sorted(
                key
                for key in hourly
                if key == variable or re.fullmatch(re.escape(variable) + r"_member\d+", key)
            )
            if not config.ensemble_members:
                columns = [variable] if variable in hourly else []
            members = []
            semantics = set()
            for key in columns[:expected_members]:
                reported_unit = units.get(key, units.get(variable))
                compatible = {unit, unit + "/h"} if variable == "precipitation" else {unit}
                if reported_unit in compatible:
                    if variable == "temperature_2m":
                        semantics.add("duration_weighted_sample_held_forward")
                    elif reported_unit.endswith("/h"):
                        semantics.add("preceding_interval_rate_integrated")
                    else:
                        semantics.add("preceding_interval_total_prorated_at_month_boundary")
                    members.append(
                        _member_summary(
                            key, hourly[key], times, variable, reported_unit, start, end
                        )
                    )
            observed = any(member.covered_seconds for member in members)
            values = [member.quantity for member in members if member.quantity is not None]
            complete = len(values) == len(columns) == expected_members
            variables.append(
                VariableCoverage(
                    variable=variable,
                    status="complete" if complete else "partial" if observed else "none",
                    complete_members=len(values),
                    expected_members=expected_members,
                )
            )
            if not observed:
                continue
            average = fmean(values) if complete else None
            climatology = next(
                (
                    reference
                    for reference in config.climatology
                    if reference.location_id == location_id
                    and (reference.month, reference.variable, reference.unit, reference.timezone)
                    == (month.month, variable, unit, zone.key)
                ),
                None,
            )
            summary = MonthlyWeather(
                period=month,
                variable=variable,
                model=_MODEL,
                timezone=zone.key,
                latitude=latitude,
                longitude=longitude,
                interval_seconds=int(_STEP.total_seconds()),
                expected_seconds=int((end - start).total_seconds()),
                members=tuple(members),
                expected_members=expected_members,
                interval_semantics=tuple(sorted(semantics)),
                minimum=min(values) if complete and expected_members > 1 else None,
                maximum=max(values) if complete and expected_members > 1 else None,
                climatology=climatology,
                anomaly=average - climatology.value
                if climatology and average is not None
                else None,
            )
            revisions.append(
                evidence_revision(
                    scope.client_id,
                    "weather",
                    "open_meteo",
                    f"{scope.coordinates}/{zone.key}/{month}/{variable}/{expected_members}",
                    captured_at,
                    released_at=issue,
                    issued_at=issue,
                    source_ref=SEASONAL_URL,
                    source_version=issue.isoformat() if issue else None,
                    valid_from=start,
                    valid_to=end,
                    quantity=average,
                    unit=unit,
                    weather=summary,
                )
            )
        status = (
            "complete"
            if all(value.status == "complete" for value in variables)
            else "partial"
            if any(value.status != "none" for value in variables)
            else "none"
        )
        reason = None if status == "complete" else "missing_intervals_variables_or_members"
        if status == "none":
            reason = "outside_returned_horizon_or_no_usable_values"
        coverage.append(
            MonthCoverage(period=month, status=status, reason=reason, variables=tuple(variables))
        )
    complete = all(value.status == "complete" for value in coverage)
    return ParsedEvidence(
        tuple(revisions),
        coverage="complete" if complete else "partial" if revisions else "none",
        reason=None if complete else "incomplete_monthly_outlook",
        months=tuple(coverage),
        covered_from=times[0] if times else None,
        covered_to=times[-1] if times else None,
    )


class _AdvisoryText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = False

    def handle_starttag(self, tag, attrs):
        if tag in {"style", "script"}:
            self.hidden = True
        if tag in {"br", "p", "div", "td", "li", "center", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"style", "script"}:
            self.hidden = False
        if tag in {"p", "div", "td", "li", "center", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


_MONTHS = {
    label.lower(): index
    for index in range(1, 13)
    for label in (calendar.month_name[index], calendar.month_abbr[index])
}
_M = "|".join(sorted(_MONTHS, key=lambda value: (-len(value), value)))
_DATE = re.compile(rf"\b(\d{{1,2}})\s+({_M})\s+(20\d{{2}})\b", re.I)
_SEASON = re.compile(rf"\b({_M})\s*[-–—]\s*({_M})(?:\s+(20\d{{2}}))?\b", re.I)


def enso_request(config: EnsoConfig) -> HttpRequest:
    return HttpRequest(NOAA_ROOT + config.path, headers={"Accept": "text/html"})


def parse_enso(
    html: bytes,
    config: EnsoConfig,
    scope: QueryScope,
    captured_at: datetime,
) -> ParsedEvidence:
    parser = _AdvisoryText()
    parser.feed(html.decode("utf-8", errors="replace"))
    text = " ".join("".join(parser.parts).split())
    heading = re.search(r"DIAGNOSTIC\s+DISCUSSION\s+issued\s+by", text, re.I)
    if heading is None:
        raise ValueError("Expected an issued NOAA CPC discussion")
    issue = _DATE.search(text, heading.end(), heading.end() + 250)
    if issue is None:
        raise ValueError("Missing NOAA CPC issue date")
    day, month, year = issue.groups()
    issue_date = date(int(year), _MONTHS[month.lower()], int(day))
    discussion = re.split(
        r"To receive an e-mail notification|The next ENSO Diagnostic",
        text[issue.end() :],
        maxsplit=1,
        flags=re.I,
    )[0][:20000]
    statements = re.split(r"(?<=[.!?])\s+(?=[A-Z])", discussion)
    seasons: dict[tuple[str, date, date], list[str]] = {}
    for statement in statements:
        for label in _SEASON.finditer(statement):
            first, last, stated_year = label.groups()
            start_month, end_month = _MONTHS[first.lower()], _MONTHS[last.lower()]
            if (end_month - start_month) % 12 != 2:
                continue
            if stated_year:
                start = date(int(stated_year) - (start_month > end_month), start_month, 1)
            else:
                options = [date(issue_date.year + offset, start_month, 1) for offset in (-1, 0, 1)]
                start = min(options, key=lambda d: abs((d - issue_date).days))
            end = next_month(next_month(next_month(start)))
            published = seasons.setdefault((label[0], start, end), [])
            if "%" in statement and statement.strip() not in published:
                published.append(statement.strip())
    advisory = EnsoAdvisory(
        issue_date=issue_date,
        seasons=tuple(
            EnsoSeason(label=label, start=start, end=end, statements=tuple(evidence))
            for (label, start, end), evidence in seasons.items()
        ),
    )
    issued = local_midnight(issue_date + timedelta(days=1), _NOAA_ZONE)
    coverage = tuple(
        MonthCoverage(
            period=month,
            status="complete"
            if any(s.start <= month < s.end for s in advisory.seasons)
            else "none",
        )
        for month in scope.months
    )
    covered = sum(month.status == "complete" for month in coverage)
    start = min((s.start for s in advisory.seasons), default=None)
    end = max((s.end for s in advisory.seasons), default=None)
    evidence = evidence_revision(
        scope.client_id,
        "enso",
        "noaa_cpc",
        issue_date.isoformat(),
        captured_at,
        source_ref=NOAA_ROOT + config.path,
        source_version=issue_date.isoformat(),
        released_at=issued,
        valid_from=local_midnight(start, _NOAA_ZONE) if start else None,
        valid_to=local_midnight(end, _NOAA_ZONE) if end else None,
        excerpt=discussion.strip(),
        enso=advisory,
    )
    return ParsedEvidence(
        (evidence,),
        coverage="complete" if covered == len(coverage) else "partial",
        reason=None if covered == len(coverage) else "uncovered_enso_season",
        months=coverage,
        covered_from=evidence.valid_from,
        covered_to=evidence.valid_to,
    )
