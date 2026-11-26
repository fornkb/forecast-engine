"""Connectors, change detection, measured effects and the company profile (no network)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from reason_kit import PROFILE, make_client, seed_store

from engine.reason.config import SourcesConfig
from engine.reason.connectors import Context, calendar, enso, news, weather
from engine.reason.connectors.http import HttpResponse, ProviderError
from engine.reason.forecast import analog_uplift, assign_family
from engine.reason.profile import (
    draft_profile,
    from_draft,
    load_catalog,
    load_profile,
    resolve,
    save_profile,
)
from engine.reason.schemas import CompanyProfile, ProfileDraftOut, Signal
from engine.reason.signals import fingerprint

FIXTURE = Path(__file__).parent / "fixtures" / "reasoning" / "noaa_enso.html"
MONTHS = [date(2027, 3, 1), date(2027, 4, 1)]


def ctx(http, **sources) -> Context:
    return Context(
        as_of=datetime(2027, 2, 26, 12, tzinfo=UTC),
        timezone="Asia/Kolkata",
        months=MONTHS,
        sources=SourcesConfig.model_validate(sources),
        http=http,
    )


def driver(template: str, params: dict, families=("bottles",)):
    profile = CompanyProfile.model_validate(PROFILE)
    from engine.reason.schemas import Driver

    item = Driver(id="d", template=template, families=list(families), params=params)
    return resolve(profile, load_catalog(), item)


def test_gdelt_dedupes_syndicated_copies_and_reports_bad_queries():
    body = {
        "articles": [
            {
                "url": "https://a.example/1?utm_source=x",
                "title": "Resin prices jump in India",
                "seendate": "20270220T101500Z",
                "domain": "a.example",
                "sourcecountry": "India",
            },
            {
                "url": "https://b.example/2",
                "title": "Resin prices jump in India!",
                "seendate": "20270220T111500Z",
                "domain": "b.example",
            },
            {
                "url": "https://c.example/3",
                "title": "Freight rates ease at western ports",
                "seendate": "bad",
                "domain": "c.example",
            },
        ]
    }
    seen = {}

    def http(request, timeout):
        seen.update(request.params)
        return HttpResponse(200, json.dumps(body).encode())

    (signal,) = news.fetch(driver("commodity_news", {"commodity": "PET resin"}), ctx(http))
    assert seen["query"].startswith('"PET resin" price') and "sourcelang:english" in seen["query"]
    assert [a.url for a in signal.articles] == ["https://a.example/1", "https://c.example/3"]
    assert signal.articles[0].published.day == 20 and signal.articles[1].published is None
    with pytest.raises(ProviderError, match="GDELT"):
        news.fetch(
            driver("commodity_news", {"commodity": "x"}),
            ctx(lambda r, t: HttpResponse(200, b"Your query was too short")),
        )


def test_enso_parses_status_and_issue_date():
    page = (
        "<html><b>DIAGNOSTIC DISCUSSION issued by CLIMATE PREDICTION CENTER/NCEP/NWS "
        "12 November 2026</b> ENSO Alert System Status: El Ni&ntilde;o Advisory "
        "Synopsis: El Ni&ntilde;o is likely to continue (79% chance). More text.</html>"
    )
    data = enso.parse(page)
    assert data == {
        "issue_date": "2026-11-12",
        "status": "El Niño Advisory",
        "synopsis": "El Niño is likely to continue (79% chance). More text.",
    }
    assert (
        enso.parse(FIXTURE.read_text(encoding="utf-8", errors="replace"))["status"] == "not stated"
    )


def test_weather_monthly_anomaly_and_missing_months():
    def daily(start: date, days: int, temp: float, rain: float) -> dict:
        times = [date.fromordinal(start.toordinal() + i).isoformat() for i in range(days)]
        return {
            "daily": {
                "time": times,
                "temperature_2m_mean": [temp] * days,
                "precipitation_sum": [rain] * days,
            }
        }

    seasonal = daily(date(2027, 3, 1), 31, 30.0, 0.1)  # March only: April has no outlook
    archive = {"daily": {k: [] for k in ("time", "temperature_2m_mean", "precipitation_sum")}}
    for year in range(2017, 2027):
        part = daily(date(year, 3, 1), 31, 28.0, 0.2)["daily"]
        for key in archive["daily"]:
            archive["daily"][key] += part[key]

    def http(request, timeout):
        return HttpResponse(
            200, json.dumps(seasonal if "seasonal" in request.url else archive).encode()
        )

    (signal,) = weather.fetch(driver("seasonal_weather", {"region": "r1"}), ctx(http))
    march, april = signal.data["months"]
    assert march["temp_anomaly_c"] == 2.0 and march["rain_mm"] == pytest.approx(3.1)
    assert march["rain_normal_mm"] == pytest.approx(6.2) and april["coverage"] == "none"
    assert "no outlook" in signal.summary


def test_festival_shift_and_measured_uplift():
    history = {
        date(y, m, 1): (150.0 if (y, m) in {(2024, 10), (2025, 11)} else 100.0)
        for y in (2024, 2025, 2026)
        for m in range(1, 13)
    }
    item = driver(
        "festival",
        {"festival": "Diwali", "dates": ["2024-10-31", "2025-11-12", "2026-10-25", "2027-11-02"]},
        families=("bottles",),
    )
    context = Context(
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        timezone="Asia/Kolkata",
        months=[date(2026, 10, 1), date(2026, 11, 1)],
        sources=SourcesConfig(),
        history={"bottles": history},
    )
    signals = calendar.fetch(item, context)
    shift = signals[0].data["in_horizon"]
    assert shift == [
        {"month": "2026-10", "date": "2026-10-25", "last_year": "2025-11-12", "moved": True}
    ]
    measured = next(s for s in signals if s.kind == "measured")
    assert measured.data["family_id"] == "bottles" and measured.data["uplift_pct"] > 30


def test_analog_uplift_removes_trend():
    history = {}
    for year, level in ((2024, 100.0), (2025, 110.0), (2026, 121.0)):
        for month in range(1, 13):
            lift = 1.2 if (year, month) == (2025, 4) else 1.0
            history[date(year, month, 1)] = level * lift
    uplift, used = analog_uplift(history, [date(2025, 4, 1)])
    assert used == 1 and uplift == pytest.approx(20.0)  # 20% lift, 10%/yr growth removed
    assert analog_uplift(history, [date(2023, 4, 1)]) is None


def test_fingerprint_ignores_small_weather_moves_but_not_status_changes():
    def weather_signal(anomaly):
        return Signal(
            driver_id="d",
            kind="weather",
            summary="w",
            data={
                "months": [
                    {
                        "period": "2027-03",
                        "coverage": "complete",
                        "temp_anomaly_c": anomaly,
                        "rain_mm": 5,
                        "rain_normal_mm": 5,
                    }
                ]
            },
        )

    assert fingerprint([weather_signal(1.1)]) == fingerprint([weather_signal(1.2)])
    assert fingerprint([weather_signal(1.1)]) != fingerprint([weather_signal(2.0)])
    status = [Signal(driver_id="d", kind="enso", summary="x", data={"status": "Watch"})]
    other = [Signal(driver_id="d", kind="enso", summary="y", data={"status": "Advisory"})]
    assert fingerprint(status) != fingerprint(other)


def test_profile_resolution_and_family_rules(tmp_path):
    profile = CompanyProfile.model_validate(PROFILE)
    assert assign_family(profile, "X", "JAR-20", "ZZ") == "jars"
    assert assign_family(profile, "X", "OTHER", "BTL") == "bottles"
    assert assign_family(profile, "X", "OTHER", "ZZ") == "unassigned"
    resin = driver("commodity_news", {"commodity": "PET resin"})
    assert resin.query == '"PET resin" price' and resin.error is None
    enso_driver = driver("enso", {}, families=("*",))
    assert "India" in enso_driver.query and enso_driver.families == ["bottles", "jars"]
    assert "missing params: topic" in driver("policy_news", {}).error
    assert driver("nope", {}).error.startswith("unknown template")
    assert {"enso", "festival", "commodity_news", "seasonal_weather"} <= load_catalog().keys()


def test_onboarding_draft_is_checked_and_saved(tmp_path):
    client = make_client(tmp_path)
    store, _ = seed_store(tmp_path)
    draft = ProfileDraftOut.model_validate(
        {
            "company_name": "TestWater",
            "summary": "Water.",
            "questions": ["What is JR?"],
            "regions": [{"id": "r1", "name": "R", "country": "IN", "latitude": 1, "longitude": 2}],
            "families": [
                {
                    "id": "Bottled Water",
                    "name": "Bottles",
                    "description": "",
                    "item_groups": ["BTL", "GHOST"],
                    "inputs": ["PET"],
                    "regions": ["r1"],
                    "customers": "",
                }
            ],
            "drivers": [
                {
                    "id": "El Nino",
                    "template": "enso",
                    "families": ["Bottled Water"],
                    "mechanism": "Hot.",
                    "params": [{"name": "analog_periods", "value": "2025-04, 2025-05"}],
                },
                {
                    "id": "made up",
                    "template": "astrology",
                    "families": [],
                    "mechanism": "",
                    "params": [],
                },
            ],
        }
    )

    class Model:
        provider = "openai"
        prompt = ""

        def generate(self, **kwargs):
            from engine.reason.llm.base import LLMResult
            from engine.reason.schemas import Usage

            Model.prompt = kwargs["prompt"]
            return LLMResult(
                data=draft,
                text=draft.model_dump_json(),
                usage=Usage(),
                provider="openai",
                model="m",
                request_hash="h",
            )

    profile, _ = draft_profile(client, store, Model(), "m", 1000, about="We bottle water.")
    assert "BTL" in Model.prompt and "We bottle water." in Model.prompt and "enso" in Model.prompt
    assert [f.id for f in profile.families] == ["bottled_water"]
    assert profile.families[0].match.item_group == ["BTL"]
    assert [d.id for d in profile.drivers] == ["el_nino"]
    assert profile.drivers[0].params["analog_periods"] == ["2025-04", "2025-05"]
    assert "GHOST" in profile.notes and "astrology" in profile.notes
    path = save_profile(client, store, profile, "admin")
    assert path.exists() and load_profile(client).open_questions == ["What is JR?"]
    assert from_draft(draft, load_catalog(), {"BTL"}).status == "draft"
