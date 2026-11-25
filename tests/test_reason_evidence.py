import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from test_reason_context import FEB, JAN, forecasts, seed
from typer.testing import CliRunner

from engine import cli
from engine.config.loader import load_client
from engine.config.schema import ClientProfile
from engine.reason import cli as reason_cli
from engine.reason import collect as collection
from engine.reason.collect import (
    collect_evidence,
    load_evidence,
    load_evidence_batch,
    load_source_check,
    lookup_scope,
    make_query_scope,
)
from engine.reason.context import load_forecast_snapshot
from engine.reason.facts import sync_context
from engine.reason.providers import HttpResponse
from engine.reason.schemas import ContextError, QueryRequestContext
from engine.reason.weather import parse_enso, parse_seasonal
from engine.store import EngineStore

FIXTURES = Path(__file__).parent / "fixtures" / "reasoning"


def _profile_data(tmp_path):
    (tmp_path / "fieldtest.yaml").write_bytes((FIXTURES / "phase2.yaml").read_bytes())
    rows = {
        "sites": "id,item,site,start\ns1,I1,A,2026-01-01\ns2,I2,A,2026-01-01\ns3,I3,B,2026-01-01\n",
        "bom": (
            "id,item,material,site,uom,output_uom,qty,at\n"
            "1,I1,M,A,KG,EA,1,2026-01-01\n2,I2,M,A,KG,EA,1,2026-01-01\n"
            "3,I3,M,B,KG,EA,1,2026-01-01\n"
        ),
        "stock": "id,material,site,uom,qty,at\na,M,A,KG,30,2027-01-01\nb,M,B,KG,100,2027-01-01\n",
        "consumption": (
            "id,material,site,uom,qty,at\n"
            "c1,M,A,KG,20,2027-01-04\nc2,M,A,KG,20,2027-01-09\n"
            "c3,M,B,KG,40,2027-01-09\n"
        ),
        "receipts": (
            "id,material,site,uom,qty,at,release,revision,status\n"
            "r,M,A,KG,10,2027-01-15,2026-01-01,1,confirmed\n"
            "r,M,A,KG,10,2027-01-05,2099-01-01,2,confirmed\n"
        ),
    }
    for name, value in rows.items():
        (tmp_path / f"{name}.csv").write_text(value, encoding="utf-8")
    return load_client("fieldtest", tmp_path)


def _forecast(store, count=3):
    points = pl.concat([forecasts(series=f"S{i}") for i in range(1, count + 1)])
    run = seed(store, points, unit="EA")
    with store.connect() as con:
        con.executemany(
            "INSERT INTO dim_series (client_id, series_id, customer, item_code, uom, "
            "source_file, ingested_at) VALUES (?, ?, 'C1', ?, 'EA', 'erp', current_timestamp)",
            [
                (store.client_id, f"S{i}", f"I{min(i, 2) if i != 3 else 3}")
                for i in range(2, count + 1)
            ],
        )
    return run


def _weather_payload(first, days):
    times = [int((first + timedelta(hours=6 * i)).timestamp()) for i in range(4 * days + 1)]
    return {
        "utc_offset_seconds": 0,
        "latitude": 19.0,
        "longitude": 72.8,
        "hourly_units": {"temperature_2m": "°C", "precipitation": "mm"},
        "hourly": {
            "time": times,
            "temperature_2m": [22] * len(times),
            "precipitation": [1] * len(times),
        },
    }


def _json(payload, status=200):
    return HttpResponse(status, json.dumps(payload).encode())


def _article(url, excerpt, *, title="Coastal port delay", event="port-event"):
    return {
        "url": url,
        "title": title,
        "description": excerpt,
        "event_id": event,
        "profile": {"name": url.split("/")[2]},
    }


def test_cli_freezes_evidence_and_reopens(tmp_path, monkeypatch):
    profile = _profile_data(tmp_path)
    store = EngineStore(profile.client_id, root=tmp_path / "stores")
    run = _forecast(store, 12)
    sync_context(profile, store)
    snapshot = load_forecast_snapshot(profile, store, run)
    monkeypatch.setattr(cli, "_load", lambda _: (profile, store))
    monkeypatch.setenv("PHASE2_TEST_BRAVE", "fixture")
    request_at = collection.seasonal_request
    monkeypatch.setattr(
        collection,
        "seasonal_request",
        lambda config, scope, now: request_at(config, scope, datetime(2026, 11, 24, tzinfo=UTC)),
    )
    requests = Counter()

    def mock(request, timeout):
        assert timeout <= 30
        if "brave" in request.url:
            term = request.params["q"]
            requests[term] += 1
            if term == "industrial flax":
                return _json({"results": []})
            assert term == "coastal trade"
            return _json(
                {
                    "results": [
                        _article(
                            "https://port.example/advisory?utm_source=feed", "Official port notice"
                        ),
                        _article("https://port.example/advisory#note", "Official port notice"),
                        _article("https://news.example/dispatch", "An independent dispatch"),
                    ],
                    "query": {"more_results_available": False},
                }
            )
        if "open-meteo" in request.url:
            requests["weather"] += 1
            assert request.params["hourly"] == "temperature_2m,precipitation"
            assert request.params["timezone"] == "GMT"
            return _json(_weather_payload(datetime(2026, 12, 31, tzinfo=UTC), 8))
        requests["enso"] += 1
        return HttpResponse(200, (FIXTURES / "noaa_enso.html").read_bytes())

    original = reason_cli.collect_evidence
    monkeypatch.setattr(
        reason_cli, "collect_evidence", lambda *args, **kw: original(*args, **kw, http=mock)
    )
    argv = ["reason-collect", "--client", profile.client_id, "--run-id", str(run), "--json"]
    result = CliRunner().invoke(cli.app, [*argv, "--snapshot-id", snapshot.snapshot_id])
    assert result.exit_code == 0, (result.output, result.exception)
    output = json.loads(result.stdout)
    assert requests == {"industrial flax": 1, "coastal trade": 1, "weather": 1, "enso": 1}
    reopened = EngineStore(profile.client_id, root=tmp_path / "stores")
    batch = load_evidence_batch(reopened, output["batch_id"])
    assert output["check_count"] == len(batch.check_ids)
    assert output["evidence_count"] == len(batch.evidence_ids)
    assert batch.snapshot_id == snapshot.snapshot_id
    checks = [load_source_check(reopened, item) for item in batch.check_ids]
    assert any(item.status == "not_applicable" for item in checks)
    assert any(item.status == "no_match" for item in checks)
    outlook = next(
        item
        for item in checks
        if item.scope.source_id == "A" and item.scope.source_kind == "weather"
    )
    assert outlook.coverage == "partial"
    assert [(m.period, m.status) for m in outlook.months] == [(JAN, "partial"), (FEB, "none")]
    assert len(batch.candidates) == 2
    assert all(
        item.basis == "broad_search_unresolved" and item.corroborating_ids
        for item in batch.candidates
    )
    evidence = load_evidence(reopened, batch.evidence_ids)
    assert len(batch.evidence_ids) == len({item.evidence_id for item in evidence})
    assert all(item.first_known_at <= batch.knowledge_cutoff for item in evidence)
    assert all(item.source_version != "2" for item in evidence if item.operational)
    assert any(item.news and item.news.primary and not item.issued_at for item in evidence)
    group = next(group for group in batch.groups if group.kind == "supply" and group.period == JAN)
    assert group.shared_shortfall == 10
    assert group.event_at.isoformat() == "2027-01-08T18:30:00+00:00"
    assert {item.series_id for item in group.members} == {f"S{i}" for i in range(1, 13)} - {"S3"}
    assert all(item.shortfall is None for item in group.members)
    assert [item.series_id for item in group.exceptions] == ["S3"]
    assert not any(group.kind == "supply" and group.location_id == "B" for group in batch.groups)
    result = CliRunner().invoke(cli.app, [*argv, "--resume-id", output["collection_id"]])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == output
    assert requests == {"industrial flax": 1, "coastal trade": 1, "weather": 1, "enso": 1}


def test_historical_reads_and_revised_receipt(tmp_path):
    profile = _profile_data(tmp_path)
    # Exercise the operational ledger without external feeds.
    raw = profile.model_dump(mode="python", exclude_none=True)
    for key in ("news", "weather", "enso"):
        raw["reasoning"]["evidence"].pop(key)
    raw["reasoning"]["sources"] = ()
    profile2 = ClientProfile.model_validate(raw)
    profile2._profile_dir = tmp_path
    store = EngineStore(profile.client_id, root=tmp_path / "stores")
    run = _forecast(store)
    first = collect_evidence(profile2, store, run)
    assert any(group.shared_shortfall == 10 for group in first.groups)
    path = tmp_path / "receipts.csv"
    # The same receipt is rescheduled to a date earlier than the second use.
    with path.open("a", encoding="utf-8") as file:
        file.write("r,M,A,KG,10,2027-01-05,2026-01-02,3,confirmed\n")
    second = collect_evidence(profile2, store, run)
    assert not second.groups
    revised = [e for e in load_evidence(store, second.evidence_ids) if e.source_version == "3"]
    assert len(revised) == 1 and revised[0].revision > 1
    historic = collect_evidence(profile2, store, run, as_of=first.knowledge_cutoff)
    assert historic.evidence_ids == first.evidence_ids
    assert historic.groups == first.groups
    assert all(usage.attempts == 0 for usage in historic.usage)
    path = tmp_path / "consumption.csv"
    with path.open("a", encoding="utf-8") as file:
        file.write("bad,M,A,LB,2,2027-01-06\n")
    mismatch = collect_evidence(profile2, store, run)
    assert any(
        "incompatible_unit" in group.flags and group.shared_shortfall is None
        for group in mismatch.groups
    )


def test_seasonal_leap_year_ensemble_and_enso():
    profile = ClientProfile(
        client_id="WX",
        display_name="Seasonal fixture",
        reasoning={
            "enabled": True,
            "timezone": "Asia/Kolkata",
            "evidence": {
                "weather": {
                    "ensemble_members": True,
                    "locations": [
                        {"location_id": "A", "role": "site", "latitude": 19, "longitude": 72.8},
                    ],
                },
                "enso": {},
            },
        },
    )
    months = (date(2027, 12, 1), date(2028, 1, 1), date(2028, 2, 1), date(2028, 4, 1))
    scope = make_query_scope(profile, "weather", "A", months, location_role="site")
    payload = _weather_payload(datetime(2027, 11, 30, tzinfo=UTC), 92)
    weather = profile.reasoning.evidence.weather
    size = len(payload["hourly"]["time"])
    payload["hourly_units"]["precipitation"] = "mm/h"
    payload["hourly"]["precipitation"] = [2] * size
    for index in range(1, 51):
        payload["hourly"][f"temperature_2m_member{index:02d}"] = [22 + index] * size
        payload["hourly"][f"precipitation_member{index:02d}"] = [2 + index] * size
    parsed = parse_seasonal(payload, weather, scope, datetime(2027, 11, 25, tzinfo=UTC))
    assert [month.status for month in parsed.months] == ["complete", "complete", "complete", "none"]
    for result in parsed.revisions:
        monthly = result.weather
        days = 29 if monthly.period.month == 2 else 31
        assert monthly.expected_seconds == days * 24 * 3600
        assert monthly.expected_members == len(monthly.members) == 51
        if monthly.variable == "precipitation":
            assert result.quantity == pytest.approx(27 * 24 * days)
            assert monthly.minimum == pytest.approx(2 * 24 * days)
        else:
            assert result.quantity == pytest.approx(47)
            assert (monthly.minimum, monthly.maximum) == (22, 72)
        assert monthly.anomaly is None and monthly.climatology is None
    enso_scope = make_query_scope(profile, "enso", "noaa_cpc", (JAN, FEB))
    advisory = parse_enso(
        (FIXTURES / "noaa_enso.html").read_bytes(),
        profile.reasoning.evidence.enso,
        enso_scope,
        datetime(2026, 11, 24, tzinfo=UTC),
    ).revisions[0]
    assert advisory.enso.issue_date == date(2026, 11, 12)
    assert advisory.enso.seasons[0].start == date(2026, 12, 1)
    assert advisory.enso.seasons[0].end == date(2027, 3, 1)
    assert "60%" in advisory.enso.seasons[0].statements[0]
    assert "72%" in advisory.enso.seasons[1].statements[0]
    assert advisory.quantity is None


def test_followup_budget_pagination_and_source_validation(tmp_path, monkeypatch):
    profile = _profile_data(tmp_path)
    store = EngineStore(profile.client_id, root=tmp_path / "stores")
    run = seed(store, unit="EA")
    snapshot = load_forecast_snapshot(profile, store, run)
    monkeypatch.setenv("PHASE2_TEST_BRAVE", "fixture")
    context = QueryRequestContext(
        client_id=store.client_id,
        forecast_run_id=run,
        snapshot_id=snapshot.snapshot_id,
        months=(JAN, FEB),
        review_id="review-fixture",
    )
    query = make_query_scope(profile, "news", "market", (JAN,))
    calls = []

    def mock(request, timeout):
        calls.append(request.params["offset"])
        if len(calls) == 1:
            raise TimeoutError()
        if request.params["offset"] == "1":
            return _json({"results": []})
        return _json(
            {
                "results": [_article("https://port.example/t", "Notice")],
                "query": {"more_results_available": True},
            }
        )

    with pytest.raises(ContextError, match="not configured"):
        lookup_scope(
            profile,
            store,
            query.model_copy(update={"source_kind": "erp", "source_id": "private.view"}),
            context,
            http=mock,
        )
    with pytest.raises(ContextError, match="public parameters"):
        lookup_scope(
            profile,
            store,
            query.model_copy(update={"public_aliases": ("private item",)}),
            context,
            http=mock,
        )
    failed = lookup_scope(profile, store, query, context, http=mock)
    assert failed.attempts_used == 1 and failed.checks[0].status == "unavailable"
    retry = lookup_scope(profile, store, query, context, http=mock)
    assert retry.attempts_used == 2 and retry.checks[0].retries == 1
    cache = lookup_scope(profile, store, query, context, http=mock)
    assert cache.attempts_used == 2 and cache.checks[0].cache_hits == 1
    other_review = context.model_copy(update={"review_id": "other-review"})
    fresh_cache = lookup_scope(profile, store, query, other_review, http=mock)
    assert fresh_cache.attempts_used == 0 and fresh_cache.evidence_ids == retry.evidence_ids
    second = make_query_scope(profile, "news", "market", (JAN,), page=1)
    paged = lookup_scope(profile, store, second, context, http=mock)
    assert paged.attempts_used == 3 and paged.checks[0].status == "no_match"
    with pytest.raises(ContextError, match="not configured"):
        make_query_scope(profile, "news", "market", (JAN,), page=2)
    cap = lookup_scope(
        profile, store, make_query_scope(profile, "news", "market", (FEB,)), context, http=mock
    )
    assert cap.checks[0].reason == "request_budget_exhausted"
    assert calls == ["0", "0", "1"]


def test_resume_after_interrupt_and_base_page_cap(tmp_path, monkeypatch):
    profile = ClientProfile(
        client_id="resume",
        display_name="Resume fixture",
        reasoning={
            "enabled": True,
            "evidence": {
                "news": {
                    "api_key_env": "PHASE2_TEST_BRAVE",
                    "page_size": 1,
                    "max_pages": 2,
                    "retries": 1,
                    "searches": [
                        {"id": "port", "public_alias": "coastal trade"},
                        {"id": "market", "public_alias": "industrial flax"},
                    ],
                }
            },
        },
    )
    monkeypatch.setenv("PHASE2_TEST_BRAVE", "fixture")
    monkeypatch.setattr(collection, "_LEASE", timedelta(0))
    store = EngineStore(profile.client_id, root=tmp_path)
    run = seed(store, unit="EA")
    calls, ids = [], []

    def mock(request, _):
        key = (request.params["q"], request.params["offset"])
        calls.append(key)
        if calls.count(key) == 1:
            if key[0] == "industrial flax":
                raise KeyboardInterrupt
            if key[1] == "0":
                return _json({}, 503)
        if key[0] == "industrial flax":
            return _json({"results": []})
        return _json(
            {
                "results": [_article(f"https://port.example/p{key[1]}", "Port notice")],
                "query": {"more_results_available": True},
            }
        )

    with pytest.raises(KeyboardInterrupt):
        collect_evidence(profile, store, run, http=mock, on_start=ids.append)
    batch = collect_evidence(profile, store, run, resume_id=ids[0], http=mock)
    assert calls == [
        ("coastal trade", "0"),
        ("coastal trade", "0"),
        ("coastal trade", "1"),
        ("industrial flax", "0"),
        ("industrial flax", "0"),
    ]
    assert batch.usage[0].attempts == len(calls)
    assert batch.usage[0].retries == 2
    checks = [load_source_check(store, key) for key in batch.check_ids]
    assert any(check.provider_status == "503" and check.retryable for check in checks)
    assert any(
        check.reason == "page_cap_reached" and check.coverage == "partial" for check in checks
    )
    assert any(check.reason == "interrupted_attempt" for check in checks)
    assert len(batch.candidates) == 1 and batch.candidates[0].duplicate_ids


def test_readonly_operational_views_and_client_isolation(tmp_path, monkeypatch):
    def make_client(client_id):
        profile = ClientProfile(
            client_id=client_id,
            display_name="View fixture",
            reasoning={
                "enabled": True,
                "target": {
                    "id": "customer_orders",
                    "definition": "Units ordered each month",
                    "outcome_source": "order_lines",
                    "unit_field": "uom",
                },
                "evidence": {
                    "erp": [
                        {
                            "id": "amendments",
                            "kind": "postgres",
                            "dataset": "order_amendment",
                            "relation": "fixtures.order_amendments",
                            "dsn_env": "PHASE2_READ_DSN",
                            "max_rows": 3,
                            "fields": {
                                "client_id": "tenant",
                                "record_id": "id",
                                "customer_id": "account",
                                "item_id": "sku",
                                "quantity": "delta",
                                "unit": "uom",
                                "status": "state",
                                "baseline_includes": "included",
                                "effective_from": "target",
                                "version": "version",
                                "prior_version": "prior",
                                "condition": "condition",
                            },
                        }
                    ]
                },
            },
        )
        store = EngineStore(client_id, root=tmp_path)
        run = seed(store, unit="EA")
        snapshot = load_forecast_snapshot(profile, store, run)
        context = QueryRequestContext(
            client_id=client_id,
            forecast_run_id=run,
            snapshot_id=snapshot.snapshot_id,
            review_id="same-review",
            months=(JAN,),
        )
        return profile, store, context

    monkeypatch.setenv("PHASE2_READ_DSN", "fixture-connection")
    left, right = make_client("left"), make_client("right")
    captured = []

    def fetch(query, dsn, parameters, relation):
        assert dsn == "fixture-connection"
        assert relation == ("fixtures", "order_amendments")
        assert 'FROM "fixtures"."order_amendments" WHERE "tenant" = $1' in query
        assert query.endswith("LIMIT 4")
        captured.append(parameters)
        return [
            {
                "tenant": parameters[0],
                "id": "same-record",
                "account": "C1",
                "sku": "I1",
                "delta": -5,
                "uom": "EA",
                "state": "contingent",
                "included": False,
                "target": JAN,
                "version": "2",
                "prior": "1",
                "condition": "Option accepted",
            }
        ]

    left_scope = make_query_scope(left[0], "erp", "amendments", (JAN,))
    right_scope = make_query_scope(right[0], "erp", "amendments", (JAN,))
    left_result = lookup_scope(left[0], left[1], left_scope, left[2], fetch=fetch)
    right_result = lookup_scope(right[0], right[1], right_scope, right[2], fetch=fetch)
    assert left_result.attempts_used == right_result.attempts_used == 1
    assert left_result.evidence_ids != right_result.evidence_ids
    amendment = load_evidence(left[1], left_result.evidence_ids)[0]
    assert (amendment.quantity, amendment.operational.prior_version, amendment.source_version) == (
        -5,
        "1",
        "2",
    )
    assert amendment.operational.baseline_includes is False
    assert amendment.operational.movement == "reference"
    assert amendment.first_known_at == amendment.captured_at
    with pytest.raises(ContextError, match="found"):
        load_evidence(right[1], left_result.evidence_ids)
    with pytest.raises(ContextError, match="configured client"):
        lookup_scope(left[0], left[1], right_scope, left[2], fetch=fetch)
    assert captured == [("left",), ("right",)]

    def transient_fetch(*args):
        if len(captured) == 2:
            captured.append("timeout")
            raise TimeoutError
        return fetch(*args)

    batch = collect_evidence(left[0], left[1], left[2].forecast_run_id, fetch=transient_fetch)
    assert batch.usage[0].attempts == 2 and batch.usage[0].retries == 1
    assert [(group.kind, group.members[0].series_id) for group in batch.groups] == [
        ("customer_order", "S1")
    ]
