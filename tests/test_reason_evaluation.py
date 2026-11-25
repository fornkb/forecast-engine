"""Phase 4 replay, scoring, and operations acceptance."""

from __future__ import annotations

import csv
import json
import socket
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from itertools import count
from threading import Event
from unittest.mock import MagicMock

import duckdb
import polars as pl
import pytest
from test_reason_context import FEB, JAN, forecasts, seed
from test_reason_evidence import (
    FIXTURES,
    _article,
    _forecast,
    _json,
    _profile_data,
    _weather_payload,
)
from test_reason_review import _envelope, _invoke, _order_response, _packet, _result, _setup
from typer.testing import CliRunner

from engine import cli
from engine.config.loader import load_client
from engine.config.schema import CostModelConfig, OutputConfig
from engine.decide import build_policy
from engine.reason import admin, assistant, context, evaluation, facts, questions
from engine.reason import cli as reason_cli
from engine.reason import collect as collection
from engine.reason import review as review_service
from engine.reason.admin import resolve_review_question
from engine.reason.collect import (
    collect_evidence,
    load_evidence,
    load_evidence_batch,
    lookup_scope,
    make_query_scope,
)
from engine.reason.config import ReasoningConfig
from engine.reason.evaluation import EvaluationResult
from engine.reason.providers import HttpResponse
from engine.reason.replay import ReplayResult, replay_review
from engine.reason.report import ReviewReport, load_report
from engine.reason.review import run_review
from engine.reason.schemas import AssumptionInput, ContextError, QueryRequestContext, ReviewManifest
from engine.reason.usage import provider_usage
from engine.store import EngineStore, duckdb_store
from engine.store import publish as publishing


class _AfterOutcomes(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2027, 4, 1, tzinfo=UTC).astimezone(tz or UTC)


@pytest.fixture(autouse=True)
def _decision_clock(monkeypatch):
    """Keep the historical scoring cohorts stable when the wall calendar advances."""
    counter = count()

    class DateTimeType(type):
        def __instancecheck__(cls, value):
            return isinstance(value, datetime)

    class FixtureClock(datetime, metaclass=DateTimeType):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2026, 11, 24, 12, tzinfo=UTC) + timedelta(microseconds=next(counter))
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)

    for module in (admin, collection, context, duckdb_store, facts, questions, review_service):
        monkeypatch.setattr(module, "datetime", FixtureClock)


def _write_csv(path, rows):
    fields = tuple(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _actual(*, month=JAN, series="S1", actual=120, observed="2027-02-02T00:00:00Z", **extra):
    return dict(
        client_id="reviewtest",
        target="ordered_units",
        series_id=series,
        period=month,
        unit="EA",
        actual=actual,
        observed_at=observed,
        source_ref="actuals:monthly",
        **extra,
    )


def test_offline_replay_and_paired_error_oracle(tmp_path, monkeypatch):
    profile, store, run_id = _setup(
        tmp_path, monkeypatch, baseline=100, qty=10, option=20, with_other_month=True
    )
    batch = collect_evidence(profile, store, run_id)
    review = run_review(profile, store, run_id, batch_id=batch.batch_id, transport=_order_response)
    loaded = load_report(store, review.review_id)
    january = next(item for item in loaded.assessments if item.period == JAN)

    def no_network(*args, **kwargs):
        pytest.fail("Offline operation attempted to use a provider")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(assistant, "post_messages", no_network)
    replay = ReplayResult.model_validate_json(
        _invoke("reason-replay", "--review-id", review.review_id, "--json").stdout
    )
    assert replay.reproduced and replay.seal_verified, replay.issues
    assert replay.stored_assessments == replay.replayed_assessments == loaded.assessments
    outcome = _write_csv(
        tmp_path / "outcomes.csv",
        [
            _actual(
                planner_point=115,
                selected_at="2027-01-01T00:00:00+09:00",
                scenario_id=january.scenario_id,
                condition_id="exercise",
            ),
            _actual(month=FEB, observed="2027-03-03", actual=0),
            _actual(series="S2", actual=0),
        ],
    )
    monkeypatch.setattr(evaluation, "datetime", _AfterOutcomes)
    result = EvaluationResult.model_validate_json(
        _invoke(
            "reason-evaluate",
            "--review-id",
            review.review_id,
            "--outcomes",
            outcome,
            "--json",
        ).stdout
    )
    by_group = {
        item.comparison: item for item in result.point_comparisons if item.review_id is None
    }
    paired = by_group["planner_point_paired"]
    assert paired.n == 1 and paired.actual_sum == 120
    assert [(item.comparator, item.mae, item.wape) for item in paired.errors] == [
        ("baseline", 20, 20 / 120),
        ("rule_final", 10, 10 / 120),
        ("planner", 5, 5 / 120),
    ]
    assert by_group["baseline_vs_final_rule"].n == 3
    assert by_group["realized_condition_paired"].provenance == "evidence_bound"
    feb = next(
        item
        for item in result.point_comparisons
        if item.horizon == 2 and item.comparison == "baseline_vs_final_rule"
    )
    assert feb.actual_sum == 0 and all(value.wape is None for value in feb.errors)
    (bounds,) = result.conditional_ranges
    assert bounds.resolved == bounds.unresolved == 1
    assert bounds.condition_counts == {"exercise": 1} and bounds.conditional_mae == 10
    assert all(item.baseline == 100 for item in result.rows)
    assert [call.request_kind for call in review.model_calls] == ["initial", "initial"]


def test_replay_seal_and_missing_revision_keep_original_answer(tmp_path, monkeypatch):
    profile, store, run_id = _setup(tmp_path, monkeypatch, baseline=100, qty=10, option=20)
    review = run_review(profile, store, run_id, transport=_order_response)
    report = load_report(store, review.review_id)
    assert replay_review(store, review.review_id).reproduced
    unsupported = review.model_copy(
        update={"method_version": "obsolete/9", "seal": None, "approved_lookups": None}
    )
    broken = report.model_copy(update={"manifest": unsupported})
    assert "obsolete/9" in evaluation.verify_report(broken).issues[0]
    legacy = report.model_copy(
        update={"manifest": review.model_copy(update={"seal": None, "approved_lookups": None})}
    )
    assert evaluation.verify_report(legacy).reproduced
    assert not evaluation.verify_report(legacy).seal_verified
    record = report.evidence[0]
    with store.connect() as con:
        con.execute(
            "UPDATE reason_evidence_revisions SET evidence_data = ? WHERE evidence_id = ?",
            [record.model_copy(update={"quantity": 999.0}).model_dump_json(), record.evidence_id],
        )
    corrupted = replay_review(store, review.review_id)
    assert not corrupted.reproduced and record.evidence_id in " ".join(corrupted.issues)
    assert corrupted.stored_assessments == report.assessments
    with store.connect() as con:
        con.execute(
            "DELETE FROM reason_evidence_revisions WHERE evidence_id = ?", [record.evidence_id]
        )
    missing = replay_review(store, review.review_id)
    assert not missing.reproduced and record.evidence_id in " ".join(missing.issues)
    assert missing.stored_assessments == report.assessments


def test_evaluation_revision_keys_and_repeated_review_denominator(tmp_path, monkeypatch):
    profile, store, run_id = _setup(tmp_path, monkeypatch, baseline=100, qty=10, option=20)
    batch = collect_evidence(profile, store, run_id)
    first = run_review(profile, store, run_id, batch_id=batch.batch_id, transport=_order_response)
    second = run_review(
        profile, store, run_id, batch_id=batch.batch_id, transport=_order_response, new_review=True
    )
    monkeypatch.setattr(evaluation, "datetime", _AfterOutcomes)
    outcomes = _write_csv(
        tmp_path / "outcomes.csv",
        [
            _actual(actual=100, revision=1),
            _actual(actual=120, observed="2027-03-01", revision=2),
        ],
    )
    result = evaluation.evaluate_reviews(
        profile, store, (first.review_id, second.review_id), outcomes
    )
    assert result.repeated_rows_removed == 4
    aggregate = next(
        item
        for item in result.point_comparisons
        if item.review_id is None and item.comparison == "baseline_vs_final_rule"
    )
    assert aggregate.n == 1 and aggregate.actual_sum == 120
    assert {item.actual_revision for item in result.rows if item.actual is not None} == {2}
    old = evaluation.evaluate_reviews(
        profile, store, (first.review_id,), outcomes, as_of=datetime(2027, 2, 5, tzinfo=UTC)
    )
    assert {item.actual for item in old.rows if item.actual is not None} == {100}
    assert sum(item.attempts for item in result.aggregate_usage if item.stage == "collection") == 2
    _write_csv(outcomes, [_actual(), _actual()])
    with pytest.raises(ContextError, match="Duplicate outcome"):
        evaluation.evaluate_reviews(profile, store, (first.review_id,), outcomes)
    _write_csv(outcomes, [_actual(planner_point=115, selected_at="2027-02-05T00:00:00Z")])
    with pytest.raises(ContextError, match="precede first outcome"):
        evaluation.evaluate_reviews(profile, store, (first.review_id,), outcomes)
    _write_csv(outcomes, [_actual() | {"client_id": "other"}])
    with pytest.raises(ContextError, match="different client"):
        evaluation.evaluate_reviews(profile, store, (first.review_id,), outcomes)


def test_busy_draft_cannot_be_published_as_complete_and_resumes(tmp_path, monkeypatch):
    profile, store, run_id = _setup(
        tmp_path, monkeypatch, baseline=100, qty=10, option=20, with_other_month=True
    )
    batch = collect_evidence(profile, store, run_id)
    entering, release = Event(), Event()
    calls, ids = [], []

    def interrupted(body, key, timeout):
        packet = _packet(body)
        calls.append(packet["group"]["group_id"])
        if len(calls) == 2:
            entering.set()
            assert release.wait(10)
            raise KeyboardInterrupt
        return _order_response(body, key, timeout)

    original = store.load_forecasts(run_id)
    quantiles = store.load_forecast_quantiles(run_id)
    costs = CostModelConfig(
        holding_cost_per_unit=1,
        shortage_cost_per_unit=1,
        lead_time_periods=1,
        review_period_periods=1,
    )
    policy = build_policy(original, quantiles, costs)
    policy_run = store.start_run("decide", {"forecast_run_id": run_id})
    store.write_decision_policy(policy_run, policy)
    output = OutputConfig.model_validate({"postgres": {"dsn_env": "PHASE4_PUBLISH_STUB"}})
    publisher = profile.model_copy(update={"output": output})
    monkeypatch.setenv("PHASE4_PUBLISH_STUB", "fixture")
    monkeypatch.setattr(publishing, "_connect", lambda _: MagicMock())
    frames = []

    def capture(cursor, schema, table, frame, columns):
        frames.append((table, frame.select(columns).to_dicts()))
        return frame.height

    def publish_inputs():
        frames.clear()
        selected = store.latest_run_from("decide", "forecast_run_id", run_id)
        assert selected == policy_run
        publishing.publish_forecasts(store, publisher, run_id)
        publishing.publish_policy(store, publisher, selected, run_id)
        return list(frames)

    monkeypatch.setattr(publishing, "_ingest", capture)
    before = publish_inputs()
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(
            run_review,
            profile,
            store,
            run_id,
            batch_id=batch.batch_id,
            transport=interrupted,
            on_start=ids.append,
        )
        try:
            assert entering.wait(10)
            assert not replay_review(store, ids[0]).reproduced
            with pytest.raises(ContextError, match="already in progress"):
                run_review(profile, store, run_id, resume_id=ids[0], transport=interrupted)
            draft = load_report(store, ids[0])
            assert draft.manifest.state == "draft" and not draft.assessments
        finally:
            release.set()
        with pytest.raises(KeyboardInterrupt):
            task.result(timeout=10)
    assert store.load_forecasts(run_id).equals(original)
    assert store.load_forecast_quantiles(run_id).equals(quantiles)
    assert store.load_decision_policy(policy_run).select(policy.columns).equals(policy)
    assert publish_inputs() == before
    resumed = []

    def resumed_model(body, key, timeout):
        resumed.append(_packet(body)["group"]["group_id"])
        return _order_response(body, key, timeout)

    finished = run_review(profile, store, run_id, resume_id=ids[0], transport=resumed_model)
    assert resumed == [calls[1]]
    assert replay_review(store, finished.review_id).reproduced
    assert store.load_forecasts(run_id).equals(original)
    with pytest.raises(ContextError, match="not found"):
        replay_review(EngineStore("foreign", root=tmp_path), finished.review_id)


def _cli_for(client, command, *args):
    if command == "reason-context":
        command_path = [command, str(args[0])]
        args = args[1:]
    else:
        command_path = [command]
    result = CliRunner().invoke(cli.app, [*command_path, "--client", client, *map(str, args)])
    assert result.exit_code == 0, (result.output, result.exception)
    return result.stdout


def test_public_transfer_weather_cache_mapping_and_assumption(tmp_path, monkeypatch):
    plant_path, dist_path = tmp_path / "plant", tmp_path / "distributor"
    plant_path.mkdir()
    dist_path.mkdir()
    plant = _profile_data(plant_path)
    plant_config = plant.reasoning.model_dump(mode="python")
    plant_config["timezone"] = "UTC"
    plant_config["assistant"] = {"model_id": "test-model", "api_key_env": "TEST_REVIEW_MODEL"}
    plant_config["evaluation"] = {
        "rates": {
            "brave": {"rate_id": "fixture-brave", "currency": "USD", "per_request": 0.01},
            "anthropic": {
                "rate_id": "fixture-model",
                "currency": "USD",
                "per_million_input": 2,
                "per_million_output": 5,
                "per_million_cache_read": 1,
            },
        }
    }
    plant = plant.model_copy(update={"reasoning": ReasoningConfig.model_validate(plant_config)})
    store_one = EngineStore(plant.client_id, root=tmp_path)
    run_one = _forecast(store_one, count=10)
    dist = _distributor_profile(dist_path)
    store_two = EngineStore(dist.client_id, root=tmp_path)
    points = pl.concat([forecasts(series=name, baseline=1000) for name in ("S1", "S2")])
    run_two = seed(store_two, points, unit="CS")
    with store_two.connect() as con:
        con.execute(
            "INSERT INTO dim_series "
            "(client_id,series_id,customer,item_code,uom,source_file,ingested_at) "
            "VALUES (?, 'S2', 'C2', 'I1', 'CS', 'fixture', current_timestamp)",
            [dist.client_id],
        )
    assert run_one == run_two
    profiles = {plant.client_id: (plant, store_one), dist.client_id: (dist, store_two)}
    monkeypatch.setattr(cli, "_load", profiles.__getitem__)
    monkeypatch.setenv("TEST_REVIEW_MODEL", "synthetic")
    monkeypatch.setenv("PHASE2_TEST_BRAVE", "synthetic")
    monkeypatch.setenv("PHASE4_NEWS", "synthetic")
    external, assessments = [], []

    def http(request, timeout):
        if "search.brave" in request.url:
            query, offset = request.params["q"], request.params["offset"]
            external.append(("news", query, offset))
            if "carton" in query:
                if offset == "0":
                    article = _article(
                        "https://port.example/dist",
                        "Route R7 closed for carton freight",
                        event="route-r7",
                    )
                else:
                    article = _article(
                        "https://port.example/racing", "Race calendar", event="racing"
                    )
            elif "flax" in query:
                article = _article(
                    "https://port.example/flax", "Industrial flax is delayed", event="flax-delay"
                )
            else:
                return _json({"results": []})
            article.update(event_start="2026-12-31T00:00:00Z", event_end="2027-02-01T00:00:00Z")
            return _json({"results": [article]})
        if "open-meteo" in request.url:
            external.append(("weather", request.params["latitude"], request.params["longitude"]))
            payload = _weather_payload(datetime(2027, 1, 1, tzinfo=UTC), 31)
            payload["issue_time"] = "2026-11-15T00:00:00Z"
            return _json(payload)
        if "noaa" in request.url:
            external.append(("enso",))
            return HttpResponse(200, (FIXTURES / "noaa_enso.html").read_bytes())
        pytest.fail(f"Unexpected external request {request.url}")

    first_lookup = False

    def model(body, key, timeout):
        nonlocal first_lookup
        packet = _packet(body)
        if "group" in packet:
            group = packet["group"]
            assessments.append((group["group_id"], group["kind"], len(group["members"])))
            channel = (
                "material_availability"
                if group["kind"] == "supply"
                else ("production" if group["kind"] == "weather" else "demand")
            )
            return _result(packet, target_type=channel, direction="decrease")
        if (
            not first_lookup
            and "market" in packet["configured_lookups"].get("news", [])
            and any(
                "carton" in item["article"].get("excerpt", "")
                for event in packet["events"]
                for item in event["articles"]
            )
        ):
            first_lookup = True
            return _envelope(
                "lookup_scope",
                {
                    "source_kind": "news",
                    "source_id": "market",
                    "page": 1,
                    "months": [JAN.isoformat()],
                },
            )
        decisions = []
        for event in packet["events"]:
            articles = [item["article"] for item in event["articles"]]
            excerpt = articles[0].get("excerpt", "")
            entities = []
            evidence_ids = []
            if "flax" in excerpt:
                entities = [{"type": "material", "id": "M"}]
                evidence_ids = [
                    item["id"]
                    for item in packet["operating_links"]
                    if item["operational"]["material_id"] == "M"
                ]
            if "carton" in excerpt:
                entities = [{"type": "route", "id": "R7"}]
            decisions.append(
                {
                    "event_id": event["event_id"],
                    "article_ids": [item["id"] for item in articles],
                    "entities": entities,
                    "months": [JAN.isoformat()] if entities else [],
                    "evidence_ids": evidence_ids,
                    "status": "established" if entities else "irrelevant",
                    "basis": "Dated shipping report" if entities else "Unrelated sporting event",
                }
            )
        return _envelope("submit_result", {"decisions": decisions})

    monkeypatch.setattr(assistant, "post_messages", model)
    monkeypatch.setattr(review_service, "http_get", http)
    monkeypatch.setattr(
        reason_cli, "collect_evidence", lambda *args, **kw: collect_evidence(*args, **kw, http=http)
    )
    for profile, run in ((plant, run_one), (dist, run_two)):
        _cli_for(profile.client_id, "reason-context", "sync")
        _cli_for(profile.client_id, "reason-context", "inspect", "--run-id", run, "--json")
    plant_batch = json.loads(
        _cli_for(plant.client_id, "reason-collect", "--run-id", run_one, "--json")
    )
    plant_review = ReviewManifest.model_validate_json(
        _cli_for(
            plant.client_id,
            "reason",
            "--run-id",
            run_one,
            "--batch-id",
            plant_batch["batch_id"],
            "--json",
        )
    )
    plant_report = load_report(store_one, plant_review.review_id)
    plant_base_frozen = plant_report.base_batch.model_dump_json()
    monthly = [
        month
        for check in plant_report.source_checks
        if check.scope.source_kind == "weather"
        for month in check.months
    ]
    assert any(month.period == JAN and month.status == "complete" for month in monthly)
    assert any(month.period == FEB and month.status == "none" for month in monthly)
    assert any(
        item.weather and item.issued_at == datetime(2026, 11, 15, tzinfo=UTC)
        for item in plant_report.evidence
    )
    shared = [
        group
        for group in plant_review.groups
        if group.group.kind == "news" and group.group.dependency.id == "M"
    ]
    assert len(shared) == 1 and len(shared[0].group.members) == 10
    assert Counter(kind for key, kind, count in assessments if key == shared[0].group.group_id) == {
        "news": 1
    }
    plant_calls = len(external)
    repeated = ReviewManifest.model_validate_json(
        _cli_for(
            plant.client_id,
            "reason",
            "--run-id",
            run_one,
            "--batch-id",
            plant_batch["batch_id"],
            "--json",
        )
    )
    assert repeated == plant_review and len(external) == plant_calls
    path = plant_path / "receipts.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "2027-01-15,2026-01-01,1", "2027-01-06,2026-11-02,3"
        ),
        encoding="utf-8",
    )
    altered = json.loads(_cli_for(plant.client_id, "reason-collect", "--run-id", run_one, "--json"))
    changed = load_evidence_batch(store_one, altered["batch_id"])
    assert changed.batch_id != plant_report.base_batch.batch_id
    assert tuple(changed.evidence_ids) != tuple(plant_report.base_batch.evidence_ids)
    old_revisions = {
        item.record_id: item for item in plant_report.evidence if item.source_id == "incoming"
    }
    (corrected,) = [
        item
        for item in load_evidence(store_one, changed.evidence_ids)
        if item.source_id == "incoming" and item.source_version == "3"
    ]
    assert corrected.revision > old_revisions[corrected.record_id].revision
    assert [item[0] for item in external[plant_calls:]] == ["weather"]
    assert (
        load_evidence_batch(store_one, plant_batch["batch_id"]).model_dump_json()
        == plant_base_frozen
    )
    assert replay_review(store_one, plant_review.review_id).reproduced

    dist_batch = json.loads(
        _cli_for(dist.client_id, "reason-collect", "--run-id", run_two, "--json")
    )
    dist_review = ReviewManifest.model_validate_json(
        _cli_for(
            dist.client_id,
            "reason",
            "--run-id",
            run_two,
            "--batch-id",
            dist_batch["batch_id"],
            "--json",
        )
    )
    assert len(dist_review.lookup_attempts) == 1
    report = load_report(store_two, dist_review.review_id)
    assert any(
        check.scope.source_kind == "weather" and check.status == "not_applicable"
        for check in report.source_checks
    )
    mapping = next(item for item in dist_review.questions if item.purpose == "mapping_confirmation")
    mapped = ReviewReport.model_validate_json(
        _cli_for(
            dist.client_id,
            "reason-admin",
            "--review-id",
            dist_review.review_id,
            "--action",
            "confirm",
            "--question-id",
            mapping.question_id,
            "--value",
            "S1",
            "--actor",
            "Route planner",
            "--json",
        )
    )
    route_groups = [item.group for item in mapped.manifest.groups if item.group.kind == "news"]
    (route_group,) = route_groups
    assert [item.series_id for item in route_group.members] == ["S1"]
    assert [item.series_id for item in route_group.exceptions] == ["S2"]
    question = next(
        item
        for item in mapped.manifest.questions
        if item.scope and item.scope.group_id == route_group.group_id
    )
    approved = ReviewReport.model_validate_json(
        _cli_for(
            dist.client_id,
            "reason-admin",
            "--review-id",
            mapped.manifest.review_id,
            "--action",
            "confirm",
            "--question-id",
            question.question_id,
            "--actor",
            "Route planner",
            "--method",
            "proportional_change",
            "--unit",
            "CS",
            "--series",
            "S1",
            "--low",
            5,
            "--high",
            12,
            "--low-condition",
            "Small effect",
            "--high-condition",
            "Full effect",
            "--rationale",
            "Expected extra carton orders",
            "--json",
        )
    )
    row = next(item for item in approved.assessments if item.group_id == route_group.group_id)
    assert [item.endpoint for item in row.conditions] == [1050, 1120]
    assert approved.manifest.model_calls == mapped.manifest.model_calls
    assert approved.manifest.lookup_attempts == dist_review.lookup_attempts
    replay = ReplayResult.model_validate_json(
        _cli_for(
            dist.client_id,
            "reason-replay",
            "--review-id",
            approved.manifest.review_id,
            "--json",
        )
    )
    assert replay.reproduced and replay.seal_verified
    assert replay_review(store_two, dist_review.review_id).reproduced
    assert "1120 CS" in _cli_for(
        dist.client_id, "reason-report", "--review-id", approved.manifest.review_id, "--markdown"
    )
    dist_outcomes = _write_csv(
        tmp_path / "cartons.csv",
        [
            {
                "tenant": dist.client_id,
                "series_id": "S1",
                "period": JAN,
                "target": "cartons_ordered",
                "unit": "CS",
                "actual": 1110,
                "observed_at": "2027-02-02",
                "source_ref": "orders:jan",
                "scenario_id": row.scenario_id,
                "condition_id": row.conditions[1].condition_id,
            },
        ],
    )
    triage_label_path = _write_csv(
        tmp_path / "triage.csv",
        [
            {
                "client_id": dist.client_id,
                "review_id": approved.manifest.review_id,
                "article_id": next(
                    d.article_ids[0]
                    for d in approved.manifest.mappings
                    if d.status == "established"
                ),
                "label": "relevant",
                "entity_type": "route",
                "entity_id": "R7",
                "period": JAN,
                "series_id": "S1",
            },
        ],
    )
    groups_path = _write_csv(
        tmp_path / "groups.csv",
        [
            {
                "client_id": dist.client_id,
                "review_id": approved.manifest.review_id,
                "group_id": route_group.group_id,
                "label": "affected",
            },
            {
                "client_id": dist.client_id,
                "review_id": approved.manifest.review_id,
                "group_id": "independent-quiet-group",
                "label": "unaffected",
            },
        ],
    )
    monkeypatch.setattr(evaluation, "datetime", _AfterOutcomes)
    outcome = EvaluationResult.model_validate_json(
        _cli_for(
            dist.client_id,
            "reason-evaluate",
            "--review-id",
            approved.manifest.review_id,
            "--outcomes",
            dist_outcomes,
            "--triage-labels",
            triage_label_path,
            "--group-labels",
            groups_path,
            "--json",
        )
    )
    assert outcome.labels.matched_associations == 1 and outcome.labels.incorrect_product_links == 0
    assert outcome.labels.group_results == {"affected_found": 1, "unaffected_quiet": 1}
    assert any(
        item.provenance == "planner_assumption" and item.conditional_mae == 10
        for item in outcome.conditional_ranges
    )
    assert len(report.manifest.lookup_attempts) <= report.manifest.followup_limit
    assert plant_review.client_id != approved.manifest.client_id
    with pytest.raises(ContextError, match="not found"):
        replay_review(store_one, approved.manifest.review_id)
    with pytest.raises(ContextError, match="one client"):
        provider_usage((plant_report, approved))
    plant_usage = provider_usage((plant_report,))
    assert any(item.provider == "brave" and item.estimated_cost is not None for item in plant_usage)
    assert all(item.estimated_cost is None for item in outcome.aggregate_usage)
    print(
        "TRANSFER COUNTS",
        {
            "manufacturer_assistant": len(plant_review.model_calls),
            "manufacturer_external": plant_calls,
            "shared_notice_members": len(shared[0].group.members),
            "distributor_assistant": len(approved.manifest.model_calls),
            "distributor_followups": len(approved.manifest.lookup_attempts),
        },
    )


def _distributor_profile(path):
    (path / "markets.csv").write_text(
        "map_id,buyer,region,month_start\nmap,C1,Zone-East,2026-01-01\n", encoding="utf-8"
    )
    (path / "vendors.csv").write_text(
        "link,product,buyer,stockcode,warehouse,pack,seller,lane,start\n"
        "v1,I1,C1,I1,D1,CS,KnownVendor,R8,2026-01-01\n",
        encoding="utf-8",
    )
    profile = {
        "client_id": "distribution",
        "display_name": "Fixture distributor",
        "reasoning": {
            "enabled": True,
            "timezone": "Asia/Kolkata",
            "target": {
                "id": "cartons_ordered",
                "definition": "Carton orders",
                "outcome_source": "orders",
                "unit_field": "uom",
            },
            "assistant": {"model_id": "test-model", "api_key_env": "TEST_REVIEW_MODEL"},
            "sources": [
                {
                    "id": "markets",
                    "kind": "csv",
                    "dataset": "relationship",
                    "relation_type": "serves",
                    "subject_type": "customer",
                    "object_type": "location",
                    "location_role": "customer",
                    "path": "markets.csv",
                    "fields": {
                        "record_id": "map_id",
                        "subject_id": "buyer",
                        "object_id": "region",
                        "effective_from": "month_start",
                    },
                }
            ],
            "evidence": {
                "erp": [
                    {
                        "id": "formula",
                        "kind": "csv",
                        "dataset": "vendor_link",
                        "path": "vendors.csv",
                        "fields": {
                            "record_id": "link",
                            "item_id": "product",
                            "customer_id": "buyer",
                            "material_id": "stockcode",
                            "site_id": "warehouse",
                            "unit": "pack",
                            "supplier_id": "seller",
                            "route_id": "lane",
                            "effective_from": "start",
                        },
                    }
                ],
                "news": {
                    "api_key_env": "PHASE4_NEWS",
                    "max_pages": 2,
                    "page_size": 5,
                    "searches": [{"id": "market", "public_alias": "carton freight"}],
                },
                "weather": {"applicable_roles": [], "access": "noncommercial"},
            },
            "evaluation": {"outcome_columns": {"client_id": "tenant"}},
        },
    }
    (path / "distribution.yaml").write_text(json.dumps(profile), encoding="utf-8")
    return load_client("distribution", path)


def test_concurrent_followups_cannot_reserve_a_fourth_attempt(tmp_path, monkeypatch):
    profile, store, run_id = _setup(tmp_path, monkeypatch, news=True)
    batch = collect_evidence(profile, store, run_id, http=lambda *_: _json({"results": []}))
    identifiers = []

    def interrupted(*_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_review(
            profile,
            store,
            run_id,
            batch_id=batch.batch_id,
            transport=interrupted,
            on_start=identifiers.append,
        )
    context = QueryRequestContext(
        client_id=store.client_id,
        forecast_run_id=run_id,
        snapshot_id=batch.snapshot_id,
        review_id=identifiers[0],
        months=(JAN, FEB),
    )
    scopes = [
        make_query_scope(profile, "news", source, (JAN,), page=page)
        for source in ("routes", "markets")
        for page in (1, 2)
    ]
    sent = []

    def request(value, timeout):
        sent.append(value.params)
        return _json({"results": []})

    def dispatch(scope):
        return lookup_scope(profile, store, scope, context, http=request)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(dispatch, scopes))
    assert len(sent) == 3
    assert (
        sum(
            check.reason == "request_budget_exhausted"
            for result in results
            for check in result.checks
        )
        == 1
    )
    resumed = run_review(
        profile, store, run_id, resume_id=identifiers[0], transport=_order_response, http=request
    )
    assert len(resumed.lookup_attempts) == 3
    assert len(resumed.appended_check_ids) == 4
    assert load_evidence_batch(store, batch.batch_id) == batch
    assert replay_review(store, resumed.review_id).reproduced


def test_real_duckdb_lock_has_bounded_actionable_error(tmp_path):
    store = EngineStore("busy", root=tmp_path)
    store.init_schema()
    source = (
        "import duckdb, sys; "
        "con = duckdb.connect(sys.argv[1]); con.execute('SELECT 1'); "
        "print('ready', flush=True); sys.stdin.readline(); con.close()"
    )
    with subprocess.Popen(
        [sys.executable, "-c", source, str(store.path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            assert process.stdout.readline().strip() == "ready"
            with pytest.raises(duckdb.IOException, match="busy.*Retry"), store.connect():
                pytest.fail("Two writer processes acquired the database")
        finally:
            process.communicate(input="\n", timeout=10)
    with store.connect() as con:
        assert con.execute("SELECT 1").fetchone() == (1,)


def test_successor_is_issued_after_outcome_and_parent_stays_reproducible(tmp_path, monkeypatch):
    profile, store, run_id = _setup(tmp_path, monkeypatch, baseline=100, qty=10, option=20)

    def qualitative(body, *_):
        return _result(_packet(body))

    parent = run_review(profile, store, run_id, transport=qualitative)
    question = next(item for item in parent.questions if item.scope)
    with monkeypatch.context() as clock:
        clock.setattr(review_service, "datetime", _AfterOutcomes)
        clock.setattr(admin, "datetime", _AfterOutcomes)
        successor = resolve_review_question(
            profile,
            store,
            parent.review_id,
            question.question_id,
            "confirm",
            "Planner",
            assumption=AssumptionInput(
                method_id="proportional_change",
                low=5,
                high=20,
                low_condition="Some additional work",
                high_condition="All additional work",
                rationale="Review condition",
                unit="EA",
                series_ids=("S1",),
            ),
        )
    assert replay_review(store, parent.review_id).reproduced
    assert replay_review(store, successor.review_id).reproduced
    monkeypatch.setattr(evaluation, "datetime", _AfterOutcomes)
    outcomes = _write_csv(tmp_path / "outcomes.csv", [_actual()])
    result = evaluation.evaluate_reviews(
        profile, store, (parent.review_id, successor.review_id), outcomes
    )
    eligible = next(
        row
        for row in result.rows
        if row.review_id == parent.review_id and row.series_id == "S1" and row.period == JAN
    )
    late = next(
        row
        for row in result.rows
        if row.review_id == successor.review_id and row.series_id == "S1" and row.period == JAN
    )
    assert eligible.phase == "pre_period" and eligible.exclusion is None
    assert late.phase == "after_actual" and late.exclusion == "issued_after_outcome"
    assert late.decision_at == successor.knowledge_cutoff > parent.knowledge_cutoff
    assert successor.collection_cutoff == parent.collection_cutoff
