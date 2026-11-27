"""The web app drives the same reasoning functions as the CLI, end to end, offline."""

from __future__ import annotations

import json
import shutil
import threading

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from reason_kit import FakeModel, make_client, seed_store, write_pack  # noqa: E402

from engine.reason import store as records  # noqa: E402
from engine.reason.llm.base import LLMResult  # noqa: E402
from engine.reason.schemas import ProfileDraftOut, Usage  # noqa: E402
from engine.web.app import create_app  # noqa: E402
from engine.web.jobs import Busy, JobRunner  # noqa: E402
from engine.web.services import Workspace  # noqa: E402

PACK = "data/demo/scenarios/pack.json"


class DraftModel(FakeModel):
    def generate(self, *, system, prompt, schema, model, max_output_tokens, effort=None, name=""):
        if schema is not ProfileDraftOut:
            return super().generate(
                system=system,
                prompt=prompt,
                schema=schema,
                model=model,
                max_output_tokens=max_output_tokens,
                effort=effort,
                name=name,
            )
        self.prompts.append((name, prompt))
        data = ProfileDraftOut.model_validate(
            {
                "company_name": "TestWater",
                "summary": "Bottled water.",
                "scan_topics": ["bottled water"],
                "questions": ["Which month is your peak?"],
                "regions": [
                    {"id": "r1", "name": "R", "country": "IN", "latitude": 1, "longitude": 2}
                ],
                "families": [
                    {
                        "id": "bottles",
                        "name": "Bottles",
                        "description": "",
                        "item_groups": ["BTL"],
                        "inputs": [],
                        "regions": ["r1"],
                        "customers": "",
                    }
                ],
                "drivers": [
                    {
                        "id": "el_nino",
                        "template": "enso",
                        "families": ["bottles"],
                        "mechanism": "heat lifts demand",
                        "params": [],
                    }
                ],
            }
        )
        return LLMResult(
            data=data,
            text=data.model_dump_json(),
            usage=Usage(input_tokens=500, output_tokens=200),
            provider="openai",
            model=model,
            request_hash="onboardhash01",
        )


@pytest.fixture
def web(tmp_path):
    client = make_client(tmp_path)
    store, _ = seed_store(tmp_path)
    config = tmp_path / "config" / "clients"
    config.mkdir(parents=True, exist_ok=True)
    (config / "testwater.yaml").write_text(
        yaml.safe_dump(client.model_dump(mode="json", exclude_none=True)), encoding="utf-8"
    )
    scenarios = tmp_path / "data" / "demo" / "scenarios"
    scenarios.mkdir(parents=True)
    shutil.copy(write_pack(tmp_path), scenarios / "pack.json")
    model = DraftModel()
    app = create_app(
        Workspace(config_dir=config, store_root=tmp_path / "stores"),
        JobRunner(inline=True),
        llm_factory=lambda _client: model,
    )
    return TestClient(app), store, model


def _review(http, **form):
    return http.post("/c/testwater/review", data={"scenario": PACK, "action": "run", **form})


def test_portfolio_and_client_pages(web):
    http, _, _ = web
    home = http.get("/")
    assert home.status_code == 200 and "Portfolio" in home.text and "Test water" in home.text
    assert "Forecast not reviewed yet" in home.text  # attention feed
    summary = http.get("/c/testwater")
    assert "Demand outlook" in summary.text and "Next 2 months" in summary.text
    assert "Run first review" in summary.text
    forecast = http.get("/c/testwater/forecast")
    assert "Total demand" in forecast.text and "Largest SKUs" in forecast.text
    assert "BTL-1" in forecast.text and "Not measured yet" in forecast.text  # no backtest
    reviews = http.get("/c/testwater/reviews")
    assert "New review" in reviews.text and PACK in reviews.text
    settings = http.get("/c/testwater/settings")
    assert "Setup checklist" in settings.text and "Cost limits" in settings.text
    assert "AI this month" in settings.text  # cost is one line in settings, not its own page
    assert http.get("/usage").status_code == 404
    exported = http.get("/c/testwater/forecast.csv")
    assert exported.text.startswith("series_id,customer,item_code") and "BTL-1" in exported.text
    assert http.get("/c/nobody").status_code == 404
    assert http.get("/c/..%2Fsecrets").status_code == 404


def test_estimate_spends_nothing_and_saves_nothing(web):
    http, store, model = web
    page = _review(http, action="estimate")
    assert page.status_code == 200 and "Cost estimate" in page.text and "El Nino" in page.text
    assert model.prompts == [] and records.latest_run(store) is None


def test_review_then_decisions_and_downloads(web):
    http, store, model = web
    page = _review(http)
    assert page.status_code == 200 and "/runs/R" in str(page.url)
    assert [name for name, _ in model.prompts] == ["reason"]
    assert "Bottles" in page.text and "<svg" in page.text and "What-if band" in page.text
    assert "Scenario data" in page.text
    run = records.latest_run(store)
    base = f"/c/testwater/runs/{run.reason_id}"
    target = next(f for f in run.findings if f.driver_id == "el_nino")

    http.cookies.set("planner", "Asha")
    done = http.post(f"{base}/findings/{target.finding_id}", data={"action": "dismiss"})
    assert "dismissed" in done.text and "no AI call" in done.text
    saved = {f.finding_id: f for f in records.load_run(store, run.reason_id).findings}
    assert "dismissed" in saved[target.finding_id].flags
    bad = http.post(
        f"{base}/findings/{target.finding_id}", data={"action": "edit", "low": "5", "high": "1"}
    )
    assert "Low must not be above high" in bad.text
    edited = http.post(
        f"{base}/findings/{target.finding_id}", data={"action": "edit", "low": "1", "high": "2"}
    )
    assert "Edited by Asha" in edited.text
    assert len(records.call_ledger(store)) == 1  # decisions never call the model

    csv = http.get(f"{base}/download/csv")
    assert (
        csv.text.startswith("series_id,family_id,period")
        and "attachment" in csv.headers["content-disposition"]
    )
    assert "<html" in http.get(f"{base}/download/html").text
    assert http.get(f"{base}/download/json").json()["reason_id"] == run.reason_id
    assert http.get(f"{base}/download/exe").status_code == 404

    assert http.get("/c/testwater/cost").status_code == 404
    page = http.get(base)
    assert "could affect the forecast" in page.text and "1 AI call" in page.text

    home = http.get("/")
    assert "Changes found" in home.text and "Open decisions" in home.text
    summary = http.get("/c/testwater")
    assert "Latest review" in summary.text and "What-if band" in summary.text
    listing = http.get("/c/testwater/reviews")
    assert run.reason_id in listing.text and "decided" in listing.text


def test_bad_inputs_are_explained(web):
    http, _, _ = web
    page = http.post("/c/testwater/review", data={"scenario": "../../etc/passwd"})
    assert "Unknown scenario" in page.text
    page = _review(http, llm_mode="free")
    assert "Unknown AI mode" in page.text


def test_cross_site_posts_are_refused(web):
    http, _, model = web
    refused = http.post(
        "/c/testwater/review", data={"scenario": PACK}, headers={"Origin": "https://evil.example"}
    )
    assert refused.status_code == 403 and model.prompts == []
    allowed = http.post(
        "/c/testwater/review",
        data={"scenario": PACK, "action": "estimate"},
        headers={"Origin": "http://testserver"},
    )
    assert allowed.status_code == 200


def test_profile_edit_validates_and_versions(web):
    http, store, _ = web
    page = http.get("/c/testwater/profile")
    assert "Product families" in page.text and "el_nino" in page.text

    broken = http.post("/c/testwater/profile", data={"yaml_text": "company: [unclosed"})
    assert broken.status_code == 422 and "company: [unclosed" in broken.text

    current = (store.path.parents[2] / "config" / "profiles" / "testwater.yaml").read_text()
    edited = "# edited in the web app\n" + current.replace("status: confirmed", "status: draft")
    saved = http.post("/c/testwater/profile", data={"yaml_text": edited})
    assert "Profile saved" in saved.text and "Mark confirmed" in saved.text
    confirmed = http.post("/c/testwater/profile/confirm")
    assert "Profile confirmed" in confirmed.text
    disk = (store.path.parents[2] / "config" / "profiles" / "testwater.yaml").read_text()
    assert disk.startswith("# edited in the web app") and "status: confirmed" in disk
    assert len(records.profile_versions(store)) == 2


def test_onboarding_drafts_profile_and_asks_questions(web):
    http, store, model = web
    refused = http.post("/c/testwater/onboard", data={"about": "We bottle water."})
    assert "A profile exists" in refused.text and model.prompts == []
    page = http.post("/c/testwater/onboard", data={"about": "We bottle water.", "replace": "true"})
    assert "Saved a draft profile" in page.text and "Which month is your peak?" in page.text
    assert [name for name, _ in model.prompts] == ["onboard"]
    assert records.call_ledger(store)[0]["stage"] == "onboard"


def test_one_job_per_client_at_a_time():
    runner = JobRunner()
    gate = threading.Event()
    first = runner.submit("c", "review", gate.wait)
    with pytest.raises(Busy):
        runner.submit("c", "estimate", lambda: None)
    runner.submit("other", "estimate", lambda: None)
    gate.set()
    runner._pool.shutdown(wait=True)
    assert first.status == "done"


def test_preview_mode_uses_sample_answers_and_keeps_profiles_read_only(tmp_path):
    from engine.web.preview import MODEL, PreviewWorkspace

    client = make_client(tmp_path)
    store, _ = seed_store(tmp_path)
    config = tmp_path / "config" / "clients"
    config.mkdir(parents=True, exist_ok=True)
    (config / "testwater.yaml").write_text(
        yaml.safe_dump(client.model_dump(mode="json", exclude_none=True)), encoding="utf-8"
    )
    scenarios = tmp_path / "data" / "demo" / "scenarios"
    scenarios.mkdir(parents=True)
    shutil.copy(write_pack(tmp_path), scenarios / "pack.json")
    http = TestClient(
        create_app(
            PreviewWorkspace(config_dir=config, store_root=tmp_path / "stores"),
            JobRunner(inline=True),
            preview=True,
        )
    )
    assert "Preview mode" in http.get("/").text
    page = _review(http)
    run = records.latest_run(store)
    assert run is not None and run.model == MODEL and run.calls[0].cost == 0
    assert run.findings and all("[Sample answer]" in f.reasoning for f in run.findings)
    assert "Preview mode" in page.text and "not by an AI model" in page.text
    before = (tmp_path / "config" / "profiles" / "testwater.yaml").read_text()
    blocked = http.post("/c/testwater/profile", data={"yaml_text": "status: draft"})
    assert "read-only" in blocked.text
    assert (tmp_path / "config" / "profiles" / "testwater.yaml").read_text() == before


def test_forecast_refresh_runs_the_engine_command_as_a_job(tmp_path):
    client = make_client(tmp_path)
    seed_store(tmp_path)
    config = tmp_path / "config" / "clients"
    config.mkdir(parents=True, exist_ok=True)
    (config / "testwater.yaml").write_text(
        yaml.safe_dump(client.model_dump(mode="json", exclude_none=True)), encoding="utf-8"
    )
    calls = []

    def fake_pipeline(command, client_id, **options):
        calls.append((command, client_id, options))
        return "ok"

    workspace = Workspace(config_dir=config, store_root=tmp_path / "stores")
    http = TestClient(create_app(workspace, JobRunner(inline=True), pipeline=fake_pipeline))
    page = http.post("/c/testwater/forecast/refresh", data={"horizon": "6"})
    assert "Forecast refreshed" in page.text and calls == [
        ("forecast", "testwater", {"horizon": 6})
    ]
    page = http.post("/c/testwater/backtest")
    assert "Backtest finished" in page.text and calls[-1][0] == "backtest"
    assert (
        "Choose 1 to 24" in http.post("/c/testwater/forecast/refresh", data={"horizon": "99"}).text
    )

    # A custom workspace without an injected pipeline never runs the real engine commands.
    plain = TestClient(create_app(workspace, JobRunner(inline=True)))
    assert "preview mode" in plain.post("/c/testwater/forecast/refresh", data={"horizon": "6"}).text


def test_new_factor_can_join_the_watch_list(web, tmp_path):
    http, store, _ = web
    pack = tmp_path / "data" / "demo" / "scenarios" / "pack.json"
    document = json.loads(pack.read_text(encoding="utf-8"))
    document["drivers"]["_scan"] = [
        {
            "kind": "news",
            "summary": "open search",
            "data": {"query": "bottled water India"},
            "articles": [
                {
                    "url": "https://news.example/port",
                    "title": "Port workers begin an indefinite strike at the main harbour",
                    "source": "news.example",
                    "published": "2027-02-20T00:00:00Z",
                }
            ],
        }
    ]
    pack.write_text(json.dumps(document), encoding="utf-8")
    page = _review(http)
    assert "New factor" in page.text and "Add to watch list" in page.text
    assert "1 new factor not on the watch list" in page.text
    run = records.latest_run(store)
    new = next(f for f in run.findings if f.new_factor)

    http.cookies.set("planner", "Asha")
    done = http.post(f"/c/testwater/runs/{run.reason_id}/findings/{new.finding_id}/watch")
    assert "Added" in done.text and "watch list" in done.text
    profile = yaml.safe_load((tmp_path / "config" / "profiles" / "testwater.yaml").read_text())
    added = next(d for d in profile["drivers"] if d["id"] == "port_strike")
    assert added["template"] == "topic_news" and added["params"] == {"topic": "port strike"}
    assert added["families"] == ["jars"]
    saved = {f.finding_id: f for f in records.load_run(store, run.reason_id).findings}
    assert "accepted_by_Asha" in saved[new.finding_id].flags
    again = http.post(f"/c/testwater/runs/{run.reason_id}/findings/{new.finding_id}/watch")
    assert "already watched" in again.text
