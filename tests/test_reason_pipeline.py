"""End-to-end reasoning runs on a tiny client, with a fake model and a scenario pack."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from reason_kit import FakeModel, make_client, seed_store, write_pack

from engine.reason import store as records
from engine.reason.report import render_html, render_markdown
from engine.reason.run import Options, apply_override, run_review
from engine.reason.schemas import Override, ReasoningError


def _run(client, store, pack, model=None, **kwargs):
    return run_review(client, store, None, Options(scenario=pack, client=model, **kwargs))


def test_one_call_checks_and_bands(tmp_path):
    client, (store, run_id) = make_client(tmp_path), seed_store(tmp_path)
    before = store.load_forecasts(run_id)
    model = FakeModel()
    run = _run(client, store, write_pack(tmp_path), model)

    assert [name for name, _ in model.prompts] == ["reason"]  # few articles: no triage call
    assert run.status == "complete" and len(run.calls) == 1 and run.calls[0].cost is not None
    by_key = {(f.driver_id, f.family_id): f for f in run.findings}
    nino = by_key["el_nino", "bottles"]
    assert nino.tier == "measured" and "unknown_citation_removed" in nino.flags
    assert [m.isoformat() for m in nino.months] == ["2027-04-01"]  # 2030 month dropped
    jars = by_key["el_nino", "jars"]
    assert jars.tier == "judgment" and "quote_not_found_downgraded" in jars.flags
    assert jars.high_pct == 10  # judgment cap
    resin = by_key["resin", "bottles"]
    assert resin.channel == "cost" and resin.low_pct == resin.high_pct == 0
    assert [q.driver_id for q in run.quiet] == ["rules"]

    bands = {(b.series_id, b.period.month): b for b in run.bands}
    assert bands["B1", 4].low == pytest.approx(103) and bands["B1", 4].high == pytest.approx(109)
    assert bands["B1", 3].low == bands["B1", 3].high == 100  # cost finding moves nothing
    assert bands["J1", 4].high == pytest.approx(220) and bands["J1", 4].low == 200
    assert store.load_forecasts(run_id).equals(before)  # stored forecast untouched

    page = render_html(run)
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert "Bottles" in page and "What-if band" in page and "Scenario data" in page
    assert "## Bottles" in render_markdown(run)


def test_nothing_new_costs_nothing_and_same_inputs_return_saved_run(tmp_path):
    client, (store, _) = make_client(tmp_path), seed_store(tmp_path)
    pack = write_pack(tmp_path)
    first = _run(client, store, pack, FakeModel())
    second = _run(client, store, pack, model=None)  # no model available, none needed
    assert second.calls == [] and {d.status for d in second.drivers} == {"unchanged"}
    carried = {(f.driver_id, f.family_id): f for f in second.findings}
    assert carried["el_nino", "bottles"].carried_from == first.reason_id
    assert {b.high for b in second.bands} == {b.high for b in first.bands}
    third = _run(client, store, pack, model=None)
    assert third.reason_id == second.reason_id and third.calls == []


def test_new_article_reassesses_only_its_driver(tmp_path):
    client, (store, _) = make_client(tmp_path), seed_store(tmp_path)
    _run(client, store, write_pack(tmp_path), FakeModel())
    extra = [
        {
            "url": "https://news.example/resin2",
            "title": "Resin supply tightens further now",
            "source": "news.example",
            "published": "2027-02-25T00:00:00Z",
            "snippet": "",
        }
    ]
    model = FakeModel()
    run = _run(client, store, write_pack(tmp_path, extra_resin=extra), model)
    assert {d.driver_id: d.status for d in run.drivers} == {
        "el_nino": "unchanged",
        "resin": "changed",
        "rules": "unchanged",
    }
    ((_, prompt),) = model.prompts
    assert "[resin]" in prompt and "[el_nino]" not in prompt
    assert "Resin supply tightens" in prompt and "PET resin prices rise" not in prompt
    assert any(f.driver_id == "el_nino" and f.carried_from for f in run.findings)


def test_triage_only_when_many_articles(tmp_path):
    client = make_client(tmp_path, triage_threshold=3, max_articles=4)
    store, _ = seed_store(tmp_path)
    model = FakeModel()
    run = _run(client, store, write_pack(tmp_path, many=6), model)
    assert [name for name, _ in model.prompts] == ["triage", "reason"]
    assert [c.model for c in run.calls] == ["small", "big"]
    reason_prompt = model.prompts[1][1]
    assert len([line for line in reason_prompt.splitlines() if line.strip().startswith("A")]) <= 4


def test_zero_call_budget_and_dry_run_spend_nothing(tmp_path):
    client, (store, _) = make_client(tmp_path, max_calls_per_run=0), seed_store(tmp_path)
    run = _run(client, store, write_pack(tmp_path), model=None)
    assert run.calls == [] and run.status == "partial"
    assert {item.split(":")[0] for item in run.not_assessed} == {"el_nino", "resin", "rules"}

    client2 = make_client(tmp_path / "b")
    store2, _ = seed_store(tmp_path / "b")
    dry = _run(client2, store2, write_pack(tmp_path / "b"), model=None, dry_run=True)
    assert dry.status == "dry_run" and dry.calls == [] and dry.estimate["calls"] == 1
    assert dry.estimate["input_tokens"] > 0 and records.latest_run(store2) is None


def test_override_recomputes_bands_without_a_call(tmp_path):
    client, (store, _) = make_client(tmp_path), seed_store(tmp_path)
    run = _run(client, store, write_pack(tmp_path), FakeModel())
    target = next(f for f in run.findings if f.driver_id == "el_nino" and f.family_id == "bottles")
    now = datetime.now(UTC)
    edited = apply_override(
        client,
        store,
        Override(
            reason_id=run.reason_id,
            finding_id=target.finding_id,
            action="edit",
            low_pct=1,
            high_pct=2,
            actor="planner",
            at=now,
        ),
    )
    band = next(b for b in edited.bands if b.series_id == "B1" and b.period.month == 4)
    assert (band.low, band.high) == (pytest.approx(101), pytest.approx(102))
    dismissed = apply_override(
        client,
        store,
        Override(
            reason_id=run.reason_id,
            finding_id=target.finding_id,
            action="dismiss",
            actor="planner",
            at=now,
        ),
    )
    band = next(b for b in dismissed.bands if b.series_id == "B1" and b.period.month == 4)
    assert band.low == band.high == 100
    assert len(records.call_ledger(store)) == 1
    with pytest.raises(ReasoningError):
        apply_override(
            client,
            store,
            Override(reason_id=run.reason_id, finding_id="F99", action="accept", actor="p", at=now),
        )


def test_weekly_forecasts_are_rejected(tmp_path):
    client = make_client(tmp_path)
    weekly = client.model_copy(
        update={"calendar": client.calendar.model_copy(update={"granularity": "weekly"})}
    )
    weekly._profile_dir = client.profile_dir
    store, _ = seed_store(tmp_path)
    with pytest.raises(ReasoningError, match="monthly"):
        _run(weekly, store, write_pack(tmp_path), FakeModel())


def test_reasoning_must_be_enabled(tmp_path):
    client = make_client(tmp_path)
    off = client.model_copy(
        update={"reasoning": client.reasoning.model_copy(update={"enabled": False})}
    )
    store, _ = seed_store(tmp_path)
    with pytest.raises(ReasoningError, match="not enabled"):
        _run(off, store, write_pack(tmp_path), FakeModel())
