import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from engine import cli
from engine.config.loader import load_client
from engine.config.schema import ClientProfile
from engine.reason.context import get_forecast_snapshot, load_forecast_snapshot
from engine.reason.facts import get_fact_version, list_facts, sync_context
from engine.reason.questions import confirm_question, get_question_answer, list_questions
from engine.reason.schemas import ContextError
from engine.reason.store import init_reason_schema
from engine.store import EngineStore

PILOT = Path(__file__).parent / "fixtures" / "reasoning" / "pilot.yaml"
JAN = date(2027, 1, 1)
FEB = date(2027, 2, 1)
CUTOFF = date(2026, 12, 1)


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    (tmp_path / "context").mkdir()
    (tmp_path / PILOT.name).write_bytes(PILOT.read_bytes())

    def make_store(client_id):
        return EngineStore(client_id, root=tmp_path / "stores")

    monkeypatch.setattr(cli, "EngineStore", make_store)
    monkeypatch.setattr(cli, "load_client", lambda client_id: load_client(client_id, tmp_path))
    return load_client("pilot", tmp_path), make_store("pilot"), tmp_path


def forecasts(*, periods=(JAN, FEB), series="S1", baseline=0.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": [series] * len(periods),
            "period": list(periods),
            "horizon": list(range(1, len(periods) + 1)),
            "cutoff_period": [CUTOFF] * len(periods),
            "model": ["saved_model"] * len(periods),
            "forecast": [baseline] * len(periods),
            "q50": [12.0] * len(periods),
        }
    )


def seed(store: EngineStore, points: pl.DataFrame | None = None, *, horizon=2, unit=None) -> int:
    run_id = store.start_run("forecast", {"horizon": horizon})
    points = forecasts() if points is None else points
    store.write_forecasts(run_id, points)
    store.write_forecast_quantiles(run_id, points, [0.5])
    with store.connect() as con:
        con.execute(
            """INSERT INTO dim_series (client_id, series_id, customer, item_code, uom,
            source_file, ingested_at) VALUES (?, 'S1', 'C1', 'I1', ?, 'erp', current_timestamp)""",
            [store.client_id, unit],
        )
        con.execute(
            "INSERT INTO fact_demand VALUES (?, 'S1', ?, 15, 'erp', current_timestamp)",
            [store.client_id, CUTOFF],
        )
    return run_id


def invoke(command, *args, client="pilot", success=True):
    result = CliRunner().invoke(cli.app, ["reason-context", command, "--client", client, *args])
    if success:
        assert result.exit_code == 0, (result.output, result.exception)
    else:
        assert result.exit_code != 0, result.output
    return result.output


def write_vendors(root, rows, *, header="link_id,sku,vendor_code,start_date,released_at"):
    (root / "context" / "vendors.csv").write_text(header + "\n" + rows, encoding="utf-8")


def test_cli_sync_inspect_confirm_and_reopen(pilot):
    profile, store, root = pilot
    write_vendors(root, "V1,I2,Vendor2,2026-01-01,2026-01-01T00:00:00Z\n")
    run = seed(store)
    original = [
        store.load_fact(),
        store.load_dim(),
        store.load_forecasts(run),
        store.load_forecast_quantiles(run),
    ]
    assert original[-1].height == 2
    assert "new=1 reused=0 conflicted=0 invalid=0" in invoke("sync")
    assert "new=0 reused=1" in invoke("sync")
    args = ("--run-id", str(run), "--json")
    before = json.loads(invoke("inspect", *args))
    assert before["row_count"] == 2 and before["preview"][0]["baseline"] == 0
    assert before["target"]["id"] == "customer_orders"
    assert before["gap_counts"]["missing_relationship"] == 2
    assert before["preview"][0]["target_start"].startswith("2026-12-31T18:30:00")
    assert before["preview"][0]["target_end"].startswith("2027-01-31T18:30:00")
    assert json.loads(invoke("inspect", *args, "--series", "other"))["matched_row_count"] == 0
    assert json.loads(invoke("inspect", *args))["snapshot_id"] == before["snapshot_id"]

    pending = list_questions(store)
    output = invoke("questions")
    assert {question.spec.field for question in pending} == {"unit", "supplied_by"}
    assert all(question.question_id in output for question in pending)
    assert all(question.spec.affected_months == (JAN, FEB) for question in pending)
    for question in pending:
        value = "EA" if question.spec.field == "unit" else "Vendor1"
        answer_args = (
            "--question-id",
            question.question_id,
            "--value",
            value,
            "--actor",
            "planner",
        )
        assert invoke("confirm", *answer_args) == invoke("confirm", *answer_args)
        assert "already has an answer" in invoke(
            "confirm",
            "--question-id",
            question.question_id,
            "--value",
            "new",
            "--actor",
            "planner",
            success=False,
        )
    assert "Pending questions 0" in invoke("questions")
    after = json.loads(invoke("inspect", *args))
    assert after["snapshot_id"] != before["snapshot_id"]
    assert after["units"] == ["EA"]
    assert "missing_relationship" not in after["gap_counts"]
    row = after["preview"][0]
    facts = [get_fact_version(store, key) for key in row["fact_version_ids"]]
    assert {fact.relation for fact in facts} == {"unit", "supplied_by"}
    assert all(fact.actor == "planner" for fact in facts)
    assert all(fact.known_at > datetime.fromisoformat(before["captured_at"]) for fact in facts)

    reopened = EngineStore(profile.client_id, store.dir.parent)
    older = get_forecast_snapshot(reopened, before["snapshot_id"])
    newer = get_forecast_snapshot(reopened, after["snapshot_id"])
    assert older.rows[0].unit is None and older.rows[0].baseline == 0
    assert newer.rows[0].unit == "EA" and newer.rows[0].baseline == 0
    assert newer.rows[0].model == "saved_model"
    assert list(newer.iter_rows("S1")) == list(newer.rows)
    for saved, current in zip(
        original,
        [
            reopened.load_fact(),
            reopened.load_dim(),
            reopened.load_forecasts(run),
            reopened.load_forecast_quantiles(run),
        ],
        strict=True,
    ):
        assert saved.equals(current)


def test_metadata_refresh_and_two_client_stores(pilot):
    profile, store, root = pilot
    points = pl.concat([forecasts(), forecasts(series="S2", baseline=7.0)])
    run = seed(store, points, unit="EA")
    first = load_forecast_snapshot(profile, store, run)
    assert first.rows[0].unit == "EA"
    assert {gap.series_id for gap in first.gaps if gap.code == "missing_dimension"} == {"S2"}
    with store.connect() as con:
        con.execute("UPDATE dim_series SET uom = 'BOX'")
    second = load_forecast_snapshot(profile, store, run)
    assert second.snapshot_id != first.snapshot_id
    assert second.rows[0].unit == "BOX"
    assert get_forecast_snapshot(store, first.snapshot_id).rows[0].unit == "EA"

    other = ClientProfile(client_id="other", display_name="Second", reasoning={"enabled": True})
    other_store = EngineStore("other", root / "stores")
    assert seed(other_store) == run
    other_snapshot = load_forecast_snapshot(other, other_store, run)
    assert other_snapshot.rows[0].series_id == first.rows[0].series_id
    assert other_snapshot.snapshot_id != first.snapshot_id
    with pytest.raises(ContextError, match="was not found"):
        get_forecast_snapshot(other_store, first.snapshot_id)
    with pytest.raises(ContextError, match="same client"):
        load_forecast_snapshot(other, store, run)
    with pytest.raises(ContextError, match="was not found"):
        confirm_question(other, other_store, first.question_ids[0], "EA", "operator")
    with store.connect() as con:
        con.execute("UPDATE runs SET client_id = 'other' WHERE run_id = ?", [run])
    with pytest.raises(ContextError, match="was not found"):
        load_forecast_snapshot(profile, store, run)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("absent", "was not found"),
        ("backtest", "not a forecast run"),
        ("header", "no stored point rows"),
        ("missing-month", "Incomplete forecast horizons"),
        ("nan", "must be finite"),
        ("duplicate", "Duplicate point forecast"),
        ("wrong-month", "Invalid monthly target period"),
        ("cutoffs", "inconsistent history cutoffs"),
    ],
)
def test_bad_run_creates_no_reasoning_work(pilot, kind, message):
    _, store, _ = pilot
    if kind == "absent":
        run = 999
    elif kind in {"backtest", "header"}:
        run = store.start_run("backtest" if kind == "backtest" else "forecast")
    else:
        points = {
            "missing-month": forecasts().head(1),
            "nan": forecasts(baseline=float("nan")),
            "duplicate": pl.concat([forecasts(), forecasts()]),
            "wrong-month": forecasts(periods=(date(2027, 1, 2), FEB)),
            "cutoffs": forecasts().with_columns(pl.Series("cutoff_period", [CUTOFF, JAN])),
        }[kind]
        run = seed(store, points)
    assert message in invoke("inspect", "--run-id", str(run), success=False)
    assert not store.table_exists("reason_forecast_snapshots")


def test_weekly_profile_and_forecast_only_profile(pilot):
    _, store, root = pilot
    run = seed(store)
    config = yaml.safe_load(PILOT.read_text(encoding="utf-8"))
    config["calendar"] = {"granularity": "weekly"}
    (root / PILOT.name).write_text(yaml.safe_dump(config), encoding="utf-8")
    assert "requires a monthly" in invoke("inspect", "--run-id", str(run), success=False)
    assert not store.table_exists("reason_forecast_snapshots")
    config.pop("reasoning")
    (root / PILOT.name).write_text(yaml.safe_dump(config), encoding="utf-8")
    assert "reasoning.enabled" in invoke("questions", success=False)


def test_revision_history_idempotence_withdrawal_and_rollback(pilot):
    _, store, root = pilot
    config = yaml.safe_load(PILOT.read_text(encoding="utf-8"))
    config["reasoning"]["sources"][0]["fields"].update(
        version="revision", status="status", effective_to="end"
    )
    (root / PILOT.name).write_text(yaml.safe_dump(config), encoding="utf-8")
    profile = load_client("pilot", root)
    header = "link_id,sku,vendor_code,start_date,released_at,revision,status,end"
    first = "V1,I1,Vendor1,2026-12-01,2026-08-01T05:30:00+05:30,1,active,2027-02-01\n"
    second = "V1,I1,Vendor2,2026-12-01,2026-09-01T00:00:00Z,2,active,2027-02-01\n"
    write_vendors(root, first + first + second, header=header)
    init_reason_schema(store)
    init_reason_schema(store)
    result = sync_context(profile, store)
    assert (result.new, result.reused) == (2, 1)
    jan_start = datetime(2026, 12, 31, 18, 30, tzinfo=UTC)
    feb_start = datetime(2027, 1, 31, 18, 30, tzinfo=UTC)

    def query(month, start=jan_start, end=feb_start):
        return list_facts(
            store, known_at=datetime(2026, month, 15, tzinfo=UTC), valid_from=start, valid_to=end
        )

    assert not query(7)
    (before,) = query(8)
    (after,) = query(9)
    assert before.object.id == "Vendor1" and after.object.id == "Vendor2"
    assert before.known_at == datetime(2026, 8, 1, tzinfo=UTC)
    assert before.fact_version_id != after.fact_version_id
    assert not query(9, feb_start, datetime(2027, 3, 1, tzinfo=UTC))
    write_vendors(root, "", header=header)
    assert sync_context(profile, store).new == 0 and query(9) == (after,)

    # The first row is well-formed. The second conflicts with a stored immutable
    # version during persistence, which must roll back the first insert too.
    extra = "V2,I2,Vendor3,2026-12-01,2026-09-01T00:00:00Z,1,active,\n"
    write_vendors(root, extra + second.replace("Vendor2", "changed"), header=header)
    error = invoke("sync", success=False)
    assert "product_vendors" in error and "invalid=1" in error
    assert query(9) == (after,)
    write_vendors(root, extra + extra.replace("2026-12-01", "invalid-date"), header=header)
    assert "invalid=1" in invoke("sync", success=False)
    assert query(9) == (after,)

    withdrawn = "V1,I1,,2026-12-01,2026-10-01T00:00:00Z,3,withdrawn,\n"
    write_vendors(root, withdrawn, header=header)
    assert sync_context(profile, store).new == 1
    assert not query(10)
    assert query(9) == (after,)
    assert get_fact_version(store, before.fact_version_id) == before


def test_conflicting_single_relationship_retains_both_observations(pilot):
    _, store, root = pilot
    config = yaml.safe_load(PILOT.read_text(encoding="utf-8"))
    source = config["reasoning"]["sources"][0]
    source["cardinality"] = "one"
    (root / PILOT.name).write_text(yaml.safe_dump(config), encoding="utf-8")
    profile = load_client("pilot", root)
    write_vendors(root, "V1,I1,A,2026-01-01,\nV2,I1,B,2026-01-01,\n")
    assert sync_context(profile, store).conflicted == 2
    run = seed(store, unit="EA")
    old = load_forecast_snapshot(profile, store, run)
    (question,) = list_questions(store)
    assert question.spec.purpose == "fact_conflict"
    assert question.spec.affected_months == (JAN, FEB)
    assert len(question.spec.candidate_fact_ids) == 2
    assert "conflicting_fact" in {gap.code for gap in old.gaps}
    answer = confirm_question(profile, store, question.question_id, "A", "operator")
    fresh = load_forecast_snapshot(profile, store, run)
    assert "conflicting_fact" not in {gap.code for gap in fresh.gaps}
    assert {
        fact.object.id
        for fact in list_facts(
            store,
            known_at=fresh.captured_at,
            valid_from=fresh.rows[0].target_start,
            valid_to=fresh.rows[0].target_end,
        )
    } == {"A", "B"}
    assert set(fresh.rows[0].fact_version_ids) == {
        *old.rows[0].fact_version_ids,
        answer.fact_version_id,
    }
    assert fresh.rows[0].superseded_fact_version_ids == old.rows[0].fact_version_ids
    assert all(
        get_fact_version(store, fact_id).known_at <= old.captured_at
        for fact_id in old.rows[0].fact_version_ids
    )
    corrected = confirm_question(
        profile, store, question.question_id, "B", "operator", supersedes_answer_id=answer.answer_id
    )
    assert corrected.revision == 2 and corrected.supersedes_answer_id == answer.answer_id
    assert corrected.fact_version_id != answer.fact_version_id
    assert get_fact_version(store, answer.fact_version_id).object.id == "A"
    assert get_question_answer(store, answer.answer_id) == answer

    # A replacement ERP revision falls outside the evidence used for the prior resolution.
    write_vendors(root, "V1,I1,C,2026-01-01,\nV2,I1,B,2026-01-01,\n")
    assert sync_context(profile, store).conflicted == 3
    reopened = load_forecast_snapshot(profile, store, run)
    (pending,) = list_questions(store)
    assert pending.question_id == question.question_id
    assert pending.answer == corrected
    assert "conflicting_fact" in {gap.code for gap in reopened.gaps}


def test_unknown_availability_uses_capture_time_and_quantiles_are_optional(pilot):
    profile, store, root = pilot
    write_vendors(root, "V1,I1,A,2026-01-01,\n")
    sync_context(profile, store)
    run = seed(store, unit="EA")
    with store.connect() as con:
        con.execute("DELETE FROM forecast_quantiles WHERE run_id = ?", [run])
    snapshot = load_forecast_snapshot(profile, store, run)
    row = snapshot.rows[0]
    (fact_id,) = row.fact_version_ids
    fact = get_fact_version(store, fact_id)
    assert fact.known_at == fact.captured_at <= snapshot.captured_at
    assert not list_facts(
        store,
        known_at=fact.known_at - timedelta(microseconds=1),
        valid_from=row.target_start,
        valid_to=row.target_end,
    )
    assert store.load_forecast_quantiles(run).is_empty()


def test_unknown_target_can_be_confirmed_as_typed_configuration(pilot):
    _, store, _ = pilot
    profile = ClientProfile(client_id="pilot", display_name="Pilot", reasoning={"enabled": True})
    run = seed(store, unit="EA")
    old = load_forecast_snapshot(profile, store, run)
    question = next(
        question for question in list_questions(store) if question.spec.field == "target"
    )
    with pytest.raises(ContextError, match="JSON object"):
        confirm_question(profile, store, question.question_id, "orders", "operator")
    confirm_question(
        profile,
        store,
        question.question_id,
        json.dumps(
            {
                "id": "orders",
                "definition": "Customer orders",
                "outcome_source": "orders",
                "unit_field": "uom",
            }
        ),
        "operator",
    )
    current = load_forecast_snapshot(profile, store, run)
    assert current.target.id == "orders" and current.rows[0].unit == "EA"
    assert old.target is None and old.rows[0].unit is None
