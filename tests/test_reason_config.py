from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from engine.cli import app
from engine.config.loader import load_client
from engine.config.schema import ClientProfile
from engine.reason.config import ContextSource, ReasoningConfig
from engine.reason.erp import read_context_source, require_reasoning
from engine.reason.schemas import ContextError

PILOT = Path(__file__).parent / "fixtures" / "reasoning" / "pilot.yaml"


def test_optional_config_and_cli_group():
    client = ClientProfile(client_id="legacy", display_name="Legacy")
    assert client.reasoning is None
    with pytest.raises(ContextError, match="reasoning.enabled"):
        require_reasoning(client)
    with pytest.raises(ValidationError):
        ClientProfile(client_id="legacy", display_name="Legacy", reasoning={"enabeld": True})
    assert CliRunner().invoke(app, ["reason-context", "--help"]).exit_code == 0


def test_csv_uses_profile_directory_and_canonical_names(tmp_path):
    (tmp_path / PILOT.name).write_bytes(PILOT.read_bytes())
    (tmp_path / "context").mkdir()
    (tmp_path / "context" / "vendors.csv").write_text(
        "link_id,sku,vendor_code,start_date,released_at\nV1,I1,S1,2026-12-01,\n",
        encoding="utf-8",
    )
    profile = load_client("pilot", config_dir=tmp_path)
    source = profile.reasoning.sources[0]
    assert read_context_source(profile, source) == [
        {
            "record_id": "V1",
            "subject_id": "I1",
            "object_id": "S1",
            "effective_from": "2026-12-01",
            "known_at": "",
        }
    ]


def test_postgres_context_uses_validated_view_and_binds_client(monkeypatch):
    profile = load_client("pilot", config_dir=PILOT.parent)
    config = profile.reasoning.sources[0].model_dump()
    config.update(kind="postgres", path=None, relation="erp.v_vendors", dsn_env="TEST_REASON_DSN")
    config["fields"]["client_id"] = "client_code"
    source = ContextSource.model_validate(config)
    monkeypatch.setenv("TEST_REASON_DSN", "test-dsn")
    with pytest.raises(ContextError, match="not configured"):
        read_context_source(profile, source)
    profile = profile.model_copy(
        update={"reasoning": ReasoningConfig(enabled=True, sources=(source,))}
    )

    def fetch(query, dsn, parameters, relation):
        assert query.endswith('FROM "erp"."v_vendors" WHERE "client_code" = $1')
        assert dsn == "test-dsn"
        assert parameters == ("pilot",)
        assert relation == ("erp", "v_vendors")
        return [
            {
                "link_id": "1",
                "sku": "I",
                "vendor_code": "V",
                "start_date": "2026-01-01",
                "released_at": None,
                "client_code": "pilot",
            }
        ]

    assert read_context_source(profile, source, fetch=fetch)[0]["object_id"] == "V"
    with pytest.raises(ContextError, match="outside the configured client scope"):
        read_context_source(
            profile, source, fetch=lambda *args: [{**fetch(*args)[0], "client_code": "other"}]
        )
    with pytest.raises(ValidationError, match="relation"):
        ContextSource.model_validate({**config, "relation": "erp.v_vendors; SELECT 1"})
