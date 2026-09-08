from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from engine.config.loader import ConfigError, load_client
from engine.config.schema import ClientProfile


def test_defaults_are_coherent():
    profile = ClientProfile(client_id="acme", display_name="Acme")
    assert profile.calendar.seasonal_period == 12
    assert profile.models.baseline in profile.models.enabled


def test_unknown_key_is_rejected():
    """Typos must fail loudly rather than silently falling back to a default."""
    with pytest.raises(ValidationError):
        ClientProfile(client_id="acme", display_name="Acme", seasonal_periods=12)


def test_baseline_must_be_enabled():
    with pytest.raises(ValidationError):
        ClientProfile(
            client_id="acme",
            display_name="Acme",
            models={"enabled": ["naive"], "baseline": "seasonal_naive"},
        )


def test_client_id_must_match_filename(tmp_path):
    (tmp_path / "alpha.yaml").write_text(
        yaml.safe_dump({"client_id": "beta", "display_name": "Beta"}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="but the file is named"):
        load_client("alpha", config_dir=tmp_path)


def test_missing_profile_reports_path(tmp_path):
    with pytest.raises(ConfigError, match="No client profile"):
        load_client("ghost", config_dir=tmp_path)


def test_shipped_profile_loads():
    profile = load_client("teknic_mumbai")
    assert profile.calendar.granularity == "monthly"
    assert "lightgbm_global" in profile.models.enabled
