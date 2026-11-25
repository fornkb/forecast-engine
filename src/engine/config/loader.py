"""Locate and load client profiles from disk."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from engine.config.schema import ClientProfile


class ConfigError(RuntimeError):
    """Raised when a client profile is missing or fails validation."""


def project_root() -> Path:
    """Repository root, derived from this file's location (src/engine/config/loader.py)."""
    return Path(__file__).resolve().parents[3]


def client_config_path(client_id: str, config_dir: Path | None = None) -> Path:
    base = config_dir or project_root() / "config" / "clients"
    return base / f"{client_id}.yaml"


def load_client(client_id: str, config_dir: Path | None = None) -> ClientProfile:
    path = client_config_path(client_id, config_dir)
    if not path.exists():
        raise ConfigError(f"No client profile at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        profile = ClientProfile(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid client profile {path}:\n{exc}") from exc

    if profile.client_id != client_id:
        raise ConfigError(
            f"client_id in {path} is {profile.client_id!r} but the file is named {client_id!r}"
        )
    profile._profile_dir = path.resolve().parent
    return profile


def list_clients(config_dir: Path | None = None) -> list[str]:
    base = config_dir or project_root() / "config" / "clients"
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))
