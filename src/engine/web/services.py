"""What the pages show: client status, scenario files, profile checks. No HTTP here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from pydantic import ValidationError

from engine.config import ConfigError, list_clients, load_client
from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.connectors.scenario import ScenarioPack
from engine.reason.profile import (
    _config_root,
    load_catalog,
    load_profile,
    profile_path,
    resolve_all,
)
from engine.reason.run import recordings_path
from engine.reason.schemas import CompanyProfile, ReasoningError
from engine.store.duckdb_store import EngineStore
from engine.web import insights

SCENARIO_DIRS = ("data/demo/scenarios", "data/scenarios")


class NotFound(LookupError):
    pass


@dataclass
class Workspace:
    """Where client configs and stores live. Defaults match the CLI."""

    config_dir: Path | None = None  # config/clients
    store_root: Path | None = None  # data/processed

    def client_ids(self) -> list[str]:
        return list_clients(self.config_dir)

    def load(self, client_id: str) -> tuple[ClientProfile, EngineStore]:
        # Only ids that exist as files are accepted, so a URL can never name a path.
        if client_id not in self.client_ids():
            raise NotFound(f"Unknown client {client_id!r}")
        client = load_client(client_id, self.config_dir)
        return client, EngineStore(client.client_id, root=self.store_root)


@dataclass
class ForecastInfo:
    run_id: int
    created_at: object = None
    months: list[date] = field(default_factory=list)
    series: int = 0


@dataclass
class ClientStatus:
    client_id: str
    name: str
    granularity: str = ""
    enabled: bool = False
    provider: str = ""
    model: str = ""
    triage_model: str = ""
    effort: str = ""
    key_env: str = ""
    key_set: bool = False
    profile_status: str = ""  # draft | confirmed | missing | invalid
    profile_error: str = ""
    families: int = 0
    drivers: int = 0
    driver_problems: list[str] = field(default_factory=list)
    forecast: ForecastInfo | None = None
    last_review: dict | None = None
    reviews: int = 0
    recordings: int = 0
    error: str = ""

    @property
    def ready(self) -> bool:
        return (
            self.enabled
            and self.forecast is not None
            and self.profile_status in ("draft", "confirmed")
            and self.granularity == "monthly"
        )


def forecast_info(store: EngineStore) -> ForecastInfo | None:
    if not store.path.exists():
        return None
    run_id = store.latest_run("forecast")
    if run_id is None:
        return None
    info = store.run_info(run_id) or {}
    with store.connect() as con:
        months, series = con.execute(
            "SELECT list(DISTINCT period ORDER BY period), count(DISTINCT series_id)"
            " FROM forecasts WHERE run_id = ?",
            [run_id],
        ).fetchone()
    return ForecastInfo(run_id, info.get("created_at"), list(months or []), int(series or 0))


def client_status(workspace: Workspace, client_id: str, detail: bool = True) -> ClientStatus:
    try:
        client, store = workspace.load(client_id)
    except ConfigError as exc:
        return ClientStatus(client_id, client_id, error=str(exc).splitlines()[0])
    status = ClientStatus(
        client_id=client.client_id,
        name=client.display_name or client.client_id,
        granularity=client.calendar.granularity,
    )
    config = client.reasoning
    if config is not None:
        llm = config.llm
        status.enabled = config.enabled
        status.provider, status.model = llm.provider, llm.model or ""
        status.triage_model, status.effort = llm.triage_model or "", llm.effort or ""
        status.key_env = llm.key_env if llm.provider != "replay" else ""
        status.key_set = bool(status.key_env and os.environ.get(status.key_env))
        try:
            status.recordings = len(list(recordings_path(client).glob("*.json")))
        except (OSError, ReasoningError):
            status.recordings = 0
        if profile_path(client).exists():
            try:
                profile = load_profile(client)
                status.profile_status = profile.status
                status.families, status.drivers = len(profile.families), len(profile.drivers)
                if detail:
                    status.driver_problems = [
                        f"{d.id}: {d.error}"
                        for d in resolve_all(profile, load_catalog(client))
                        if d.error
                    ]
            except ReasoningError as exc:
                status.profile_status, status.profile_error = "invalid", str(exc)
        else:
            status.profile_status = "missing"
    try:
        status.forecast = forecast_info(store)
        runs = records.list_runs(store)
    except Exception as exc:  # a locked or half-built store should not break the page
        status.error = f"Could not read the store: {exc}"
        runs = []
    status.reviews = len(runs)
    if runs:
        reason_id, forecast_run, state, created = runs[0]
        status.last_review = {
            "reason_id": reason_id,
            "forecast_run_id": forecast_run,
            "status": state,
            "created_at": created,
        }
    return status


@dataclass
class Scenario:
    value: str  # path relative to the project, as the form sends it
    name: str
    note: str
    as_of: date | None


def scenarios(client: ClientProfile) -> list[Scenario]:
    """Scenario packs that mention at least one of this client's drivers."""
    root = _config_root(client).parent
    try:
        drivers = {driver.id for driver in load_profile(client).drivers}
    except ReasoningError:
        drivers = set()
    found = []
    for folder in SCENARIO_DIRS:
        for path in sorted((root / folder).glob("*.json")):
            try:
                pack = ScenarioPack(path)
            except (ReasoningError, ValueError, KeyError):
                continue
            if drivers and not drivers & set(pack._drivers):
                continue
            found.append(
                Scenario(path.relative_to(root).as_posix(), pack.name, pack.note, pack.as_of)
            )
    return found


def scenario_path(client: ClientProfile, value: str) -> Path | None:
    """Accept only a scenario that `scenarios()` offered."""
    if not value:
        return None
    for item in scenarios(client):
        if item.value == value:
            return Path(item.value)
    raise ReasoningError(f"Unknown scenario {value!r}")


def profile_text(client: ClientProfile) -> str:
    path = profile_path(client)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_profile(text: str) -> CompanyProfile:
    try:
        return CompanyProfile.model_validate(yaml.safe_load(text) or {})
    except yaml.YAMLError as exc:
        raise ReasoningError(f"Not valid YAML: {exc}") from exc
    except ValidationError as exc:
        lines = [
            f"{'.'.join(str(p) for p in err['loc']) or 'profile'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ReasoningError("The profile does not validate:\n" + "\n".join(lines)) from exc


# ------------------------------------------------------------------ manager view


@dataclass
class ClientView:
    """Everything the portfolio and summary pages show for one client."""

    status: ClientStatus
    summary: str = ""
    profile: CompanyProfile | None = None
    forecast: insights.ForecastOverview | None = None
    accuracy: insights.Accuracy | None = None
    digest: insights.ReviewDigest | None = None
    spend: tuple[float, int] = (0.0, 0)
    attention: list[insights.Attention] = field(default_factory=list)

    @property
    def level(self) -> str:
        """Worst attention level: bad, warn, info, ok, or off."""
        if not self.status.enabled and not self.status.error:
            return "off"
        return self.attention[0].level if self.attention else "ok"

    @property
    def reviewed_current(self) -> bool:
        return bool(
            self.digest
            and self.forecast
            and self.digest.run.forecast_run_id == self.forecast.run_id
        )


def client_view(workspace: Workspace, client_id: str) -> ClientView:
    status = client_status(workspace, client_id)
    view = ClientView(status=status)
    if status.error and not status.name:
        return view
    try:
        client, store = workspace.load(client_id)
    except ConfigError:
        return view
    if status.profile_status in ("draft", "confirmed"):
        try:
            view.profile = load_profile(client)
            view.summary = view.profile.company.summary
        except ReasoningError:
            view.profile = None
    try:
        view.forecast = insights.forecast_overview(store, view.profile)
        view.accuracy = insights.accuracy(store)
        if status.enabled:
            latest = records.latest_run(store)
            view.digest = insights.review_digest(latest) if latest else None
            view.spend = insights.month_spend(store)
    except Exception as exc:  # a locked or half-built store should not break the portfolio
        status.error = status.error or f"Could not read the store: {exc}"
    view.attention = insights.attention(status, view.forecast, view.digest)
    return view
