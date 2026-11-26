"""The optional `reasoning:` section of a client profile.

Everything a company customizes lives here or in its company profile
(config/profiles/<client_id>.yaml): which model provider and model to use, how
many calls a run may make, and which public sources to read. No code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, StringConstraints, field_validator

from engine.config.base import StrictModel

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EnvName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]

Provider = Literal["openai", "anthropic", "gemini", "replay"]
# One shared scale; each adapter maps it to what its provider accepts.
Effort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]

DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ModelRates(StrictModel):
    """Price per million tokens, used only to report cost. Unknown rates mean unknown cost."""

    input_per_m: float = Field(ge=0)
    output_per_m: float = Field(ge=0)
    cached_input_per_m: float | None = Field(default=None, ge=0)
    currency: str = "USD"


class LLMConfig(StrictModel):
    """Which model answers. `model` and `triage_model` are whatever the provider calls them."""

    provider: Provider = "openai"
    model: Text | None = None
    # A smaller, cheaper model for article triage. Defaults to `model`.
    triage_model: Text | None = None
    api_key_env: EnvName | None = None
    effort: Effort | None = None
    max_output_tokens: int = Field(default=8000, ge=256, le=64000)
    timeout_seconds: float = Field(default=180, gt=0, le=900)
    max_retries: int = Field(default=2, ge=0, le=5)
    # Price table keyed by model id, e.g. {"gpt-6.2-sol": {input_per_m: 2, output_per_m: 10}}.
    rates: dict[str, ModelRates] = Field(default_factory=dict)
    # Recorded responses for `--llm record|replay` (relative to the client profile folder).
    recordings_dir: Path | None = None

    @property
    def key_env(self) -> str | None:
        return self.api_key_env or DEFAULT_KEY_ENV.get(self.provider)


class BudgetConfig(StrictModel):
    """Hard limits that keep a monthly run cheap and predictable."""

    max_calls_per_run: int = Field(default=3, ge=0, le=10)
    # Estimated input tokens per call; bigger packets are trimmed or split.
    max_input_tokens: int = Field(default=20000, ge=2000, le=200000)
    # An AI "judgment" estimate can never move the forecast more than this (percent).
    max_judgment_pct: float = Field(default=10.0, gt=0, le=50)
    # Triage runs only when more articles than this survive the free filters.
    triage_threshold: int = Field(default=12, ge=0, le=200)
    # Articles sent to the reasoning call after triage.
    max_articles: int = Field(default=24, ge=1, le=100)


class NewsSource(StrictModel):
    provider: Literal["gdelt", "brave", "none"] = "gdelt"
    lookback_days: int = Field(default=30, ge=1, le=90)
    max_articles_per_driver: int = Field(default=8, ge=1, le=25)
    # GDELT `sourcecountry:` filter (e.g. "india") and language; Brave ignores these.
    source_country: str | None = None
    language: str | None = "english"
    api_key_env: EnvName = "BRAVE_SEARCH_API_KEY"
    timeout_seconds: float = Field(default=20, gt=0, le=60)


class WeatherSource(StrictModel):
    enabled: bool = True
    # Open-Meteo is free for non-commercial use; company deployments need the commercial key.
    access: Literal["commercial", "noncommercial"] = "noncommercial"
    api_key_env: EnvName = "OPEN_METEO_API_KEY"
    normal_years: int = Field(default=10, ge=3, le=30)
    timeout_seconds: float = Field(default=30, gt=0, le=120)


class SourcesConfig(StrictModel):
    news: NewsSource = Field(default_factory=NewsSource)
    weather: WeatherSource = Field(default_factory=WeatherSource)
    enso: bool = True
    # A JSON file of prepared signals. When set, it replaces live sources (demos and tests).
    scenario_pack: Path | None = None
    cache_hours: int = Field(default=12, ge=0, le=720)


class ReasoningConfig(StrictModel):
    enabled: bool = Field(default=False, strict=True)
    timezone: Text = "UTC"
    # Defaults to config/profiles/<client_id>.yaml.
    profile: Path | None = None
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)

    @field_validator("timezone")
    @classmethod
    def iana_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("timezone must be an IANA zone such as Asia/Kolkata") from exc
        return value
