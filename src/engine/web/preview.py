"""Preview mode: see every page work without an API key.

`engine web --preview` (or `python web.py --preview`) runs the app on a copy of
each client's store, with a script standing in for the AI. The script reads the
same packet the model would get and writes plausible sample answers that cite
the packet's own ids, so Python's checks, bands, decisions and reports all run
for real. Every page carries a banner; nothing is written to the real stores,
profiles are read-only, and the run's model is named "preview-sample (not AI)".
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from engine.config import load_client
from engine.config.loader import project_root
from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.config import ModelRates
from engine.reason.llm.base import LLMError, LLMResult, request_hash
from engine.reason.schemas import ReasoningOut, TriageOut, Usage
from engine.store.duckdb_store import EngineStore
from engine.web.services import Workspace

MODEL = "preview-sample (not AI)"
COST_KINDS = {"commodity": "cost", "geopolitics": "supply_risk", "policy": "supply_risk"}
BLOCK = re.compile(r"^\[(\w+)\] (\w+) · families: (.*)$", re.M)
MEASURED = re.compile(r"^\s+(M\d+) Measured in this company's sales: (\w+) ran ([+-][\d.]+)%(.*)$")
MONTHS = re.compile(r"\d{4}-\d{2}")
# Sections that follow the last driver block in a packet.
SECTION_END = re.compile(r"\n\n(?:DRIVERS UNCHANGED|OTHER NEWS|SOURCES UNAVAILABLE)")


def _forecast_months(prompt: str) -> list[str]:
    line = next((x for x in prompt.splitlines() if "Forecast months:" in x), "")
    return MONTHS.findall(line.split("Forecast months:", 1)[-1])


def sample_answer(prompt: str) -> ReasoningOut:
    """Plausible answers built only from the packet, labelled as samples."""
    months = _forecast_months(prompt)
    blocks = list(BLOCK.finditer(prompt))
    findings, quiet = [], []
    for index, block in enumerate(blocks):
        driver, kind = block.group(1), block.group(2)
        families = [f.strip() for f in block.group(3).split(",") if f.strip()]
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(prompt)
        body = SECTION_END.split(prompt[block.end() : end])[0].splitlines()
        cites = [m.group(1) for line in body if (m := re.match(r"^\s+([AS]\d+) ", line))]
        added = False
        for line in body:
            measured = MEASURED.match(line)
            if not measured:
                continue
            key, family, value, rest = measured.groups()
            calendar = {m[-2:] for m in MONTHS.findall(rest)}
            target = [m for m in months if m[-2:] in calendar]
            size = float(value)
            if not target or size <= 0 or family not in families:
                continue
            findings.append(
                {
                    "driver_id": driver,
                    "family_id": family,
                    "channel": "demand",
                    "direction": "up",
                    "months": target,
                    "low_pct": round(size / 3, 1),
                    "high_pct": round(size, 1),
                    "evidence_tier": "measured",
                    "quote": "",
                    "citations": [key, *cites[:1]],
                    "already_in_forecast": False,
                    "confidence": "medium",
                    "reasoning": f"[Sample answer] The last time this driver was active, "
                    f"{family} ran {size:+.1f}% above its usual pattern in the same months. "
                    "If the signals repeat that pattern, demand could again run above a "
                    "normal season.",
                }
            )
            added = True
        if added or not cites or not families or not months:
            if not added:
                quiet.append(
                    {"driver_id": driver, "reason": "[Sample answer] No usable signal this month."}
                )
            continue
        channel = COST_KINDS.get(kind, "demand")
        moves = channel == "demand"
        findings.append(
            {
                "driver_id": driver,
                "family_id": families[0],
                "channel": channel,
                "direction": "up" if moves else "none",
                "months": months[-2:],
                "low_pct": 0,
                "high_pct": 3 if moves else 0,
                "evidence_tier": "judgment" if moves else "none",
                "quote": "",
                "citations": cites[:2],
                "already_in_forecast": False,
                "confidence": "low",
                "reasoning": "[Sample answer] The cited signals point "
                + (
                    "to a small possible lift in demand."
                    if moves
                    else "to cost or supply pressure, not a change in demand."
                ),
            }
        )
    return ReasoningOut.model_validate(
        {
            "summary": "[Sample answers written by a script in preview mode, not by an AI "
            "model.] They show how findings, checks and bands appear.",
            "findings": findings,
            "new_factors": [],
            "no_change": quiet,
        }
    )


class SampleModel:
    provider = "replay"

    def generate(self, *, system, prompt, schema, model, max_output_tokens, effort=None, name=""):
        if schema is TriageOut:
            ids = re.findall(r"^(T\d+) \|", prompt, flags=re.M)
            data = TriageOut.model_validate(
                {"keep": [{"id": i, "relevance": 2, "why": "sample"} for i in ids]}
            )
        elif schema is ReasoningOut:
            data = sample_answer(prompt)
        else:
            raise LLMError("preview_only: drafting a profile needs a real model")
        return LLMResult(
            data=data,
            text=data.model_dump_json(),
            usage=Usage(input_tokens=len(system + prompt) // 4, output_tokens=600),
            provider="replay",
            model=MODEL,
            request_hash=request_hash("preview", model, system, prompt, schema),
        )


class PreviewWorkspace(Workspace):
    """Real configs, copied stores, and the AI replaced by SampleModel."""

    def load(self, client_id: str) -> tuple[ClientProfile, EngineStore]:
        client, store = super().load(client_id)
        if client.reasoning is not None:
            llm = client.reasoning.llm.model_copy(
                update={
                    "provider": "replay",
                    "model": MODEL,
                    "triage_model": MODEL,
                    "rates": {MODEL: ModelRates(input_per_m=0, output_per_m=0)},
                }
            )
            folder = client.profile_dir
            client = client.model_copy(
                update={"reasoning": client.reasoning.model_copy(update={"llm": llm})}
            )
            client._profile_dir = folder
        return client, store


def preview_workspace(config_dir: Path | None = None) -> PreviewWorkspace:
    """Fresh copies of every reasoning client's store under data/processed/_preview."""
    source = project_root() / "data" / "processed"
    target = source / "_preview"
    shutil.rmtree(target, ignore_errors=True)
    workspace = PreviewWorkspace(config_dir=config_dir, store_root=target)
    for client_id in workspace.client_ids():
        try:
            client = load_client(client_id, config_dir)
        except Exception:  # a broken config shows its error on the page
            continue
        if client.reasoning is None or not (source / client_id).exists():
            continue
        shutil.copytree(source / client_id, target / client_id)
        store = EngineStore(client_id, root=target)
        # Start each preview with no reviews, so the first one assesses every driver.
        with records.transaction(store) as con:
            for table in ("reason_runs", "reason_bands", "reason_llm_calls", "reason_overrides"):
                con.execute(f"DELETE FROM {table} WHERE client_id = ?", [client_id])
    return workspace
