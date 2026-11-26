"""Record live answers once, replay them later at zero cost.

`record` wraps a live client and saves every answer as JSON keyed by a hash of
the exact request. `replay` serves those files and never touches the network,
so a demo or a test reproduces the same report offline. A changed prompt,
packet or model gets a different hash and needs a fresh recording.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.reason.llm.base import LLMClient, LLMError, LLMResult, T, parse_json, request_hash
from engine.reason.schemas import Usage


class ReplayClient:
    def __init__(self, directory: Path, provider: str, *, inner: LLMClient | None = None) -> None:
        self.directory = directory
        self.provider = provider  # the provider the recordings came from
        self.inner = inner  # set only in record mode

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        model: str,
        max_output_tokens: int,
        effort: str | None = None,
        name: str = "result",
    ) -> LLMResult:
        # Provider-independent key, so `provider: replay` finds recordings made with any provider.
        key = request_hash("recording", model, system, prompt, schema)
        path = self._path(key)
        if self.inner is None:
            if not path.exists():
                raise LLMError(
                    f"replay_missing: no recording {path.name} in {self.directory}. "
                    "Run once with --llm record (needs an API key)."
                )
            saved = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(
                data=parse_json(saved["text"], schema),
                text=saved["text"],
                usage=Usage.model_validate(saved["usage"]),
                provider=self.provider,
                model=model,
                request_hash=key,
                replayed=True,
            )
        result = self.inner.generate(
            system=system,
            prompt=prompt,
            schema=schema,
            model=model,
            max_output_tokens=max_output_tokens,
            effort=effort,
            name=name,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stage": name,
                    "model": model,
                    "text": result.text,
                    "usage": result.usage.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
