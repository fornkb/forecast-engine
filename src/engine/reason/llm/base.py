"""One small interface every model provider implements.

The reasoning code only ever calls `generate(...)` and gets back a validated
Pydantic object plus token usage. Provider details (SDK, parameter names,
structured-output format, caching) stay inside each adapter.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from engine.reason.schemas import ReasoningError, Usage, digest

T = TypeVar("T", bound=BaseModel)

# Keywords some providers reject in strict mode. Values are checked in Python instead.
_DROP = {
    "title",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
    "description",
}


class LLMError(ReasoningError):
    def __init__(self, reason: str, *, retryable: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


@dataclass(frozen=True)
class LLMResult:
    data: BaseModel
    text: str
    usage: Usage
    provider: str
    model: str
    request_hash: str
    replayed: bool = False


class LLMClient(Protocol):
    provider: str

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
    ) -> LLMResult: ...


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema with $refs inlined, every property required and no extra keys.

    This is the common subset of OpenAI strict mode, Anthropic structured
    outputs and Gemini `response_json_schema`.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
        out = {key: walk(value) for key, value in node.items() if key not in _DROP}
        if out.get("type") == "object" or "properties" in out:
            out["type"] = "object"
            out.setdefault("properties", {})
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    return walk(raw)


def request_hash(provider: str, model: str, system: str, prompt: str, schema: type) -> str:
    return digest([provider, model, system, prompt, schema.__name__, strict_schema(schema)])


def parse_json(text: str, schema: type[T]) -> T:
    try:
        return schema.model_validate(json.loads(text))
    except (ValueError, ValidationError) as exc:
        raise LLMError(f"invalid_structured_output: {str(exc)[:200]}") from None


def api_key(env_name: str | None) -> str:
    value = os.environ.get(env_name or "")
    if not value:
        raise LLMError(
            f"missing_api_key: set the {env_name} environment variable "
            "(or change reasoning.llm.api_key_env)"
        )
    return value
