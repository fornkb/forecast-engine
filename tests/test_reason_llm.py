"""Provider adapters build the right requests and read usage; replay works offline."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from engine.reason.config import LLMConfig, ModelRates
from engine.reason.llm import get_client
from engine.reason.llm.base import LLMError, strict_schema
from engine.reason.llm.cost import cost
from engine.reason.llm.replay import ReplayClient
from engine.reason.schemas import ProfileDraftOut, ReasoningOut, Usage

ANSWER = {"summary": "s", "findings": [], "no_change": [{"driver_id": "d", "reason": "r"}]}


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


@pytest.mark.parametrize("model", [ReasoningOut, ProfileDraftOut])
def test_strict_schema_is_the_common_subset(model):
    schema = strict_schema(model)
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    for node in _walk(schema):
        assert not {"title", "default", "minimum", "maximum"} & node.keys()
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])


class Recorder:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_openai_adapter_request_and_usage():
    pytest.importorskip("openai")
    from engine.reason.llm.openai_client import OpenAIClient

    usage = NS(
        input_tokens=1000,
        output_tokens=200,
        input_tokens_details=NS(cached_tokens=600),
        output_tokens_details=NS(reasoning_tokens=50),
    )
    create = Recorder(
        NS(
            status="completed",
            output=[],
            output_text=json.dumps(ANSWER),
            usage=usage,
            incomplete_details=None,
        )
    )
    client = OpenAIClient("k", timeout=1, max_retries=0, sdk=NS(responses=NS(create=create)))
    result = client.generate(
        system="sys",
        prompt="p",
        schema=ReasoningOut,
        model="gpt-x",
        max_output_tokens=900,
        effort="low",
        name="reason",
    )
    fmt = create.kwargs["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "reason"
    assert create.kwargs["instructions"] == "sys" and create.kwargs["store"] is False
    assert create.kwargs["reasoning"] == {"effort": "low"}
    assert result.data.no_change[0].driver_id == "d"
    assert result.usage == Usage(
        input_tokens=1000, output_tokens=200, cached_input_tokens=600, reasoning_tokens=50
    )

    create.response = NS(
        status="incomplete",
        incomplete_details=NS(reason="max_output_tokens"),
        output=[],
        output_text="",
        usage=usage,
    )
    with pytest.raises(LLMError, match="incomplete"):
        client.generate(
            system="s", prompt="p", schema=ReasoningOut, model="m", max_output_tokens=10
        )
    create.response = NS(
        status="completed",
        incomplete_details=None,
        usage=usage,
        output_text="",
        output=[NS(content=[NS(type="refusal", refusal="no")])],
    )
    with pytest.raises(LLMError, match="refused"):
        client.generate(
            system="s", prompt="p", schema=ReasoningOut, model="m", max_output_tokens=10
        )


def test_anthropic_adapter_uses_output_format_and_caching():
    pytest.importorskip("anthropic")
    from engine.reason.llm.anthropic_client import AnthropicClient

    usage = NS(
        input_tokens=100,
        output_tokens=40,
        cache_read_input_tokens=900,
        cache_creation_input_tokens=0,
    )
    create = Recorder(
        NS(
            stop_reason="end_turn",
            usage=usage,
            content=[NS(type="thinking"), NS(type="text", text=json.dumps(ANSWER))],
        )
    )
    client = AnthropicClient("k", timeout=1, max_retries=0, sdk=NS(messages=NS(create=create)))
    result = client.generate(
        system="sys",
        prompt="p",
        schema=ReasoningOut,
        model="claude-x",
        max_output_tokens=900,
        effort="minimal",
    )
    assert "tool_choice" not in create.kwargs and "tools" not in create.kwargs
    assert create.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert create.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert create.kwargs["output_config"]["effort"] == "low"
    assert result.usage.input_tokens == 1000 and result.usage.cached_input_tokens == 900
    create.response = NS(stop_reason="refusal", usage=usage, content=[])
    with pytest.raises(LLMError, match="refused"):
        client.generate(
            system="s", prompt="p", schema=ReasoningOut, model="m", max_output_tokens=10
        )


def test_gemini_adapter_reads_usage_and_finish_reason():
    pytest.importorskip("google.genai")
    from engine.reason.llm.gemini_client import GeminiClient

    meta = NS(
        prompt_token_count=500,
        candidates_token_count=60,
        cached_content_token_count=100,
        thoughts_token_count=40,
    )
    call = Recorder(
        NS(
            candidates=[NS(finish_reason=NS(name="STOP"))],
            text=json.dumps(ANSWER),
            usage_metadata=meta,
        )
    )
    client = GeminiClient("k", timeout=1, max_retries=0, sdk=NS(models=NS(generate_content=call)))
    result = client.generate(
        system="sys",
        prompt="p",
        schema=ReasoningOut,
        model="gemini-x",
        max_output_tokens=900,
        effort="max",
    )
    config = call.kwargs["config"]
    assert config.response_mime_type == "application/json" and config.system_instruction == "sys"
    assert config.response_json_schema["additionalProperties"] is False
    assert result.usage.output_tokens == 100 and result.usage.reasoning_tokens == 40
    call.response = NS(
        candidates=[NS(finish_reason=NS(name="MAX_TOKENS"))], text="", usage_metadata=meta
    )
    with pytest.raises(LLMError, match="incomplete"):
        client.generate(
            system="s", prompt="p", schema=ReasoningOut, model="m", max_output_tokens=10
        )


def test_record_then_replay_offline(tmp_path):
    from engine.reason.llm.base import LLMResult

    class Live:
        provider = "openai"
        calls = 0

        def generate(self, **kwargs):
            Live.calls += 1
            return LLMResult(
                data=ReasoningOut.model_validate(ANSWER),
                text=json.dumps(ANSWER),
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="openai",
                model=kwargs["model"],
                request_hash="h",
            )

    request = dict(system="s", prompt="p", schema=ReasoningOut, model="m", max_output_tokens=10)
    ReplayClient(tmp_path, "openai", inner=Live()).generate(**request)
    replayed = ReplayClient(tmp_path, "replay").generate(**request)
    assert Live.calls == 1 and replayed.replayed and replayed.data.summary == "s"
    with pytest.raises(LLMError, match="replay_missing"):
        ReplayClient(tmp_path, "openai").generate(**{**request, "prompt": "changed"})


def test_missing_key_names_the_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(LLMError, match="MY_KEY"):
        get_client(LLMConfig(provider="openai", model="m", api_key_env="MY_KEY"), "live", tmp_path)


def test_cost_uses_cached_rate_when_given():
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000, cached_input_tokens=400_000)
    assert cost(usage, None) is None
    assert cost(usage, ModelRates(input_per_m=2, output_per_m=10)) == pytest.approx(3.0)
    assert cost(
        usage, ModelRates(input_per_m=2, output_per_m=10, cached_input_per_m=0.2)
    ) == pytest.approx(1.2 + 0.08 + 1.0)
