"""Anthropic adapter: Messages API with `output_config.format` JSON schema (official SDK).

No forced tool choice (current Claude models reject it). The system prompt is
marked for prompt caching so repeat calls with the same instructions are cheaper.
"""

from __future__ import annotations

from engine.reason.llm.base import LLMError, LLMResult, T, parse_json, request_hash, strict_schema
from engine.reason.schemas import Usage

# Claude's effort scale has no "minimal".
_EFFORT = {"minimal": "low"}


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, key: str, *, timeout: float, max_retries: int, sdk=None) -> None:
        if sdk is None:
            try:
                import anthropic
            except ImportError:
                raise LLMError(
                    'The Anthropic SDK is not installed. Run: pip install -e ".[anthropic]"'
                ) from None
            sdk = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=max_retries)
        self.sdk = sdk

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
        import anthropic

        output_config: dict = {"format": {"type": "json_schema", "schema": strict_schema(schema)}}
        if effort:
            output_config["effort"] = _EFFORT.get(effort, effort)
        try:
            response = self.sdk.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
            )
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"anthropic_http_{exc.status_code}: {str(exc)[:200]}",
                retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500,
            ) from None
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            raise LLMError("anthropic_network_error", retryable=True) from None
        if response.stop_reason == "refusal":
            raise LLMError("model_refused")
        if response.stop_reason == "max_tokens":
            raise LLMError("incomplete_output:max_tokens (raise reasoning.llm.max_output_tokens)")
        text = next((block.text for block in response.content if block.type == "text"), "")
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        tokens = Usage(
            # Anthropic reports uncached input separately; the ledger keeps total input.
            input_tokens=(usage.input_tokens or 0) + cache_read + cache_write,
            output_tokens=usage.output_tokens or 0,
            cached_input_tokens=cache_read,
        )
        return LLMResult(
            data=parse_json(text, schema),
            text=text,
            usage=tokens,
            provider=self.provider,
            model=model,
            request_hash=request_hash(self.provider, model, system, prompt, schema),
        )
