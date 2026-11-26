"""OpenAI adapter: Responses API with strict JSON-schema output (official `openai` SDK)."""

from __future__ import annotations

from engine.reason.llm.base import LLMError, LLMResult, T, parse_json, request_hash, strict_schema
from engine.reason.schemas import Usage


class OpenAIClient:
    provider = "openai"

    def __init__(self, key: str, *, timeout: float, max_retries: int, sdk=None) -> None:
        if sdk is None:
            try:
                import openai
            except ImportError:
                raise LLMError(
                    'The OpenAI SDK is not installed. Run: pip install -e ".[openai]"'
                ) from None
            sdk = openai.OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)
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
        import openai

        kwargs = dict(
            model=model,
            instructions=system,
            input=prompt,
            max_output_tokens=max_output_tokens,
            store=False,
            # Same key for the same stage helps OpenAI route repeat prefixes to its prompt cache.
            prompt_cache_key=f"engine-reason-{name}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": strict_schema(schema),
                    "strict": True,
                }
            },
        )
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        try:
            response = self.sdk.responses.create(**kwargs)
        except openai.APIStatusError as exc:
            raise LLMError(
                f"openai_http_{exc.status_code}: {str(exc)[:200]}",
                retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500,
            ) from None
        except (openai.APIConnectionError, openai.APITimeoutError):
            raise LLMError("openai_network_error", retryable=True) from None
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            raise LLMError(f"incomplete_output:{reason} (raise reasoning.llm.max_output_tokens)")
        for item in response.output or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise LLMError("model_refused")
        text = response.output_text or ""
        usage = response.usage
        tokens = Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(
                getattr(usage, "input_tokens_details", None), "cached_tokens", 0
            )
            or 0,
            reasoning_tokens=getattr(
                getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0
            )
            or 0,
        )
        return LLMResult(
            data=parse_json(text, schema),
            text=text,
            usage=tokens,
            provider=self.provider,
            model=model,
            request_hash=request_hash(self.provider, model, system, prompt, schema),
        )
