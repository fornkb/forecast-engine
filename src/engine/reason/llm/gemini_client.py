"""Gemini adapter: `generate_content` with `response_json_schema` (official google-genai SDK)."""

from __future__ import annotations

from engine.reason.llm.base import LLMError, LLMResult, T, parse_json, request_hash, strict_schema
from engine.reason.schemas import Usage

_LEVEL = {
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "xhigh": "XHIGH",
    "max": "XHIGH",
}
_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE"}


class GeminiClient:
    provider = "gemini"

    def __init__(self, key: str, *, timeout: float, max_retries: int, sdk=None) -> None:
        if sdk is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                raise LLMError(
                    'The Gemini SDK is not installed. Run: pip install -e ".[gemini]"'
                ) from None
            del max_retries  # the SDK applies its own retry policy
            sdk = genai.Client(
                api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
            )
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
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=strict_schema(schema),
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=_LEVEL[effort]) if effort else None,
        )
        try:
            response = self.sdk.models.generate_content(model=model, contents=prompt, config=config)
        except errors.APIError as exc:
            code = getattr(exc, "code", 0) or 0
            raise LLMError(
                f"gemini_http_{code}: {str(exc)[:200]}", retryable=code in (408, 429) or code >= 500
            ) from None
        candidates = response.candidates or []
        if not candidates:
            raise LLMError("model_refused: prompt blocked")
        reason = getattr(candidates[0].finish_reason, "name", str(candidates[0].finish_reason))
        if reason == "MAX_TOKENS":
            raise LLMError("incomplete_output:max_tokens (raise reasoning.llm.max_output_tokens)")
        if reason in _BLOCKED:
            raise LLMError(f"model_refused:{reason}")
        text = response.text or ""
        meta = response.usage_metadata
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        tokens = Usage(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            # Thinking tokens are billed as output.
            output_tokens=(getattr(meta, "candidates_token_count", 0) or 0) + thoughts,
            cached_input_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
            reasoning_tokens=thoughts,
        )
        return LLMResult(
            data=parse_json(text, schema),
            text=text,
            usage=tokens,
            provider=self.provider,
            model=model,
            request_hash=request_hash(self.provider, model, system, prompt, schema),
        )
