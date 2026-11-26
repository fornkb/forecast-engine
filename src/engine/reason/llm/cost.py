"""Token estimates before a call and cost after it, from the client's configured rates."""

from __future__ import annotations

from engine.reason.config import ModelRates
from engine.reason.schemas import Usage


def estimate_tokens(text: str) -> int:
    """Rough count (about 4 characters per token) for budgets and dry runs."""
    return max(1, len(text) // 4)


def cost(usage: Usage, rates: ModelRates | None) -> float | None:
    if rates is None:
        return None
    cached = usage.cached_input_tokens if rates.cached_input_per_m is not None else 0
    fresh = max(0, usage.input_tokens - cached)
    total = fresh * rates.input_per_m + usage.output_tokens * rates.output_per_m
    if cached:
        total += cached * rates.cached_input_per_m
    return round(total / 1_000_000, 6)
