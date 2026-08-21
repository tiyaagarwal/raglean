"""Model catalog: tier + published $/1M-token pricing.

These figures are illustrative, sourced from public provider pricing pages, and will
drift as providers change prices — refresh before quoting numbers publicly or citing
them in a report. They exist so the benchmark/eval harness can compute *directionally
correct, reproducible* cost numbers without needing live billing data.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: str  # "cheap" | "mid" | "premium"
    input_price_per_1m: float  # USD per 1,000,000 input tokens
    output_price_per_1m: float  # USD per 1,000,000 output tokens
    simulated_latency_ms: tuple[float, float]  # (min, max) for MockProvider sampling


MODEL_CATALOG: dict[str, ModelSpec] = {
    "llama-3.1-8b-instant": ModelSpec(
        name="llama-3.1-8b-instant",
        tier="cheap",
        input_price_per_1m=0.05,
        output_price_per_1m=0.08,
        simulated_latency_ms=(150, 350),
    ),
    "gpt-4o-mini": ModelSpec(
        name="gpt-4o-mini",
        tier="cheap",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        simulated_latency_ms=(250, 500),
    ),
    "llama-3.1-70b-versatile": ModelSpec(
        name="llama-3.1-70b-versatile",
        tier="mid",
        input_price_per_1m=0.59,
        output_price_per_1m=0.79,
        simulated_latency_ms=(400, 800),
    ),
    "gpt-4o": ModelSpec(
        name="gpt-4o",
        tier="premium",
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
        simulated_latency_ms=(700, 1500),
    ),
}

TIER_DEFAULT_MODEL = {
    "cheap": "gpt-4o-mini",
    "mid": "llama-3.1-70b-versatile",
    "premium": "gpt-4o",
}


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_CATALOG:
        raise KeyError(
            f"Unknown model '{model_name}'. Known models: {list(MODEL_CATALOG)}"
        )
    return MODEL_CATALOG[model_name]


def compute_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    spec = get_model_spec(model_name)
    return (
        input_tokens / 1_000_000 * spec.input_price_per_1m
        + output_tokens / 1_000_000 * spec.output_price_per_1m
    )
