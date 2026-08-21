"""Unified LLM provider interface. Every optimization layer (cache, router, compressor,
budget tracker) talks to `BaseProvider.generate(...)` only — never to a vendor SDK
directly — so swapping OpenAI <-> Groq <-> Together <-> a local model is a one-line
config change, not a refactor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .pricing import compute_cost_usd


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool = False


class BaseProvider(ABC):
    """Implementations must be deterministic-enough to be testable and must report
    real token counts so cost accounting is accurate.
    """

    @abstractmethod
    def generate(self, prompt: str, model: str, max_tokens: int = 512) -> LLMResponse:
        raise NotImplementedError

    def _finalize(
        self, text: str, model: str, input_tokens: int, output_tokens: int, latency_ms: float
    ) -> LLMResponse:
        cost = compute_cost_usd(model, input_tokens, output_tokens)
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )
