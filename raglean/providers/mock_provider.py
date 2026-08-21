"""Deterministic, latency-simulated provider — no API key required.

This is what makes the whole project runnable and benchmarkable offline: every
optimization (caching, routing, compression) is provider-agnostic, so we can prove the
*system* saves cost/latency/tokens using MockProvider, then swap in
OpenAICompatibleProvider once real API keys are available, without changing any
downstream code or invalidating the architecture.

Token counting uses a simple whitespace-based approximation (~1.3 tokens/word), which is
close enough for relative (baseline vs optimized) comparisons — the eval harness cares
about deltas, not absolute token-perfect counts.
"""
from __future__ import annotations

import hashlib
import random
import time

from .base import BaseProvider, LLMResponse
from .pricing import get_model_spec


def _approx_tokens(text: str) -> int:
    words = max(1, len(text.split()))
    return int(words * 1.3)


class MockProvider(BaseProvider):
    def __init__(self, seed: int = 42, simulate_sleep: bool = False):
        """simulate_sleep=True actually sleeps for the sampled latency (realistic wall
        clock time, useful for a live demo); False (default) just records the sampled
        latency without blocking, so benchmarks over thousands of queries run fast.
        """
        self._seed = seed
        self._simulate_sleep = simulate_sleep

    def _rng_for(self, prompt: str, model: str) -> random.Random:
        # Deterministic per (prompt, model) so repeated benchmark runs are reproducible.
        h = hashlib.sha256(f"{self._seed}:{model}:{prompt}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def generate(self, prompt: str, model: str, max_tokens: int = 512) -> LLMResponse:
        spec = get_model_spec(model)
        rng = self._rng_for(prompt, model)

        lo, hi = spec.simulated_latency_ms
        latency_ms = rng.uniform(lo, hi)
        if self._simulate_sleep:
            time.sleep(latency_ms / 1000.0)

        input_tokens = _approx_tokens(prompt)
        # Cheaper/smaller models tend to be terser; simulate that lightly.
        tier_len_factor = {"cheap": 0.6, "mid": 0.85, "premium": 1.0}.get(spec.tier, 0.8)
        output_tokens = min(max_tokens, max(8, int(input_tokens * 0.35 * tier_len_factor)))

        text = self._synthesize_answer(prompt, output_tokens)

        cost = self._cost(model, input_tokens, output_tokens)
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        from .pricing import compute_cost_usd

        return compute_cost_usd(model, input_tokens, output_tokens)

    @staticmethod
    def _synthesize_answer(prompt: str, target_tokens: int) -> str:
        """Extractive stand-in for a real completion: pulls the most content-bearing
        sentences out of the prompt/context so answer text is at least topically
        grounded, keeping the eval harness's overlap-based quality metric meaningful.
        """
        sentences = [s.strip() for s in prompt.replace("\n", " ").split(".") if s.strip()]
        if not sentences:
            return "No answer could be generated from the given context."
        # Take the last 1-3 sentences (typically the actual question / most specific
        # part of a RAG prompt) as the seed for a templated answer.
        seed = ". ".join(sentences[-2:])
        words = seed.split()
        target_words = max(6, int(target_tokens / 1.3))
        answer_words = (words * ((target_words // max(1, len(words))) + 1))[:target_words]
        return "Based on the provided context: " + " ".join(answer_words) + "."
