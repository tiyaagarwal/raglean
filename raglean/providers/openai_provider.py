"""Real provider, used once API keys are available. Works with OpenAI directly, or with
any OpenAI-compatible endpoint (Groq, Together, etc.) by overriding `base_url` — that's
the whole point of routing "cheap" tier queries to Groq-hosted Llama and "premium" tier
to OpenAI's GPT-4o through the *same* client interface.

Not imported by default anywhere in the package (see providers/__init__.py) so the
project has zero hard dependency on the `openai` SDK or a network call for its tests,
benchmarks, or CI.
"""
from __future__ import annotations

import os
import time

from .base import BaseProvider, LLMResponse


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The `openai` package is required for OpenAICompatibleProvider. "
                "Run `pip install openai`."
            ) from e
        self._client = OpenAI(
            api_key=api_key or os.environ.get("RAGLEAN_API_KEY"),
            base_url=base_url or os.environ.get("RAGLEAN_BASE_URL"),
        )

    def generate(self, prompt: str, model: str, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = resp.usage
        return self._finalize(
            text=resp.choices[0].message.content or "",
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )
