"""Core orchestration: cache -> router -> compressor -> provider -> budget tracker.

This is the single piece of logic reused by the FastAPI proxy (`proxy/app.py`), the
benchmark script, and the eval harness, so "what RAGLean actually does to a request" is
defined in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass

from .budget.tracker import BudgetTracker, RequestEvent
from .cache.semantic_cache import CacheEntry, SemanticCache
from .compression.context_compressor import CompressionResult, ContextCompressor
from .providers.base import BaseProvider, LLMResponse
from .providers.pricing import compute_cost_usd, get_model_spec
from .router.complexity_router import ComplexityRouter, RoutingDecision


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class PipelineResult:
    answer: str
    cache_hit: bool
    tier: str
    model: str
    actual: LLMResponse | None
    baseline: LLMResponse | None
    compression: CompressionResult | None
    routing: RoutingDecision | None
    event: RequestEvent


class RAGLeanPipeline:
    def __init__(
        self,
        provider: BaseProvider,
        cache: SemanticCache | None = None,
        router: ComplexityRouter | None = None,
        compressor: ContextCompressor | None = None,
        budget_tracker: BudgetTracker | None = None,
        baseline_model: str = "gpt-4o",
        context_token_budget: int = 300,
        enable_cache: bool = True,
        enable_router: bool = True,
        enable_compression: bool = True,
    ):
        self.provider = provider
        self.enable_cache = enable_cache
        self.enable_router = enable_router
        self.enable_compression = enable_compression

        self.cache = cache if cache is not None else (SemanticCache() if enable_cache else None)
        self.router = router if router is not None else (ComplexityRouter() if enable_router else None)
        self.compressor = compressor if compressor is not None else (
            ContextCompressor(token_budget=context_token_budget) if enable_compression else None
        )
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.baseline_model = baseline_model

    def _build_prompt(self, query: str, context_text: str) -> str:
        if context_text:
            return f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        return f"Question: {query}\nAnswer:"

    def run(
        self,
        query: str,
        context_chunks: list[str] | None = None,
        tenant: str = "default",
        retrieval_scores: list[float] | None = None,
        enforce_budget: bool = True,
        compute_real_baseline: bool = False,
    ) -> PipelineResult:
        """compute_real_baseline=True runs an actual second (unoptimized) generation for
        an apples-to-apples comparison — used by the benchmark/eval harness, where the
        provider is a free/simulated MockProvider. Left False for the live proxy path,
        where doubling every real API call to measure "what we saved" would defeat the
        entire point of the optimization.
        """
        context_chunks = context_chunks or []

        cache_entry: CacheEntry | None = self.cache.lookup(query) if self.enable_cache else None
        compression_result: CompressionResult | None = None
        routing_decision: RoutingDecision | None = None
        actual_response: LLMResponse | None = None

        if cache_entry is not None:
            model_used = cache_entry.response.model
            tier_used = "cache"
            answer = cache_entry.response.text
            actual_cost = 0.0
            actual_latency = 1.0  # local vector lookup, effectively instant
        else:
            if self.enable_router:
                routing_decision = self.router.route(query, retrieval_scores)
                model_used = routing_decision.model
                tier_used = routing_decision.tier
            else:
                model_used = self.baseline_model
                tier_used = "premium"

            if self.enable_compression and context_chunks:
                compression_result = self.compressor.compress(query, context_chunks)
                context_text = compression_result.compressed_text
            else:
                context_text = "\n".join(context_chunks)

            prompt = self._build_prompt(query, context_text)
            actual_response = self.provider.generate(prompt, model_used)
            answer = actual_response.text
            actual_cost = actual_response.cost_usd
            actual_latency = actual_response.latency_ms

            if self.enable_cache:
                self.cache.store(query, actual_response)

        baseline_response: LLMResponse | None = None
        if compute_real_baseline:
            baseline_prompt = self._build_prompt(query, "\n".join(context_chunks))
            baseline_response = self.provider.generate(baseline_prompt, self.baseline_model)
            baseline_cost = baseline_response.cost_usd
            baseline_latency = baseline_response.latency_ms
        else:
            full_prompt = self._build_prompt(query, "\n".join(context_chunks))
            input_tokens = _approx_tokens(full_prompt)
            output_tokens = max(8, int(input_tokens * 0.35))
            baseline_cost = compute_cost_usd(self.baseline_model, input_tokens, output_tokens)
            lo, hi = get_model_spec(self.baseline_model).simulated_latency_ms
            baseline_latency = (lo + hi) / 2

        tokens_saved = compression_result.tokens_saved if compression_result else 0

        event = RequestEvent(
            tenant=tenant,
            query=query,
            cache_hit=cache_entry is not None,
            tier=tier_used,
            model=model_used,
            actual_cost_usd=actual_cost,
            baseline_cost_usd=baseline_cost,
            actual_latency_ms=actual_latency,
            baseline_latency_ms=baseline_latency,
            tokens_saved=tokens_saved,
        )
        self.budget_tracker.record(event, enforce=enforce_budget)

        return PipelineResult(
            answer=answer,
            cache_hit=cache_entry is not None,
            tier=tier_used,
            model=model_used,
            actual=actual_response,
            baseline=baseline_response,
            compression=compression_result,
            routing=routing_decision,
            event=event,
        )
