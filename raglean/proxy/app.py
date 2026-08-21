"""FastAPI proxy: the SDK-level integration point.

Run: `uvicorn raglean.proxy.app:app --reload`

Two-line integration story for an existing RAG app: point your client at
`POST /v1/rag/query` instead of calling your LLM provider directly, with the same
`query` + `context_chunks` you'd already assembled from your retriever. The response
includes a `meta` block (cache hit? tier? tokens saved? cost saved?) so you can log/
observe the optimization without extra plumbing.

Uses MockProvider by default (no API key needed) so the demo runs standalone. Set
RAGLEAN_API_KEY + RAGLEAN_BASE_URL env vars to switch to OpenAICompatibleProvider.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..budget.tracker import BudgetExceededError, BudgetTracker
from ..pipeline import RAGLeanPipeline
from ..providers.mock_provider import MockProvider

app = FastAPI(title="RAGLean", version="0.1.0")


def _build_provider():
    if os.environ.get("RAGLEAN_API_KEY"):
        from ..providers.openai_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider()
    return MockProvider()


_provider = _build_provider()
_budget_tracker = BudgetTracker()
_pipeline = RAGLeanPipeline(provider=_provider, budget_tracker=_budget_tracker)


class RAGQueryRequest(BaseModel):
    query: str
    context_chunks: list[str] = Field(default_factory=list)
    tenant: str = "default"
    retrieval_scores: list[float] | None = None
    budget_cap_usd: float | None = None


class RAGQueryMeta(BaseModel):
    cache_hit: bool
    tier: str
    model: str
    cost_usd: float
    baseline_cost_usd: float
    cost_saved_usd: float
    latency_ms: float
    baseline_latency_ms: float
    tokens_saved: int


class RAGQueryResponse(BaseModel):
    answer: str
    meta: RAGQueryMeta


@app.post("/v1/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest) -> RAGQueryResponse:
    if req.budget_cap_usd is not None:
        _budget_tracker.get_or_create(req.tenant, budget_cap_usd=req.budget_cap_usd)

    try:
        result = _pipeline.run(
            query=req.query,
            context_chunks=req.context_chunks,
            tenant=req.tenant,
            retrieval_scores=req.retrieval_scores,
        )
    except BudgetExceededError as e:
        raise HTTPException(status_code=402, detail=str(e)) from e

    ev = result.event
    return RAGQueryResponse(
        answer=result.answer,
        meta=RAGQueryMeta(
            cache_hit=result.cache_hit,
            tier=result.tier,
            model=result.model,
            cost_usd=ev.actual_cost_usd,
            baseline_cost_usd=ev.baseline_cost_usd,
            cost_saved_usd=ev.cost_saved_usd,
            latency_ms=ev.actual_latency_ms,
            baseline_latency_ms=ev.baseline_latency_ms,
            tokens_saved=ev.tokens_saved,
        ),
    )


@app.get("/v1/tenants/{tenant}/summary")
def tenant_summary(tenant: str) -> dict:
    return _budget_tracker.summary(tenant=tenant)


@app.get("/v1/summary")
def overall_summary() -> dict:
    return _budget_tracker.summary()


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "provider": type(_provider).__name__}
