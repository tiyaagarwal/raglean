# RAGLean — Architecture & Roadmap

## 1. Vision

RAGLean is a cost/latency optimization layer for production RAG (Retrieval-Augmented
Generation) pipelines. It sits between an application and its LLM provider(s) and
reduces spend and response time through retrieval-aware semantic caching, complexity-based
model routing, and context compression — while an eval harness proves that none of this
degrades answer quality.

General-purpose LLM gateways (LiteLLM, Portkey, Helicone) solve this at the API-call level.
RAGLean is scoped narrower and deeper: it understands the *retrieval* step of a RAG
pipeline, not just the final prompt, which lets it cache and compress in ways a generic
gateway cannot.

Positioning for a portfolio: not "we beat Portkey," but "here is where generic LLM
gateways fall short for RAG specifically, and a measured fix for that gap."

## 2. System overview

```
                         ┌─────────────────────────────────────────────┐
                         │                 RAGLean Proxy                │
   Client / RAG app ───▶ │  /v1/chat/completions  (FastAPI)             │
                         │                                               │
                         │   1. Semantic Cache  ──── hit ──▶ return      │
                         │        │ miss                                 │
                         │        ▼                                     │
                         │   2. Complexity Router  → picks model tier    │
                         │        │                                     │
                         │        ▼                                     │
                         │   3. Context Compressor → trims chunks        │
                         │        │                                     │
                         │        ▼                                     │
                         │   4. Provider call (mock / OpenAI-compatible) │
                         │        │                                     │
                         │        ▼                                     │
                         │   5. Budget Tracker → logs cost, enforces cap │
                         └─────────────────────────────────────────────┘
                                         │
                                         ▼
                         Eval Harness (offline) ── Benchmark Report
                         Streamlit Dashboard ── live metrics
```

## 3. Components

### 3.1 Provider abstraction (`raglean/providers/`)
Unified interface (`BaseProvider.generate`) so the rest of the system never talks to a
specific vendor SDK. Ships with:
- `MockProvider`: deterministic, latency-simulated, no API key required — used for
  development, tests, and reproducible benchmark numbers.
- `OpenAICompatibleProvider`: works against OpenAI, Groq, or Together by swapping
  `base_url` — used once real API keys are available.
- `pricing.py`: a swappable table of $/1K-token prices per model/tier. Marked
  illustrative and dated; meant to be refreshed from each provider's live pricing page
  before citing numbers publicly.

### 3.2 Semantic cache (`raglean/cache/`)
Retrieval-layer cache: embeds the *query*, not just the raw prompt string, and matches
against a FAISS index of prior queries within a similarity threshold. Cluster-aware
(Gaussian-Mixture / cosine-threshold clustering) so paraphrases of the same question
reuse the same answer. TTL + hit/miss counters for observability. Default embedding
backend is a small local sentence-transformer; falls back to a pure scikit-learn
hashing-vectorizer embedding if no network is available, so the whole system runs
offline for demos/CI.

### 3.3 Complexity router (`raglean/router/`)
Scores each incoming query for complexity (heuristics: question length, number of
sub-questions, presence of comparison/reasoning keywords, retrieval-score spread across
top-k chunks) and routes to a model tier: `cheap` (e.g. Llama-3.1-8B via Groq / GPT-4o-mini)
for simple factual lookups, `premium` for multi-hop/ambiguous queries. Interface is
pluggable so the heuristic can later be swapped for a trained classifier (logistic
regression or a distilled small model) without touching the rest of the pipeline.

### 3.4 Context compressor (`raglean/compression/`)
Given retrieved chunks + the query, scores sentences by relevance (embedding similarity
to the query) and greedily selects sentences until a token budget is hit, instead of
sending full raw chunks. Configurable compression ratio; reports tokens-saved.

### 3.5 Budget tracker (`raglean/budget/`)
Per-tenant running cost ledger and hard budget caps, modeled on the multi-tenant
architecture from the Cloud Carbon Tracker project. Emits structured events the
dashboard reads.

### 3.6 Proxy (`raglean/proxy/`)
FastAPI app exposing an OpenAI-compatible `/v1/chat/completions`-style endpoint (plus a
RAG-specific `/v1/rag/query` endpoint that also takes retrieved context). Wires cache →
router → compressor → provider → budget tracker together and returns the answer plus an
`x-raglean-meta` block (cache hit? which tier? tokens saved? cost saved?).

### 3.7 Eval harness (`raglean/eval/`)
Runs a fixed query set through (a) an unoptimized baseline pipeline and (b) the RAGLean
pipeline, on the same mock/real provider, and reports:
- cost delta, latency delta, cache hit rate
- answer-quality delta, measured via lexical/embedding overlap against gold answers
  (RAGAS-style faithfulness/answer-relevancy scoring is supported when a real judge LLM
  is configured; a lighter offline metric is the default so it runs without API keys).

### 3.8 Benchmark (`raglean/benchmark/`)
Synthetic workload generator that simulates realistic RAG traffic — including
paraphrased repeats (to exercise the cache), a mix of simple/complex queries (to
exercise the router), and long contexts (to exercise compression) — then runs it through
baseline vs. optimized pipelines and writes a report (`benchmark_report.json` +
printable summary) with real computed numbers.

### 3.9 Dashboard (`raglean/dashboard/`)
Streamlit app reading the budget tracker / benchmark logs: cache hit rate over time,
cost saved, latency distribution (baseline vs optimized), routing tier breakdown.

## 4. Tech stack

Python 3.10, FastAPI, Streamlit, FAISS, scikit-learn, sentence-transformers (optional),
Pydantic, pandas. Matches the stack already on the resume (FastAPI, Streamlit, FAISS,
Sentence Transformers, SQLAlchemy-style multi-tenant patterns).

## 5. Roadmap

**Phase 1 — Core pipeline (weeks 1-2):** provider abstraction, semantic cache, FastAPI
proxy skeleton, mock provider, basic tests.

**Phase 2 — Optimization layers (weeks 3-4):** complexity router, context compressor,
budget tracker wired into the proxy.

**Phase 3 — Proof (weeks 5-6):** eval harness, benchmark workload generator, first real
numbers.

**Phase 4 — Productization (weeks 7-8):** dashboard, budget/multi-tenant polish, SDK
ergonomics (two-line integration for a LangChain/LlamaIndex pipeline).

**Phase 5 — Publish (weeks 9-12):** package for PyPI, write the benchmark report as a
public writeup, deploy dashboard demo, record a walkthrough video, open to a few real
users if possible for genuine usage data.

## 6. Known gaps / improvisation notes

- Router starts heuristic, not ML-trained — upgrading it to a trained classifier is a
  natural "v2" extension and a good place to show ML depth later.
- Pricing table is illustrative and must be refreshed from live provider pricing before
  quoting numbers publicly.
- Real embedding backend (sentence-transformers) requires model download on first run;
  offline fallback exists so the project still runs in network-restricted environments
  (like CI or this sandbox).
- Eval harness's offline quality metric (lexical/embedding overlap) is a proxy for true
  answer quality; swapping in an LLM-judge (RAGAS) gives a stronger claim once API
  budget is available — documented as a config flag, not built by default.
