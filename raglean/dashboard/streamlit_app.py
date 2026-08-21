"""Observability dashboard. Run with:

    streamlit run raglean/dashboard/streamlit_app.py

Uses MockProvider + the synthetic workload generator so it's runnable standalone with
no API keys — in production this would instead read live events off the running
proxy's BudgetTracker (e.g. via a shared DB/queue instead of in-process state).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raglean.benchmark.synthetic_workload import generate_workload  # noqa: E402
from raglean.budget.tracker import BudgetTracker  # noqa: E402
from raglean.pipeline import RAGLeanPipeline  # noqa: E402
from raglean.providers.mock_provider import MockProvider  # noqa: E402

st.set_page_config(page_title="RAGLean Dashboard", layout="wide")
st.title("RAGLean — RAG Cost & Latency Optimization Dashboard")
st.caption(
    "Simulated traffic via MockProvider — no API keys required. "
    "See docs/ARCHITECTURE.md for what swapping in a real provider/embedder looks like."
)

with st.sidebar:
    st.header("Simulation settings")
    n = st.slider("Simulated requests", 50, 2000, 500, step=50)
    seed = st.number_input("Random seed", value=7, step=1)
    context_budget = st.slider("Compression token budget", 50, 500, 150, step=10)
    zipf_skew = st.slider(
        "Traffic skew (higher = more repeat questions)", 0.5, 2.5, 1.2, step=0.1
    )
    run_clicked = st.button("Run simulation", type="primary")

needs_run = "summary" not in st.session_state or run_clicked
if needs_run:
    provider = MockProvider()
    tracker = BudgetTracker()
    pipeline = RAGLeanPipeline(
        provider=provider,
        budget_tracker=tracker,
        baseline_model="gpt-4o",
        context_token_budget=context_budget,
    )
    workload = generate_workload(n_requests=n, seed=int(seed), zipf_skew=zipf_skew)
    with st.spinner(f"Running {n} simulated requests through the pipeline..."):
        for item in workload:
            pipeline.run(
                query=item["query"],
                context_chunks=item["context_chunks"],
                compute_real_baseline=True,
            )
    st.session_state["df"] = pd.DataFrame(tracker.to_records())
    st.session_state["summary"] = tracker.summary()

df: pd.DataFrame = st.session_state["df"]
summary: dict = st.session_state["summary"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cache hit rate", f"{summary['cache_hit_rate'] * 100:.1f}%")
col2.metric(
    "Cost saved",
    f"{summary['cost_saved_pct']:.1f}%",
    f"-${summary['cost_saved_usd']:.4f} of ${summary['baseline_cost_usd']:.4f}",
)
col3.metric("Latency saved", f"{summary['latency_saved_pct']:.1f}%")
col4.metric("Tokens saved (compression)", f"{summary['tokens_saved']:,}")

st.subheader("Cumulative cost: baseline vs. optimized")
df_sorted = df.sort_values("timestamp").reset_index(drop=True)
df_sorted["cumulative_optimized_usd"] = df_sorted["actual_cost_usd"].cumsum()
df_sorted["cumulative_baseline_usd"] = df_sorted["baseline_cost_usd"].cumsum()
st.line_chart(df_sorted[["cumulative_optimized_usd", "cumulative_baseline_usd"]])

left, right = st.columns(2)

with left:
    st.subheader("Routing tier breakdown")
    tier_counts = df["tier"].value_counts().rename_axis("tier").reset_index(name="requests")
    st.bar_chart(tier_counts.set_index("tier"))

with right:
    st.subheader("Latency distribution (ms)")
    bins = np.linspace(0, max(df["baseline_latency_ms"].max(), 1), 20)
    actual_hist, _ = np.histogram(df["actual_latency_ms"], bins=bins)
    baseline_hist, _ = np.histogram(df["baseline_latency_ms"], bins=bins)
    hist_df = pd.DataFrame(
        {"optimized": actual_hist, "baseline": baseline_hist},
        index=[f"{int(b)}" for b in bins[:-1]],
    )
    st.bar_chart(hist_df)

st.subheader("Recent request log")
st.dataframe(
    df.sort_values("timestamp", ascending=False).head(200),
    use_container_width=True,
)

st.caption(
    "Cost figures use the illustrative pricing table in providers/pricing.py — refresh "
    "against live provider pricing before quoting these numbers publicly."
)
