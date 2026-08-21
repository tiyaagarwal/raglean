"""End-to-end benchmark: quality check (small dataset, one pass, real baseline
comparison via the eval harness) + scale check (large simulated traffic, Zipfian
repeat pattern, cost/latency/cache numbers). Writes data/benchmark_report.json and
prints a human-readable summary. This is the script whose output numbers should be
quoted in the resume bullet / project writeup.

Run: `python -m raglean.benchmark.run_benchmark --n 500`
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..budget.tracker import BudgetTracker
from ..eval.dataset import load_eval_dataset
from ..eval.harness import EvalHarness
from ..pipeline import RAGLeanPipeline
from ..providers.mock_provider import MockProvider
from .synthetic_workload import generate_workload

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "benchmark_report.json"


def run_quality_check() -> dict:
    provider = MockProvider()
    tracker = BudgetTracker()
    pipeline = RAGLeanPipeline(provider=provider, budget_tracker=tracker, baseline_model="gpt-4o", context_token_budget=150)
    harness = EvalHarness(pipeline)
    report = harness.run(load_eval_dataset())
    report.pop("items", None)  # keep the top-level report compact; per-item detail is logged separately if needed
    return report


def run_scale_check(n_requests: int, seed: int) -> dict:
    provider = MockProvider()
    tracker = BudgetTracker()
    pipeline = RAGLeanPipeline(provider=provider, budget_tracker=tracker, baseline_model="gpt-4o", context_token_budget=150)

    workload = generate_workload(n_requests=n_requests, seed=seed)

    start = time.perf_counter()
    for item in workload:
        pipeline.run(
            query=item["query"],
            context_chunks=item["context_chunks"],
            compute_real_baseline=True,
        )
    wall_s = time.perf_counter() - start

    summary = tracker.summary()
    summary["n_requests"] = n_requests
    summary["wall_clock_s"] = wall_s
    return summary


def run(n_requests: int = 500, seed: int = 7, output_path: Path | None = None) -> dict:
    quality = run_quality_check()
    scale = run_scale_check(n_requests=n_requests, seed=seed)

    report = {"quality_check": quality, "scale_check": scale}

    output_path = output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    _print_report(report, output_path)
    return report


def _print_report(report: dict, output_path: Path) -> None:
    q = report["quality_check"]
    s = report["scale_check"]

    print("=" * 64)
    print("RAGLean benchmark report")
    print("=" * 64)
    print()
    print(f"[Quality check — {q['n_items']} dataset items, real baseline comparison]")
    print(f"  Mean answer F1 (optimized)  : {q['quality']['mean_optimized_f1']:.4f}")
    print(f"  Mean answer F1 (baseline)   : {q['quality']['mean_baseline_f1']:.4f}")
    print(f"  Items with no quality loss  : {q['quality']['pct_items_no_quality_loss']:.1f}%")
    print()
    print(f"[Scale check — {s['n_requests']} simulated requests, Zipfian repeat traffic]")
    print(f"  Cache hit rate              : {s['cache_hit_rate']*100:.1f}%")
    print(f"  Cost saved                  : ${s['cost_saved_usd']:.4f} ({s['cost_saved_pct']:.1f}%)")
    print(f"  Latency saved               : {s['latency_saved_pct']:.1f}%  "
          f"({s['avg_baseline_latency_ms']:.0f}ms -> {s['avg_actual_latency_ms']:.0f}ms avg)")
    print(f"  Tokens saved via compression: {s['tokens_saved']}")
    print(f"  Tier breakdown              : {s['tier_breakdown']}")
    print(f"  Wall clock for {s['n_requests']} requests   : {s['wall_clock_s']:.2f}s "
          f"(MockProvider, no real network calls)")
    print()
    print(f"Full report written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAGLean benchmark suite.")
    parser.add_argument("--n", type=int, default=500, help="number of simulated requests for the scale check")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(n_requests=args.n, seed=args.seed)
