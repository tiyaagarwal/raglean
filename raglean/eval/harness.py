"""Eval harness: runs the dataset through the RAGLean pipeline with
`compute_real_baseline=True` so every item gets both an optimized answer and a true
unoptimized-baseline answer from the same provider, then compares each against the gold
answer with the offline F1 metric. This is the piece that turns "we saved cost" into
"we saved cost without answers getting worse" — the actual claim that matters.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from ..pipeline import RAGLeanPipeline
from .metrics import f1_score


@dataclass
class EvalItemResult:
    id: str
    query: str
    tag: str
    cache_hit: bool
    tier: str
    optimized_answer: str
    baseline_answer: str
    optimized_f1: float
    baseline_f1: float
    cost_saved_usd: float
    latency_saved_ms: float
    tokens_saved: int

    @property
    def f1_delta(self) -> float:
        return self.optimized_f1 - self.baseline_f1


class EvalHarness:
    def __init__(self, pipeline: RAGLeanPipeline):
        self.pipeline = pipeline

    def run(self, dataset: list[dict], repeat_fraction: float = 0.3) -> dict:
        """repeat_fraction re-runs a slice of the dataset a second time (simulating
        repeated user questions in real traffic) so the cache has something to hit —
        a single pass through a small, all-unique dataset would understate real-world
        cache effectiveness.
        """
        items = list(dataset)
        if repeat_fraction > 0:
            n_repeat = max(1, int(len(items) * repeat_fraction))
            items = items + items[:n_repeat]

        item_results: list[EvalItemResult] = []
        for item in items:
            result = self.pipeline.run(
                query=item["query"],
                context_chunks=item["context_chunks"],
                compute_real_baseline=True,
            )
            baseline_text = result.baseline.text if result.baseline else result.answer
            item_results.append(
                EvalItemResult(
                    id=item["id"],
                    query=item["query"],
                    tag=item.get("tag", "unknown"),
                    cache_hit=result.cache_hit,
                    tier=result.tier,
                    optimized_answer=result.answer,
                    baseline_answer=baseline_text,
                    optimized_f1=f1_score(result.answer, item["gold_answer"]),
                    baseline_f1=f1_score(baseline_text, item["gold_answer"]),
                    cost_saved_usd=result.event.cost_saved_usd,
                    latency_saved_ms=result.event.latency_saved_ms,
                    tokens_saved=result.event.tokens_saved,
                )
            )

        return self._aggregate(item_results)

    def _aggregate(self, results: list[EvalItemResult]) -> dict:
        n = len(results)
        opt_f1 = [r.optimized_f1 for r in results]
        base_f1 = [r.baseline_f1 for r in results]

        cost_summary = self.pipeline.budget_tracker.summary()

        return {
            "n_items": n,
            "quality": {
                "mean_optimized_f1": statistics.mean(opt_f1) if opt_f1 else 0.0,
                "mean_baseline_f1": statistics.mean(base_f1) if base_f1 else 0.0,
                "mean_f1_delta": statistics.mean(r.f1_delta for r in results) if results else 0.0,
                "pct_items_no_quality_loss": (
                    sum(1 for r in results if r.optimized_f1 >= r.baseline_f1 - 1e-9) / n * 100
                    if n else 0.0
                ),
            },
            "cost_and_latency": cost_summary,
            "by_tag": self._breakdown_by(results, "tag"),
            "items": [asdict(r) for r in results],
        }

    @staticmethod
    def _breakdown_by(results: list[EvalItemResult], field: str) -> dict:
        groups: dict[str, list[EvalItemResult]] = {}
        for r in results:
            groups.setdefault(getattr(r, field), []).append(r)
        out = {}
        for key, group in groups.items():
            n = len(group)
            out[key] = {
                "n": n,
                "mean_optimized_f1": statistics.mean(r.optimized_f1 for r in group),
                "mean_baseline_f1": statistics.mean(r.baseline_f1 for r in group),
                "cache_hit_rate": sum(1 for r in group if r.cache_hit) / n,
            }
        return out
