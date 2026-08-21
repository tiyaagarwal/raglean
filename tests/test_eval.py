from raglean.budget.tracker import BudgetTracker
from raglean.eval.dataset import load_eval_dataset
from raglean.eval.harness import EvalHarness
from raglean.eval.metrics import f1_score
from raglean.pipeline import RAGLeanPipeline
from raglean.providers.mock_provider import MockProvider


def test_f1_score_basic():
    assert f1_score("30 days", "30 days") == 1.0
    assert f1_score("completely unrelated text", "30 days") == 0.0
    assert 0 < f1_score("the refund window is 30 days", "30 days") < 1.0


def test_eval_harness_runs_full_dataset_and_saves_cost():
    provider = MockProvider()
    tracker = BudgetTracker()
    pipeline = RAGLeanPipeline(provider=provider, budget_tracker=tracker, baseline_model="gpt-4o")
    harness = EvalHarness(pipeline)

    report = harness.run(load_eval_dataset())

    assert report["n_items"] > 0
    assert report["cost_and_latency"]["cost_saved_pct"] > 0
    # Optimized quality shouldn't collapse relative to baseline.
    assert report["quality"]["mean_optimized_f1"] >= report["quality"]["mean_baseline_f1"] * 0.5
