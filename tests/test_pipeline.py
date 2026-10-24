from raglean.budget.tracker import BudgetTracker
from raglean.pipeline import RAGLeanPipeline
from raglean.providers.mock_provider import MockProvider


def _make_pipeline():
    provider = MockProvider()
    tracker = BudgetTracker()
    return RAGLeanPipeline(provider=provider, budget_tracker=tracker, baseline_model="gpt-4o"), tracker


def test_second_identical_query_is_a_cache_hit():
    pipeline, _ = _make_pipeline()
    chunks = ["The refund window is 30 days for physical products."]

    first = pipeline.run("What is the refund window?", context_chunks=chunks)
    second = pipeline.run("What is the refund window?", context_chunks=chunks)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.event.actual_cost_usd == 0.0


def test_optimized_cost_is_never_more_than_baseline():
    pipeline, _ = _make_pipeline()
    chunks = ["The refund window is 30 days for physical products."] * 3

    result = pipeline.run(
        "What is the refund window?", context_chunks=chunks, compute_real_baseline=True
    )
    assert result.event.actual_cost_usd <= result.event.baseline_cost_usd


def test_budget_enforcement_raises_through_pipeline():
    pipeline, tracker = _make_pipeline()
    tracker.get_or_create("tight", budget_cap_usd=0.0000001)

    import pytest

    from raglean.budget.tracker import BudgetExceededError

    with pytest.raises(BudgetExceededError):
        pipeline.run(
            "Explain, in detail, the entire refund and warranty policy end to end.",
            context_chunks=["Some fairly long context. " * 20],
            tenant="tight",
        )


def test_cache_is_isolated_per_tenant():
    pipeline = RAGLeanPipeline(provider=MockProvider())
    query = "What is the refund window?"

    pipeline.run(query, ["Acme refunds within 30 days."], tenant="acme")
    assert pipeline.run(query, ["Acme refunds within 30 days."], tenant="acme").cache_hit

    other = pipeline.run(query, ["Globex refunds within 7 days."], tenant="globex")
    assert not other.cache_hit
