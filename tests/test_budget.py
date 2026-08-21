import pytest

from raglean.budget.tracker import BudgetExceededError, BudgetTracker, RequestEvent


def _event(tenant="acme", cost=0.01, baseline=0.05):
    return RequestEvent(
        tenant=tenant,
        query="q",
        cache_hit=False,
        tier="cheap",
        model="gpt-4o-mini",
        actual_cost_usd=cost,
        baseline_cost_usd=baseline,
        actual_latency_ms=100,
        baseline_latency_ms=500,
    )


def test_record_and_summary():
    tracker = BudgetTracker()
    tracker.record(_event())
    tracker.record(_event())
    summary = tracker.summary()
    assert summary["requests"] == 2
    assert summary["cost_saved_usd"] == pytest.approx(0.08)


def test_budget_cap_enforced():
    tracker = BudgetTracker()
    tracker.get_or_create("acme", budget_cap_usd=0.015)
    tracker.record(_event(cost=0.01))
    with pytest.raises(BudgetExceededError):
        tracker.record(_event(cost=0.01))


def test_budget_cap_can_be_bypassed_explicitly():
    tracker = BudgetTracker()
    tracker.get_or_create("acme", budget_cap_usd=0.015)
    tracker.record(_event(cost=0.01))
    tracker.record(_event(cost=0.01), enforce=False)  # explicit override
    assert tracker.summary("acme")["requests"] == 2
