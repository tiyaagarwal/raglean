"""Per-tenant cost accounting and budget enforcement, modeled on the multi-tenant
usage-metering pattern from the Cloud Carbon Tracker project. Every request logged here
records both the *actual* cost/latency (after cache/router/compression optimization)
and a *baseline* estimate (what it would have cost with no optimization — always the
premium-tier model, full uncompressed context, no cache), so savings are computed from
real numbers rather than asserted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    pass


@dataclass
class RequestEvent:
    tenant: str
    query: str
    cache_hit: bool
    tier: str
    model: str
    actual_cost_usd: float
    baseline_cost_usd: float
    actual_latency_ms: float
    baseline_latency_ms: float
    tokens_saved: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def cost_saved_usd(self) -> float:
        return max(0.0, self.baseline_cost_usd - self.actual_cost_usd)

    @property
    def latency_saved_ms(self) -> float:
        return max(0.0, self.baseline_latency_ms - self.actual_latency_ms)


@dataclass
class TenantLedger:
    tenant: str
    budget_cap_usd: float | None = None
    spent_usd: float = 0.0
    events: list[RequestEvent] = field(default_factory=list)

    @property
    def remaining_usd(self) -> float | None:
        if self.budget_cap_usd is None:
            return None
        return max(0.0, self.budget_cap_usd - self.spent_usd)


class BudgetTracker:
    def __init__(self):
        self._ledgers: dict[str, TenantLedger] = {}

    def get_or_create(self, tenant: str, budget_cap_usd: float | None = None) -> TenantLedger:
        if tenant not in self._ledgers:
            self._ledgers[tenant] = TenantLedger(tenant=tenant, budget_cap_usd=budget_cap_usd)
        return self._ledgers[tenant]

    def record(self, event: RequestEvent, enforce: bool = True) -> TenantLedger:
        ledger = self.get_or_create(event.tenant)
        projected = ledger.spent_usd + event.actual_cost_usd
        if enforce and ledger.budget_cap_usd is not None and projected > ledger.budget_cap_usd:
            raise BudgetExceededError(
                f"Tenant '{event.tenant}' would exceed budget cap "
                f"(${ledger.budget_cap_usd:.4f}); spent so far ${ledger.spent_usd:.4f}, "
                f"this request costs ${event.actual_cost_usd:.4f}."
            )
        ledger.spent_usd = projected
        ledger.events.append(event)
        return ledger

    def all_events(self) -> list[RequestEvent]:
        events: list[RequestEvent] = []
        for ledger in self._ledgers.values():
            events.extend(ledger.events)
        return sorted(events, key=lambda e: e.timestamp)

    def summary(self, tenant: str | None = None) -> dict:
        events = (
            self._ledgers[tenant].events
            if tenant and tenant in self._ledgers
            else self.all_events()
        )
        n = len(events)
        if n == 0:
            return {"requests": 0}

        cache_hits = sum(1 for e in events if e.cache_hit)
        actual_cost = sum(e.actual_cost_usd for e in events)
        baseline_cost = sum(e.baseline_cost_usd for e in events)
        actual_latency = sum(e.actual_latency_ms for e in events)
        baseline_latency = sum(e.baseline_latency_ms for e in events)
        tokens_saved = sum(e.tokens_saved for e in events)

        return {
            "requests": n,
            "cache_hit_rate": cache_hits / n,
            "actual_cost_usd": actual_cost,
            "baseline_cost_usd": baseline_cost,
            "cost_saved_usd": max(0.0, baseline_cost - actual_cost),
            "cost_saved_pct": (1 - actual_cost / baseline_cost) * 100 if baseline_cost else 0.0,
            "avg_actual_latency_ms": actual_latency / n,
            "avg_baseline_latency_ms": baseline_latency / n,
            "latency_saved_pct": (1 - actual_latency / baseline_latency) * 100 if baseline_latency else 0.0,
            "tokens_saved": tokens_saved,
            "tier_breakdown": {
                tier: sum(1 for e in events if e.tier == tier)
                for tier in sorted({e.tier for e in events})
            },
        }

    def to_records(self) -> list[dict]:
        """Flat list of dicts — easy to hand to pandas.DataFrame() for the dashboard,
        without importing pandas in this module.
        """
        return [
            {
                "timestamp": e.timestamp,
                "tenant": e.tenant,
                "query": e.query,
                "cache_hit": e.cache_hit,
                "tier": e.tier,
                "model": e.model,
                "actual_cost_usd": e.actual_cost_usd,
                "baseline_cost_usd": e.baseline_cost_usd,
                "cost_saved_usd": e.cost_saved_usd,
                "actual_latency_ms": e.actual_latency_ms,
                "baseline_latency_ms": e.baseline_latency_ms,
                "latency_saved_ms": e.latency_saved_ms,
                "tokens_saved": e.tokens_saved,
            }
            for e in self.all_events()
        ]
