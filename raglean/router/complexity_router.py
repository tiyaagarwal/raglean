"""Adaptive complexity router: scores each query and picks the cheapest model tier
that's likely to answer it correctly, instead of sending every query to the same
(usually over-provisioned) model.

v1 here is heuristic and explainable on purpose — a 4th-year project should be able to
show *why* a query was routed a certain way, which a black-box classifier makes harder
to demo/debug early on. `score()` is the one method a trained classifier (logistic
regression on embedding + hand features, or a distilled small model) would replace in a
v2 — everything else (tier thresholds, model selection, the proxy integration) stays
the same. See docs/ARCHITECTURE.md, section 6.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..providers.pricing import TIER_DEFAULT_MODEL

REASONING_KEYWORDS = (
    "why", "explain", "compare", "difference between", "trade-off", "tradeoff",
    "analyze", "evaluate", "pros and cons", "advantages and disadvantages",
    "what if", "how does", "relationship between", "implication", "root cause",
)

MULTI_HOP_CONNECTORS = (" and ", " then ", " after that ", " as well as ", " versus ", " vs ")


@dataclass
class RoutingDecision:
    tier: str
    model: str
    complexity_score: float
    signals: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in self.signals.items())
        return f"tier={self.tier} model={self.model} score={self.complexity_score:.2f} ({parts})"


class ComplexityRouter:
    def __init__(
        self,
        cheap_max: float = 0.35,
        mid_max: float = 0.65,
        model_for_tier: dict[str, str] | None = None,
        weights: dict[str, float] | None = None,
    ):
        """Queries scoring <= cheap_max go to the cheap tier, <= mid_max go to mid,
        everything above goes to premium.
        """
        self.cheap_max = cheap_max
        self.mid_max = mid_max
        self.model_for_tier = model_for_tier or dict(TIER_DEFAULT_MODEL)
        self.weights = weights or {
            "length": 0.25,
            "reasoning_keywords": 0.30,
            "multi_hop": 0.20,
            "retrieval_ambiguity": 0.25,
        }

    def score(self, query: str, retrieval_scores: list[float] | None = None) -> dict[str, float]:
        q_lower = query.lower()
        word_count = len(query.split())

        length_signal = min(1.0, word_count / 40.0)

        reasoning_signal = 1.0 if any(kw in q_lower for kw in REASONING_KEYWORDS) else 0.0

        multi_hop_signal = 1.0 if any(c in q_lower for c in MULTI_HOP_CONNECTORS) else 0.0
        multi_hop_signal = max(multi_hop_signal, min(1.0, (q_lower.count("?") - 1) / 2.0))

        if retrieval_scores:
            top = max(retrieval_scores)
            spread = max(retrieval_scores) - min(retrieval_scores)
            # Low top-score or high spread => retriever isn't confident => harder query.
            ambiguity_signal = max(0.0, 1.0 - top) * 0.6 + min(1.0, spread) * 0.4
        else:
            ambiguity_signal = 0.0

        return {
            "length": length_signal,
            "reasoning_keywords": reasoning_signal,
            "multi_hop": multi_hop_signal,
            "retrieval_ambiguity": ambiguity_signal,
        }

    def route(self, query: str, retrieval_scores: list[float] | None = None) -> RoutingDecision:
        signals = self.score(query, retrieval_scores)
        weighted = sum(self.weights[k] * v for k, v in signals.items())
        weighted = max(0.0, min(1.0, weighted))

        if weighted <= self.cheap_max:
            tier = "cheap"
        elif weighted <= self.mid_max:
            tier = "mid"
        else:
            tier = "premium"

        return RoutingDecision(
            tier=tier,
            model=self.model_for_tier[tier],
            complexity_score=weighted,
            signals=signals,
        )
