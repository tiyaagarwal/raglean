from raglean.router.complexity_router import ComplexityRouter


def test_simple_query_routes_cheap():
    router = ComplexityRouter()
    decision = router.route("What is the capital of France?")
    assert decision.tier == "cheap"


def test_reasoning_query_routes_above_cheap():
    router = ComplexityRouter()
    decision = router.route(
        "Why does increasing the cache TTL improve hit rate but risk staleness, "
        "and how does that compare to lowering the similarity threshold?"
    )
    assert decision.tier in ("mid", "premium")
    assert decision.complexity_score > 0.35


def test_ambiguous_retrieval_pushes_complexity_up():
    router = ComplexityRouter()
    confident = router.route("What is the refund window?", retrieval_scores=[0.95, 0.93])
    ambiguous = router.route("What is the refund window?", retrieval_scores=[0.40, 0.10])
    assert ambiguous.complexity_score > confident.complexity_score
