from raglean.cache.semantic_cache import SemanticCache
from raglean.providers.mock_provider import MockProvider


def test_exact_repeat_hits_cache():
    provider = MockProvider()
    cache = SemanticCache(similarity_threshold=0.8)

    query = "What is the refund window?"
    assert cache.lookup(query) is None

    resp = provider.generate(query, "gpt-4o-mini")
    cache.store(query, resp)

    hit = cache.lookup(query)
    assert hit is not None
    assert hit.response.text == resp.text
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_unrelated_query_misses():
    provider = MockProvider()
    cache = SemanticCache(similarity_threshold=0.8)

    resp = provider.generate("What is the refund window?", "gpt-4o-mini")
    cache.store("What is the refund window?", resp)

    assert cache.lookup("What is the weather in Bangalore today?") is None


def test_ttl_expiry_evicts_entry():
    provider = MockProvider()
    cache = SemanticCache(similarity_threshold=0.8, ttl_s=0.01)

    query = "How long is the warranty?"
    resp = provider.generate(query, "gpt-4o-mini")
    cache.store(query, resp)
    assert cache.lookup(query) is not None

    import time

    time.sleep(0.05)
    assert cache.lookup(query) is None


def test_namespaces_do_not_share_entries():
    provider = MockProvider()
    cache = SemanticCache(similarity_threshold=0.8)

    query = "What is the refund window?"
    cache.store(query, provider.generate(query, "gpt-4o-mini"), namespace="tenant-a")

    assert cache.lookup(query, namespace="tenant-b") is None
    assert cache.lookup(query) is None
    assert cache.lookup(query, namespace="tenant-a") is not None


def test_eviction_spans_namespaces():
    provider = MockProvider()
    cache = SemanticCache(similarity_threshold=0.8, max_entries=2)

    cache.store("What is the refund window?", provider.generate("q1", "gpt-4o-mini"), namespace="a")
    cache.store("How long is the warranty?", provider.generate("q2", "gpt-4o-mini"), namespace="b")
    cache.store("Do you ship to Canada?", provider.generate("q3", "gpt-4o-mini"), namespace="b")

    assert len(cache) == 2
    assert cache.lookup("What is the refund window?", namespace="a") is None
    assert cache.lookup("Do you ship to Canada?", namespace="b") is not None
