from fastapi.testclient import TestClient

from raglean.proxy.app import app

client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_rag_query_returns_meta():
    r = client.post(
        "/v1/rag/query",
        json={
            "query": "What is the refund window?",
            "context_chunks": ["The refund window is 30 days for physical products."],
            "tenant": "test-tenant",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "cache_hit" in body["meta"]
    assert body["meta"]["cost_usd"] >= 0


def test_tenant_summary_endpoint():
    client.post(
        "/v1/rag/query",
        json={"query": "How long is the warranty?", "context_chunks": [], "tenant": "summary-tenant"},
    )
    r = client.get("/v1/tenants/summary-tenant/summary")
    assert r.status_code == 200
    assert r.json()["requests"] >= 1
