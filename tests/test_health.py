"""
Smoke test for Phase 1 — validates the app boots and /health responds.

Run with: pytest tests/ -v
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    """The /health endpoint should return 200 and status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_exposes_config():
    """The /health endpoint should expose the active LLM provider."""
    response = client.get("/health")
    data = response.json()
    assert "llm_provider" in data
    assert data["llm_provider"] in ("ollama", "groq")
