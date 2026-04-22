"""
Unit tests for the LLM factory (app/core/llm.py).

Unlike test_embeddings.py, these tests do NOT hit the real LLM — neither
Ollama nor Groq. We only verify that get_llm() dispatches to the correct
builder and that builders construct the right client with the right config.

Why no real invocation:
  - Calling .invoke() on Ollama requires a running local server, making the
    suite flaky in CI.
  - Calling .invoke() on Groq requires a real API key and burns free-tier
    quota on every test run.
  - LLM output is non-deterministic, so "assert response == X" is unreliable
    by design. Response quality is a Phase 5 concern (benchmarks), not a
    unit-test concern.

What we DO test: dispatch logic, eager validation, singleton caching.
These are fast tests with no external dependencies — no `slow` marker.
"""
import pytest

from app.core.config import settings
from app.core.llm import _build_groq, get_llm


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    """
    Reset the @lru_cache singleton before AND after every test.

    Without this, a test that monkeypatches settings.llm_provider="groq"
    would poison the cache for subsequent tests — they'd receive the Groq
    client even after the monkeypatch is reverted.

    autouse=True applies this fixture to every test in the module without
    requiring it as an explicit argument.
    """
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


def test_get_llm_returns_singleton():
    """The factory must return the same instance on repeated calls."""
    first = get_llm()
    second = get_llm()
    assert first is second, "get_llm() should return a cached singleton"


def test_ollama_provider_builds_correct_client():
    """
    With LLM_PROVIDER=ollama (default), get_llm() must return a ChatOllama
    configured with the model from settings.

    We check the class name via type().__name__ rather than isinstance()
    to avoid importing ChatOllama in the test module — the lazy-import
    design of llm.py is preserved.
    """
    llm = get_llm()
    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == settings.ollama_model


def test_groq_provider_builds_correct_client(monkeypatch):
    """
    With LLM_PROVIDER=groq and a non-empty GROQ_API_KEY, get_llm() must
    return a ChatGroq client.

    monkeypatch swaps settings values for the duration of this test only —
    the original values are restored automatically on teardown. No real
    network call is made: ChatGroq's constructor does not validate the
    API key, it only fails on first .invoke().
    """
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "fake-key-for-test")

    llm = get_llm()
    assert type(llm).__name__ == "ChatGroq"


def test_groq_provider_raises_without_api_key(monkeypatch):
    """
    Eager validation: if LLM_PROVIDER=groq but GROQ_API_KEY is empty,
    the builder must raise ValueError with a helpful message.

    This guards decision #27 in CLAUDE.md — failing fast on missing cloud
    credentials is cheaper than failing at first .invoke() with an opaque
    HTTP 401.
    """
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "")

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        _build_groq()