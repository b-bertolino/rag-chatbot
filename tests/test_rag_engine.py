"""
Tests for the RAG engine (app/core/rag_engine.py).

Coverage strategy — three layers:
  - Helpers: deterministic, no model needed (formatters, dataclass).
  - Validation: input checks (empty question raises).
  - Composition: end-to-end chain wiring with the LLM mocked out via
    a RunnableLambda. This proves the LCEL pipeline is correctly composed
    without depending on Ollama or producing non-deterministic output.

We deliberately do NOT test the actual quality or content of LLM responses.
That's the job of Phase 5 benchmarks. Here we test our code, not Meta's
model.

Isolation strategy:
  - tmp_path-based vectorstore per test (same pattern as test_ingestion.py).
  - get_chain() and get_vectorstore() caches cleared between tests.
  - get_embeddings() cache shared (model load is expensive).
"""
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.core import rag_engine as rag_engine_module
from app.core import vectorstore as vectorstore_module
from app.core.config import settings
from app.core.ingestion import ingest_text
from app.core.rag_engine import (
    RAGAnswer,
    _docs_to_sources,
    _format_docs_with_sources,
    answer,
    get_chain,
)


@pytest.fixture
def isolated_chain(tmp_path, monkeypatch):
    """
    Per-test isolation for the full RAG stack.

    Points the vectorstore at a fresh temp directory and clears the cached
    singletons so both vectorstore and chain are rebuilt against it. The
    embeddings cache is intentionally left intact — reloading the model
    would multiply test time by an order of magnitude.
    """
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma_db"))
    vectorstore_module.get_vectorstore.cache_clear()
    rag_engine_module.get_chain.cache_clear()
    yield
    vectorstore_module.get_vectorstore.cache_clear()
    rag_engine_module.get_chain.cache_clear()


# ---------------------------------------------------------------------------
# Helper tests — pure functions, no I/O, no model.
# ---------------------------------------------------------------------------


def test_format_docs_with_sources_includes_filename_and_page():
    """
    PDF chunks (with 'page' metadata) must render with both source and
    page in the citation header. Page is converted from 0-indexed to
    1-indexed for human readability.
    """
    docs = [
        Document(
            page_content="Il fatturato Q3 è stato 1.2M.",
            metadata={"source": "report.pdf", "page": 2},
        ),
    ]
    rendered = _format_docs_with_sources(docs)

    assert "[Fonte: report.pdf, pagina 3]" in rendered  # 0-indexed → 1-indexed
    assert "Il fatturato Q3 è stato 1.2M." in rendered


def test_format_docs_with_sources_handles_text_without_page():
    """
    Plain-text chunks have no 'page' metadata. The header must omit the
    page reference rather than printing 'pagina None'.
    """
    docs = [
        Document(
            page_content="Roma è la capitale d'Italia.",
            metadata={"source": "rome.txt"},
        ),
    ]
    rendered = _format_docs_with_sources(docs)

    assert "[Fonte: rome.txt]" in rendered
    assert "pagina" not in rendered  # no page reference at all
    assert "Roma è la capitale d'Italia." in rendered


def test_format_docs_with_sources_handles_multiple_chunks():
    """Multiple chunks must be separated by blank lines for prompt clarity."""
    docs = [
        Document(page_content="Primo chunk.", metadata={"source": "a.txt"}),
        Document(page_content="Secondo chunk.", metadata={"source": "b.txt"}),
    ]
    rendered = _format_docs_with_sources(docs)

    assert "[Fonte: a.txt]" in rendered
    assert "[Fonte: b.txt]" in rendered
    # Two chunks → at least one blank line between them.
    assert "\n\n" in rendered


def test_docs_to_sources_truncates_preview():
    """
    The preview field of each source must be capped at 200 characters,
    so the API response stays compact even with long chunks.
    """
    long_text = "x" * 500
    docs = [
        Document(page_content=long_text, metadata={"source": "big.txt"}),
    ]
    sources = _docs_to_sources(docs)

    assert len(sources) == 1
    assert len(sources[0]["preview"]) == 200
    assert sources[0]["source"] == "big.txt"


# ---------------------------------------------------------------------------
# Validation tests — defensive checks on the public API.
# ---------------------------------------------------------------------------


def test_answer_raises_on_empty_question():
    """An empty string question is rejected before any LLM call."""
    with pytest.raises(ValueError, match="non-empty"):
        answer("")


def test_answer_raises_on_whitespace_only_question():
    """Whitespace-only questions are treated the same as empty."""
    with pytest.raises(ValueError, match="non-empty"):
        answer("   \n\t  ")


# ---------------------------------------------------------------------------
# Composition test — end-to-end wiring with a mocked LLM.
# ---------------------------------------------------------------------------


def test_get_chain_returns_singleton(isolated_chain):
    """The chain factory must return the same instance across calls."""
    first = get_chain()
    second = get_chain()
    assert first is second


def test_answer_returns_structured_result_with_mocked_llm(isolated_chain, monkeypatch):
    """
    End-to-end composition test with the LLM replaced by a deterministic
    RunnableLambda. Verifies that:
      - The retriever finds the ingested document.
      - The chain produces the canned answer string verbatim.
      - The sources field is populated with the expected metadata shape.

    A RunnableLambda is the cleanest LCEL substitute for an LLM: it
    implements the Runnable interface natively, so the chain accepts
    it without any mock magic. The chain has no idea it's not talking
    to a real model.
    """
    canned_answer = "Risposta canned di test."
    fake_llm = RunnableLambda(lambda _prompt: canned_answer)

    # Replace get_llm in the rag_engine module's namespace. We patch the
    # module-level reference (not the original llm module) because that's
    # the binding rag_engine actually uses when building the chain.
    monkeypatch.setattr(rag_engine_module, "get_llm", lambda: fake_llm)

    # Ingest a document so the retriever has something to find.
    ingest_text(
        "Roma è la capitale d'Italia ed è famosa per il Colosseo.",
        "rome.txt",
    )

    result = answer("Qual è la capitale d'Italia?")

    assert isinstance(result, RAGAnswer)
    assert result.answer == canned_answer
    assert len(result.sources) >= 1

    # The ingested document must appear among the cited sources.
    source_filenames = {s["source"] for s in result.sources}
    assert "rome.txt" in source_filenames

    # Each source has the expected shape.
    for src in result.sources:
        assert "source" in src
        assert "page" in src
        assert "preview" in src