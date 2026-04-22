"""
LLM provider factory for the RAG pipeline.

Two providers are supported, selected at runtime via the LLM_PROVIDER env var:
  - "ollama" (default): local inference, zero cost, data never leaves the machine.
  - "groq":             free-tier cloud API with LPU-accelerated inference
                        (used for deployment to Render / HF Spaces in Phase 7).

Both providers return a LangChain BaseChatModel, so downstream consumers
(rag_engine) can call .invoke() / .stream() / .batch() without knowing which
backend is behind the call.

Provider-specific packages (langchain_ollama, langchain_groq) are imported
lazily inside the builder functions. This means: (a) if Ollama is broken,
Groq-only deployments still start cleanly, (b) Docker images in Phase 6 can
drop unused provider packages without code changes.
"""
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.core.config import settings


def _build_ollama() -> BaseChatModel:
    """
    Build a ChatOllama client pointing at the configured local Ollama server.

    No eager health-check on localhost:11434 — doing so would add a network
    round-trip on every module import, which is wasteful in tests and in CI.
    If the server is down, the first .invoke() call will surface a clear
    connection error from ChatOllama itself.
    """
    # Lazy import: only pay for langchain_ollama if LLM_PROVIDER=ollama.
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )


def _build_groq() -> BaseChatModel:
    """
    Build a ChatGroq client using the configured API key.

    Unlike Ollama, cloud credentials are validated eagerly: an empty API key
    is a deploy-time configuration error, cheap to detect at build time, and
    the failure mode (HTTP 401 from Groq) is much noisier to debug at runtime.
    """
    if not settings.groq_api_key:
        raise ValueError(
            "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is empty. "
            "Set it in your .env file or switch LLM_PROVIDER to 'ollama'."
        )

    # Lazy import: only pay for langchain_groq if LLM_PROVIDER=groq.
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return a process-wide singleton LLM client for the configured provider.

    Dispatch is a plain if/else on settings.llm_provider. For two providers
    this is more readable than a registry pattern; migration to a registry
    is mechanical if a third provider is ever added.

    The return type is LangChain's BaseChatModel abstract base, not the
    concrete ChatOllama / ChatGroq, so consumers depend on the interface
    contract rather than the implementation.

    Like get_embeddings(), this uses @lru_cache(maxsize=1) as a pythonic
    singleton: lazy loading on first call, thread-safe under CPython's GIL,
    zero boilerplate compared to a module-level global or a metaclass.
    """
    if settings.llm_provider == "ollama":
        return _build_ollama()
    if settings.llm_provider == "groq":
        return _build_groq()

    # pydantic-settings validates llm_provider as Literal["ollama", "groq"],
    # so this branch is theoretically unreachable. Kept as a defensive guard
    # in case someone bypasses config validation (e.g. monkeypatching in tests).
    raise ValueError(
        f"Unknown LLM provider: {settings.llm_provider!r}. "
        "Expected 'ollama' or 'groq'."
    )