"""
Embedding model factory for the RAG pipeline.

We use intfloat/multilingual-e5-base — a multilingual model (Italian + English)
that produces 768-dimensional vectors. The E5 family was trained with a
contrastive objective using mandatory prefixes:

  - "query: "   for retrieval queries
  - "passage: " for documents being indexed

Omitting these prefixes does not crash the model, but produces out-of-distribution
embeddings and measurably degrades retrieval quality. The E5Embeddings wrapper
below enforces the correct prefix for each call site, so the rest of the codebase
can stay agnostic to this detail.

Reference: "Text Embeddings by Weakly-Supervised Contrastive Pre-training"
          (Wang et al., 2022) — https://arxiv.org/abs/2212.03533
"""
from functools import lru_cache
from typing import ClassVar

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


class E5Embeddings(HuggingFaceEmbeddings):
    """
    HuggingFaceEmbeddings subclass that prepends E5-required prefixes.

    Subclassing (instead of composition) lets us reuse all of HuggingFaceEmbeddings'
    model loading, device management, and encoding logic. We only override the two
    public methods where the prefixes need to be injected.
    """

    # ClassVar tells Pydantic these are constants, not model fields.
    # HuggingFaceEmbeddings is a Pydantic BaseModel, so un-annotated class
    # attributes would otherwise be interpreted as fields and fail validation.
    QUERY_PREFIX: ClassVar[str] = "query: "
    PASSAGE_PREFIX: ClassVar[str] = "passage: "

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for indexing — prepends 'passage: ' to each text."""
        prefixed = [f"{self.PASSAGE_PREFIX}{t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for retrieval — prepends 'query: '."""
        return super().embed_query(f"{self.QUERY_PREFIX}{text}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Return a process-wide singleton embeddings instance.

    We cache the instance because the underlying sentence-transformers model
    is ~278MB and takes a few seconds to load into memory. Instantiating it
    per-request would be unacceptable for a production API.

    lru_cache provides a simple, thread-safe singleton in CPython without
    the boilerplate of a module-level global or a metaclass.

    The return type is the abstract Embeddings interface (not E5Embeddings),
    so downstream consumers (vectorstore, ingestion, RAG engine) depend on
    LangChain's contract rather than our concrete implementation.

    Embeddings are L2-normalized (normalize_embeddings=True) because:
      1. E5 was trained expecting normalized vectors for retrieval.
      2. ChromaDB's default cosine distance is equivalent to dot product
         on normalized vectors, which is faster to compute.
    """
    return E5Embeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )