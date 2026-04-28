"""
Vector store factory for the RAG pipeline.

Wraps ChromaDB (via langchain-chroma) into a process-wide singleton that
exposes the standard LangChain VectorStore interface to ingestion and
retrieval components.

Design choices:
  - Persistent on disk (settings.chroma_path), not in-memory: documents
    ingested today must be queryable after a server restart.
  - Cosine distance (hnsw:space=cosine), not Chroma's L2 default: the
    convention for text retrieval with sentence-transformers, semantically
    correct because E5 encodes meaning in vector direction. Decision is
    immutable per collection — must be set on first creation.
  - Embedding function injected at construction: the store uses our
    E5Embeddings (with prefixes) for both writes and queries.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from app.core.config import settings
from app.core.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> VectorStore:
    """
    Return a process-wide singleton vector store backed by persistent Chroma.

    The store is configured with cosine distance because our E5 embeddings
    are L2-normalized and encode semantics in their direction — cosine is
    the geometrically correct similarity for this representation.

    Note: collection_metadata is honored by Chroma only on FIRST creation
    of the collection. If the collection already exists on disk with a
    different metric, the existing config wins. To change the metric,
    delete the persistence directory and re-ingest.

    Returns the abstract VectorStore interface (not Chroma) so downstream
    consumers depend on LangChain's contract rather than our implementation.
    """
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path,
        collection_metadata={"hnsw:space": "cosine"},
    )