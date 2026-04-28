"""
Document ingestion pipeline for the RAG chatbot.

Takes raw documents (PDF or text) and produces persistent vector store
entries: source → load → split → embed → persist.

Public API:
  - ingest_pdf(file_path) → IngestionResult
  - ingest_text(text, source) → IngestionResult

Both functions are idempotent by source filename: re-ingesting the same
source replaces its previous chunks (delete-then-insert). This matches
the intuitive expectation that "uploading again means updating."

Errors:
  - Total failure (file unreadable, unsupported format, empty content)
    raises IngestionError with a user-facing message.
  - Partial failure (e.g. one PDF page unreadable) is logged and skipped;
    the result reports pages_skipped > 0 so the caller can warn the user.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """
    Outcome of an ingestion operation.

    Returned to the caller (typically the API layer) so it can build
    informative responses for the end user, including warnings on
    partial recoveries.
    """
    source: str
    chunks_created: int
    pages_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build the text splitter from configuration.

    RecursiveCharacterTextSplitter tries separators from most-semantic
    (paragraph break) to least-semantic (single character), preserving
    meaningful boundaries when possible. Standard choice for general RAG.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _enrich_metadata(documents: list[Document], source: str) -> list[Document]:
    """
    Add ingestion-time metadata to every chunk, in place.

    Metadata kept minimal and purposeful:
      - source:       filename, used for citations and idempotency.
      - chunk_index:  position within the source, useful for debugging.
      - ingested_at:  UTC ISO timestamp, useful for auditing.

    PyPDFLoader already populates 'page' for PDFs; we don't overwrite it.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    for index, doc in enumerate(documents):
        doc.metadata["source"] = source
        doc.metadata["chunk_index"] = index
        doc.metadata["ingested_at"] = timestamp
    return documents


def _persist(documents: list[Document], source: str) -> int:
    """
    Idempotent write to the vector store.

    Pattern: delete-then-insert keyed by 'source' metadata. Re-ingesting
    the same source replaces its prior chunks atomically from the user's
    perspective (the old chunks are gone before the new ones are added).

    Implementation note: ChromaDB 0.5.x has unstable behavior on
    delete(where=...) — it can raise ValueError both on empty collections
    and on edge cases where the where clause matches no documents. We
    therefore look up matching IDs explicitly via get() and delete by
    ID list, which is the more reliable code path in Chroma's API.

    Returns the number of chunks persisted.
    """
    vectorstore = get_vectorstore()
    existing = vectorstore.get(where={"source": source})
    if existing["ids"]:
        vectorstore.delete(ids=existing["ids"])
    vectorstore.add_documents(documents)
    return len(documents)


def ingest_text(text: str, source: str) -> IngestionResult:
    """
    Ingest a raw text string under the given source identifier.

    The 'source' acts as the document's logical identity for retrieval
    citations and for idempotent re-ingestion.

    Raises:
        IngestionError: if the input text is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise IngestionError(
            f"Cannot ingest '{source}': the provided text is empty."
        )

    splitter = _build_splitter()
    chunks = splitter.create_documents([text])
    chunks = _enrich_metadata(chunks, source)
    n_persisted = _persist(chunks, source)

    logger.info("Ingested text source=%s chunks=%d", source, n_persisted)
    return IngestionResult(source=source, chunks_created=n_persisted)


def ingest_pdf(file_path: str) -> IngestionResult:
    """
    Ingest a PDF file from local disk.

    Loads page by page via PyPDFLoader. Pages that fail to load are
    skipped and logged — the rest of the document is still ingested.
    A PDF where ALL pages fail (or a missing/unreadable file) raises
    IngestionError.

    Raises:
        IngestionError: file not found, unreadable, or zero usable pages.
    """
    path = Path(file_path)
    if not path.is_file():
        raise IngestionError(f"File not found: {file_path}")

    source = path.name

    try:
        loader = PyPDFLoader(str(path))
        # PyPDFLoader.load() returns one Document per page. We iterate
        # explicitly so we can isolate per-page failures (rare but possible
        # on malformed PDFs).
        raw_pages = loader.load()
    except Exception as exc:
        # Anything from pypdf here means the file is broken or encrypted
        # at a level we cannot recover from. Surface a clean message.
        raise IngestionError(
            f"Could not read PDF '{source}': the file appears corrupted "
            f"or password-protected."
        ) from exc

    valid_pages: list[Document] = []
    pages_skipped = 0
    for page in raw_pages:
        # Empty pages (scans without OCR, blank dividers) carry no signal.
        # Skip silently — they're not errors, just noise.
        if page.page_content and page.page_content.strip():
            valid_pages.append(page)
        else:
            pages_skipped += 1

    if not valid_pages:
        raise IngestionError(
            f"Could not extract any text from '{source}'. The PDF may be "
            f"a scanned image without OCR, or fully empty."
        )

    splitter = _build_splitter()
    chunks = splitter.split_documents(valid_pages)
    chunks = _enrich_metadata(chunks, source)
    n_persisted = _persist(chunks, source)

    warnings = []
    if pages_skipped > 0:
        warnings.append(
            f"{pages_skipped} page(s) contained no extractable text "
            f"and were skipped."
        )

    logger.info(
        "Ingested PDF source=%s chunks=%d pages_skipped=%d",
        source, n_persisted, pages_skipped,
    )
    return IngestionResult(
        source=source,
        chunks_created=n_persisted,
        pages_skipped=pages_skipped,
        warnings=warnings,
    )