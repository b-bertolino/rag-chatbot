"""
Custom exceptions raised by the core RAG modules.

Centralized here so the API layer (Phase 3) can import a single source
of truth for catch-and-translate-to-HTTP logic.

Convention: every exception's message is user-facing — it must read
cleanly when surfaced to a non-technical end user via the API.
"""


class IngestionError(Exception):
    """
    Raised when a document cannot be ingested at all.

    Use this for total failures (corrupted file, unsupported format,
    empty content). For partial failures (e.g. some PDF pages unreadable
    but others fine), prefer logging and returning a partial result.
    """