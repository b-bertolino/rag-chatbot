"""
FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.core.config import settings

app = FastAPI(
    title="RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot with free, local LLMs",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
    }
