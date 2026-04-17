"""
Application settings loaded from environment variables.

Using pydantic-settings gives us validation, type safety, and a single
source of truth for configuration. This is a production pattern.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    llm_provider: Literal["ollama", "groq"] = Field(
        default="ollama",
        description="Which LLM backend to use",
    )

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Groq (cloud, free tier)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Embeddings (always local sentence-transformers)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector DB
    chroma_path: str = "./chroma_db"
    collection_name: str = "documents"

    # RAG params
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4

    # Observability (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        """LangFuse tracing is only enabled if both keys are set."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root."""
        return Path(__file__).resolve().parent.parent.parent


# Singleton instance imported everywhere else
settings = Settings()
