"""Configuration management for the multi-agent RAG system.

This module uses Pydantic Settings to load and validate environment variables
for OpenAI models, Pinecone settings, and other system parameters.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    # NOTE: these are optional at import time so the API can start;
    # runtime code should validate before use.
    openai_api_key: str | None = None
    openai_model_name: str = "gpt-4o-mini"

    # HuggingFace Embeddings (free, runs locally — no API key needed)
    hf_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Pinecone Configuration
    pinecone_api_key: str | None = None
    pinecone_index_name: str | None = None

    # Retrieval Configuration
    retrieval_k: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Create a singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the application settings instance (singleton pattern).

    Returns:
        Settings instance with all configuration values loaded.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
