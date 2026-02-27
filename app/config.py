"""Application configuration — loads environment variables with validation."""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


class ConfigError(Exception):
    """Raised when a required configuration value is missing or invalid."""
    pass


class Settings:
    """Central configuration loaded from environment variables."""

    def __init__(self):
        # --- LLM Provider ---
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

        # --- API Keys ---
        self.GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

        # --- Embedding ---
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # --- ChromaDB ---
        self.CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

        # --- Chunking ---
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

        # --- Retrieval ---
        self.TOP_K: int = int(os.getenv("TOP_K", "5"))

    def validate(self):
        """Validate that all required configuration is present."""
        # Check API key for the chosen provider
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEY:
            raise ConfigError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )
        if self.LLM_PROVIDER not in ("groq", "openai"):
            raise ConfigError(
                f"LLM_PROVIDER must be 'groq' or 'openai', got '{self.LLM_PROVIDER}'."
            )

        # Numeric sanity checks
        if self.CHUNK_SIZE <= 0:
            raise ConfigError("CHUNK_SIZE must be a positive integer.")
        if self.CHUNK_OVERLAP < 0:
            raise ConfigError("CHUNK_OVERLAP must be >= 0.")
        if self.TOP_K <= 0:
            raise ConfigError("TOP_K must be a positive integer.")

        return True

    @property
    def active_api_key(self) -> str:
        """Return the API key for the currently selected provider."""
        if self.LLM_PROVIDER == "groq":
            return self.GROQ_API_KEY
        return self.OPENAI_API_KEY


# Singleton instance
settings = Settings()
