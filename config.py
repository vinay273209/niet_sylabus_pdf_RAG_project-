from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Settings:
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = "gemini-3.5-flash"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_collection: str = "pdf_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 150
    embedding_batch_size: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE must be greater than zero.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
            )
        if self.embedding_batch_size <= 0:
            raise ConfigurationError("EMBEDDING_BATCH_SIZE must be greater than zero.")

    def require_gemini(self) -> str:
        if not self.gemini_api_key:
            raise ConfigurationError("Missing required setting: GEMINI_API_KEY")
        return self.gemini_api_key

    def require_qdrant(self) -> tuple[str, str]:
        url = self.qdrant_url
        api_key = self.qdrant_api_key
        if not url or not api_key:
            missing = [
                name
                for name, value in (
                    ("QDRANT_URL", url),
                    ("QDRANT_API_KEY", api_key),
                )
                if not value
            ]
            raise ConfigurationError(
                f"Missing required setting(s): {', '.join(missing)}"
            )
        return url, api_key


def get_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip(),
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "pdf_documents").strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
    )
