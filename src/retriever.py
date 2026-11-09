from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from config import Settings, get_settings
from src.embeddings import EmbeddingModel

FILTER_FIELDS = {
    "academic_session",
    "branch",
    "year",
    "semester",
    "subject",
    "course_code",
    "program_variant",
}
INTEGER_FILTER_FIELDS = {"year", "semester"}
RETURN_FIELDS = [
    "text",
    "document_id",
    "chunk_index",
    "source",
    "source_path",
    "page",
    "academic_session",
    "branch",
    "year",
    "semester",
    "subject",
    "course_code",
    "program_variant",
]


@dataclass(frozen=True, slots=True)
class SearchResult:
    text: str
    score: float
    source: str | None
    page: int | None
    academic_session: str | None
    branch: str | None
    year: int | None
    semester: int | None
    subject: str | None
    course_code: str | None
    program_variant: str | None
    document_id: str | None = None
    chunk_index: int | None = None

    @classmethod
    def from_point(cls, point: Any) -> SearchResult:
        payload = point.payload or {}
        return cls(
            text=str(payload.get("text") or ""),
            score=float(getattr(point, "score", 0.0)),
            source=payload.get("source"),
            page=payload.get("page"),
            academic_session=payload.get("academic_session"),
            branch=payload.get("branch"),
            year=payload.get("year"),
            semester=payload.get("semester"),
            subject=payload.get("subject"),
            course_code=payload.get("course_code"),
            program_variant=payload.get("program_variant"),
            document_id=payload.get("document_id"),
            chunk_index=payload.get("chunk_index"),
        )


def build_qdrant_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    if not filters:
        return None
    conditions: list[models.FieldCondition] = []
    for field, value in filters.items():
        if field not in FILTER_FIELDS:
            raise ValueError(f"Unsupported search filter: {field}")
        values = value if isinstance(value, (list, tuple, set)) else [value]
        if not values:
            continue
        if field in INTEGER_FILTER_FIELDS:
            if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
                raise ValueError(f"Filter '{field}' must contain integer values.")
        elif any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError(f"Filter '{field}' must contain non-empty strings.")
        match = (
            models.MatchAny(any=list(values))
            if len(values) > 1
            else models.MatchValue(value=values[0])
        )
        conditions.append(models.FieldCondition(key=field, match=match))
    return models.Filter(must=conditions) if conditions else None


class PdfRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: QdrantClient | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client
        self._embedder = embedder

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            url, api_key = self.settings.require_qdrant()
            self._client = QdrantClient(url=url, api_key=api_key, timeout=30)
        return self._client

    def _get_embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel(
                self.settings.embedding_model,
                self.settings.embedding_dimension,
            )
        return self._embedder

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def search_pdf(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 7,
    ) -> list[SearchResult]:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        vector = self._get_embedder().encode([question.strip()])[0]
        query_filter = build_qdrant_filter(filters)
        try:
            response = self._get_client().query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=RETURN_FIELDS,
                with_vectors=False,
            )
        except (UnexpectedResponse, ResponseHandlingException, httpx.HTTPError):
            raise
        return [
            SearchResult.from_point(point)
            for point in response.points
            if point.payload and point.payload.get("text")
        ]

    def get_page_chunks(self, result: SearchResult) -> list[SearchResult]:
        if result.document_id is None or result.page is None:
            return [result]
        points, _ = self._get_client().scroll(
            collection_name=self.settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=result.document_id),
                    ),
                    models.FieldCondition(
                        key="page",
                        match=models.MatchValue(value=result.page),
                    ),
                ]
            ),
            limit=32,
            with_payload=RETURN_FIELDS,
            with_vectors=False,
        )
        chunks = [
            SearchResult.from_point(point)
            for point in points
            if point.payload and point.payload.get("text")
        ]
        if not chunks:
            return [result]
        return sorted(
            chunks,
            key=lambda chunk: (
                chunk.chunk_index if chunk.chunk_index is not None else -1
            ),
        )


_default_retriever: PdfRetriever | None = None


def search_pdf(
    question: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 7,
) -> list[SearchResult]:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = PdfRetriever()
    return _default_retriever.search_pdf(question, filters=filters, top_k=top_k)
