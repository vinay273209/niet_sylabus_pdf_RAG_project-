from __future__ import annotations

import uuid
from collections.abc import Iterator

from qdrant_client import QdrantClient, models

from config import Settings

PAYLOAD_INDEXES = {
    "academic_session": models.PayloadSchemaType.KEYWORD,
    "branch": models.PayloadSchemaType.KEYWORD,
    "program_variant": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "subject": models.PayloadSchemaType.KEYWORD,
    "course_code": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "semester": models.PayloadSchemaType.INTEGER,
    "page": models.PayloadSchemaType.INTEGER,
}


def point_id(document_id: str, document_sha256: str, chunk_index: int) -> str:
    return str(
        uuid.uuid5(
            uuid.UUID(document_id),
            f"{document_sha256}:{chunk_index}",
        )
    )


class QdrantManager:
    def __init__(self, settings: Settings) -> None:
        url, api_key = settings.require_qdrant()
        self.collection = settings.qdrant_collection
        self.dimension = settings.embedding_dimension
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)

    def close(self) -> None:
        self.client.close()

    def _validate_collection(self) -> None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise ValueError("Collection must use one unnamed dense vector.")
        if vectors.size != self.dimension:
            raise ValueError(
                f"Collection dimension is {vectors.size}, expected {self.dimension}."
            )
        if vectors.distance != models.Distance.COSINE:
            raise ValueError("Collection distance must be Cosine.")

    def ensure_filter_indexes(self) -> None:
        self._validate_collection()
        info = self.client.get_collection(self.collection)
        existing = getattr(info, "payload_schema", None) or {}
        for field_name, schema in PAYLOAD_INDEXES.items():
            if field_name not in existing:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )

    def existing_documents(self) -> dict[str, dict[str, object]]:
        existing: dict[str, dict[str, object]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=["document_id", "document_sha256"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                document_id = payload.get("document_id")
                if not isinstance(document_id, str):
                    continue
                entry = existing.setdefault(
                    document_id,
                    {"versions": {}, "point_ids": []},
                )
                point_ids = entry["point_ids"]
                versions = entry["versions"]
                content_hash = payload.get("document_sha256")
                if isinstance(versions, dict) and isinstance(content_hash, str):
                    version = versions.setdefault(
                        content_hash,
                        {
                            "point_ids": [],
                            "chunk_indices": set(),
                            "chunk_count": None,
                        },
                    )
                    version_ids = version["point_ids"]
                    chunk_indices = version["chunk_indices"]
                    if isinstance(version_ids, list):
                        version_ids.append(point.id)
                    if isinstance(chunk_indices, set) and isinstance(
                        payload.get("chunk_index"), int
                    ):
                        chunk_indices.add(payload["chunk_index"])
                    if isinstance(payload.get("chunk_count"), int):
                        version["chunk_count"] = payload["chunk_count"]
                if isinstance(point_ids, list):
                    point_ids.append(point.id)
            if offset is None:
                break
        return existing

    def upsert(self, points: list[models.PointStruct]) -> None:
        self.client.upload_points(
            collection_name=self.collection,
            points=points,
            batch_size=128,
            parallel=1,
            max_retries=3,
            wait=True,
        )

    def delete_point_ids(self, point_ids: list[int | str]) -> None:
        for start in range(0, len(point_ids), 256):
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(
                    points=point_ids[start : start + 256]
                ),
                wait=True,
            )
