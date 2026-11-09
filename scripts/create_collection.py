from __future__ import annotations

import sys

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from config import get_settings


def main() -> int:
    settings = get_settings()
    try:
        url, api_key = settings.require_qdrant()
    except ValueError as exc:
        print(f"Collection creation not run: {exc}")
        return 2

    client = QdrantClient(url=url, api_key=api_key, timeout=15)
    try:
        if client.collection_exists(settings.qdrant_collection):
            print(
                f"Collection '{settings.qdrant_collection}' already exists; "
                "it was not changed."
            )
            return 1

        created = client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        if not created:
            print(
                f"Qdrant did not confirm creation of '{settings.qdrant_collection}'."
            )
            return 1

        info = client.get_collection(settings.qdrant_collection)
    except UnexpectedResponse as exc:
        print(
            f"Qdrant collection operation failed ({type(exc).__name__}, "
            f"HTTP {exc.status_code}). No existing collection was deleted."
        )
        return 1
    except (ResponseHandlingException, httpx.HTTPError, ValueError) as exc:
        print(
            f"Qdrant collection operation failed ({type(exc).__name__}). "
            "No existing collection was deleted."
        )
        return 1
    finally:
        client.close()

    vector_config = info.config.params.vectors
    if isinstance(vector_config, models.VectorParams):
        size = vector_config.size
        distance = vector_config.distance.value
    else:
        size = None
        distance = None
    print(
        f"Created empty collection '{settings.qdrant_collection}': "
        f"status={info.status}, points={info.points_count}, "
        f"vector_size={size}, distance={distance}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
