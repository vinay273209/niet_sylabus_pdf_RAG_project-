from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from config import get_settings


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def inspect_collection() -> int:
    settings = get_settings()
    try:
        url, api_key = settings.require_qdrant()
    except ValueError as exc:
        print(f"Collection inspection not run: {exc}")
        return 2

    client = QdrantClient(url=url, api_key=api_key, timeout=15)
    try:
        info = client.get_collection(settings.qdrant_collection)
        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=20,
            with_payload=True,
            with_vectors=False,
        )
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            try:
                collection_names = [
                    collection.name
                    for collection in client.get_collections().collections
                ]
            except (UnexpectedResponse, ResponseHandlingException, httpx.HTTPError) as list_exc:
                print(
                    f"Collection '{settings.qdrant_collection}' was not found; "
                    f"listing collections failed ({type(list_exc).__name__})."
                )
                return 1
            print(
                json.dumps(
                    {
                        "requested_collection": settings.qdrant_collection,
                        "exists": False,
                        "available_collections": collection_names,
                        "collection_modified": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(
                f"Qdrant inspection failed ({type(exc).__name__}, "
                f"HTTP {exc.status_code})."
            )
        return 1
    except (ResponseHandlingException, httpx.HTTPError, ValueError) as exc:
        print(f"Qdrant inspection failed ({type(exc).__name__}).")
        return 1
    finally:
        client.close()

    payload_schema = getattr(info, "payload_schema", None) or {}
    field_types: dict[str, set[str]] = {}
    examples: dict[str, list[Any]] = {}
    for point in points:
        for key, value in (point.payload or {}).items():
            field_types.setdefault(key, set()).add(type(value).__name__)
            if key.lower() in {"text", "content"} or "text" in key.lower():
                continue
            values = examples.setdefault(key, [])
            if value is not None and value not in values and len(values) < 3:
                values.append(value[:120] if isinstance(value, str) else value)

    output = {
        "collection": settings.qdrant_collection,
        "status": str(info.status),
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "vectors_config": jsonable(info.config.params.vectors),
        "payload_schema": jsonable(payload_schema),
        "sample_size": len(points),
        "observed_payload_types": {
            key: sorted(values) for key, values in sorted(field_types.items())
        },
        "non_text_metadata_examples": dict(sorted(examples.items())),
        "vectors_downloaded": False,
        "collection_modified": False,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(inspect_collection())
