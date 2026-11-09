from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
import zipfile

from qdrant_client import models

from config import PROJECT_ROOT, ConfigurationError, get_settings
from src.chunking import chunk_pages
from src.metadata_extractor import extract_folder_metadata
from src.pdf_loader import discover_pdf_bytes, load_pdf
from src.qdrant_manager import QdrantManager, point_id

logger = logging.getLogger("syllabus_ingestion")


def _count_pdfs(source: Path) -> int:
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            return sum(
                not entry.is_dir() and entry.filename.lower().endswith(".pdf")
                for entry in archive.infolist()
            )
    if source.is_file() and source.suffix.lower() == ".pdf":
        return 1
    if source.is_dir():
        return sum(
            path.is_file() and path.suffix.lower() == ".pdf"
            for path in source.rglob("*")
        )
    raise ValueError(f"PDF source must be a PDF, ZIP archive, or directory: {source}")


def _process_pdf(
    source_path: str,
    raw: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[dict[str, Any], list[Any]]:
    document = load_pdf(source_path, raw)
    metadata = extract_folder_metadata(document.source_path)
    chunks = chunk_pages(document.pages, chunk_size, chunk_overlap)
    return (
        {
            **metadata,
            "document_sha256": document.sha256,
        },
        chunks,
    )


def run_ingestion(source: Path, dry_run: bool = False) -> int:
    settings = get_settings()
    manager: QdrantManager | None = None
    try:
        if not dry_run:
            manager = QdrantManager(settings)
            manager.ensure_filter_indexes()
            existing = manager.existing_documents()
        else:
            existing = {}
    except (ConfigurationError, ValueError, OSError) as exc:
        if manager:
            manager.close()
        logger.error("Cannot prepare Qdrant ingestion: %s", exc)
        return 2
    except Exception as exc:
        if manager:
            manager.close()
        logger.error("Cannot prepare Qdrant ingestion (%s).", type(exc).__name__)
        return 2

    embedder = None
    processed = 0
    skipped = 0
    failed = 0
    total_chunks = 0
    uploaded_chunks = 0
    completed_files = 0
    total_files = _count_pdfs(source)
    try:
        for source_path, raw in discover_pdf_bytes(source):
            completed_files += 1
            try:
                document, chunks = _process_pdf(
                    source_path,
                    raw,
                    settings.chunk_size,
                    settings.chunk_overlap,
                )
            except (ValueError, OSError) as exc:
                failed += 1
                logger.warning(
                    "[%d/%d] Failed %s: %s",
                    completed_files,
                    total_files,
                    source_path,
                    exc,
                )
                continue
            except Exception as exc:
                failed += 1
                logger.warning(
                    "[%d/%d] Failed unreadable PDF %s (%s).",
                    completed_files,
                    total_files,
                    source_path,
                    type(exc).__name__,
                )
                continue

            total_chunks += len(chunks)
            if dry_run:
                processed += 1
                logger.info(
                    "[%d/%d] Preprocessed %s (%d chunks).",
                    completed_files,
                    total_files,
                    source_path,
                    len(chunks),
                )
                continue

            if manager is None:
                raise RuntimeError("Qdrant manager was not initialized.")
            doc_id = str(document["document_id"])
            prior = existing.get(doc_id, {})
            versions = prior.get("versions", {})
            version = (
                versions.get(str(document["document_sha256"]), {})
                if isinstance(versions, dict)
                else {}
            )
            previous_indices = (
                version.get("chunk_indices", set())
                if isinstance(version, dict)
                else set()
            )
            if (
                isinstance(version, dict)
                and version.get("chunk_count") == len(chunks)
                and previous_indices == set(range(len(chunks)))
            ):
                skipped += 1
                logger.info(
                    "[%d/%d] Unchanged %s; skipped embedding.",
                    completed_files,
                    total_files,
                    source_path,
                )
                continue
            if not chunks:
                failed += 1
                logger.warning(
                    "[%d/%d] Failed %s: no extractable text chunks.",
                    completed_files,
                    total_files,
                    source_path,
                )
                continue

            if embedder is None:
                from src.embeddings import EmbeddingModel

                embedder = EmbeddingModel(
                    settings.embedding_model,
                    settings.embedding_dimension,
                )

            vectors = embedder.encode(
                [chunk.text for chunk in chunks],
                batch_size=settings.embedding_batch_size,
            )
            points: list[models.PointStruct] = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                payload = {
                    **document,
                    **chunk.metadata,
                    "text": chunk.text,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "chunk_count": len(chunks),
                    "embedding_model": settings.embedding_model,
                }
                qdrant_point_id = point_id(
                    doc_id,
                    str(document["document_sha256"]),
                    chunk.chunk_index,
                )
                payload["chunk_id"] = qdrant_point_id
                points.append(
                    models.PointStruct(
                        id=qdrant_point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            manager.upsert(points)
            uploaded_chunks += len(points)
            old_ids = prior.get("point_ids", [])
            new_ids = {point.id for point in points}
            stale_ids = (
                [point_id_value for point_id_value in old_ids if point_id_value not in new_ids]
                if isinstance(old_ids, list)
                else []
            )
            if stale_ids:
                manager.delete_point_ids(stale_ids)
            existing[doc_id] = {
                "versions": {
                    str(document["document_sha256"]): {
                        "point_ids": [point.id for point in points],
                        "chunk_indices": set(range(len(points))),
                        "chunk_count": len(points),
                    }
                },
                "point_ids": [point.id for point in points],
            }
            processed += 1
            logger.info(
                "[%d/%d] Indexed %s: %d vectors; PDFs indexed=%d, unchanged=%d, failed=%d.",
                completed_files,
                total_files,
                source_path,
                len(points),
                processed,
                skipped,
                failed,
            )
    finally:
        if manager:
            manager.close()

    logger.info(
        "%s complete: PDFs=%d/%d, indexed=%d, unchanged=%d, failed=%d, "
        "chunks_seen=%d, vectors_uploaded=%d.",
        "Dry run" if dry_run else "Ingestion",
        completed_files,
        total_files,
        processed,
        skipped,
        failed,
        total_chunks,
        uploaded_chunks,
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract, chunk, and incrementally index syllabus PDFs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "Syllabus_Repository.zip",
        help="PDF file, PDF directory, or ZIP archive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk PDFs without embeddings or Qdrant writes.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.input.exists():
        logger.error("PDF input does not exist: %s", args.input)
        return 2
    try:
        return run_ingestion(args.input, dry_run=args.dry_run)
    except Exception as exc:
        logger.exception("Ingestion stopped (%s).", type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
