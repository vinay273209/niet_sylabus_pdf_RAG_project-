from __future__ import annotations

from dataclasses import dataclass

from src.metadata_extractor import extract_chunk_metadata


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    page: int
    chunk_index: int
    metadata: dict[str, object]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_pages(
    pages: tuple[str, ...],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        for text in _split_text(page_text, chunk_size, overlap):
            chunks.append(
                TextChunk(
                    text=text,
                    page=page_number,
                    chunk_index=len(chunks),
                    metadata=extract_chunk_metadata(text),
                )
            )
    return chunks
