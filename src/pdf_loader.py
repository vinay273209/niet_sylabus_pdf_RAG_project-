from __future__ import annotations

import hashlib
import logging
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pymupdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PdfDocument:
    source_path: str
    filename: str
    sha256: str
    pages: tuple[str, ...]


def discover_pdf_bytes(source: Path) -> Iterator[tuple[str, bytes]]:
    source = source.resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for info in sorted(
                (
                    entry
                    for entry in archive.infolist()
                    if not entry.is_dir() and entry.filename.lower().endswith(".pdf")
                ),
                key=lambda entry: entry.filename.casefold(),
            ):
                yield str(PurePosixPath(info.filename)), archive.read(info)
        return
    if source.is_file() and source.suffix.lower() == ".pdf":
        yield source.name, source.read_bytes()
        return
    if source.is_dir():
        for path in sorted(source.rglob("*"), key=lambda item: str(item).casefold()):
            if path.is_file() and path.suffix.lower() == ".pdf":
                yield path.relative_to(source).as_posix(), path.read_bytes()
        return
    raise ValueError(f"PDF source must be a PDF, ZIP archive, or directory: {source}")


def load_pdf(source_path: str, raw: bytes) -> PdfDocument:
    if not raw.startswith(b"%PDF-"):
        raise ValueError("File does not have a valid PDF signature.")
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")
            pages = tuple(page.get_text("text") for page in document)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise ValueError(f"Unreadable PDF: {type(exc).__name__}") from exc
    if not pages or not any(page.strip() for page in pages):
        raise ValueError("PDF contains no extractable text.")
    path = PurePosixPath(source_path)
    return PdfDocument(
        source_path=source_path,
        filename=path.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        pages=pages,
    )
