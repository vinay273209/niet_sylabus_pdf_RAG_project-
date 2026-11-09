from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pymupdf


SESSION_RE = re.compile(r"^20\d{2}[-–_ ]\d{2}$")
YEAR_RE = re.compile(
    r"^(?:(\d+)(?:st|nd|rd|th)|(first|second|third|fourth|fifth))\s+year$",
    re.IGNORECASE,
)
YEAR_NUMBERS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
}
SEMESTER_RE = re.compile(
    r"\bsemester\s*(?:no\.?\s*)?[:#-]?\s*"
    r"(VIII|VII|VI|V|IV|III|II|I|[1-8])\b",
    re.IGNORECASE,
)
UNIT_RE = re.compile(
    r"\bunit\s+(VIII|VII|VI|V|IV|III|II|I|IX|[1-9])\b",
    re.IGNORECASE,
)
COURSE_CODE_RE = re.compile(
    r"\b(?=[A-Z0-9-]{5,16}\b)(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)"
    r"[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\b"
)
LABELLED_CODE_RE = re.compile(
    r"\b(?:course|subject)\s+code\s*[:#-]\s*"
    r"([A-Z][A-Z0-9-]{3,15})\b",
    re.IGNORECASE,
)
LABELLED_SUBJECT_RE = re.compile(
    r"(?im)^\s*(?:course|subject)\s+(?:name|title)\s*[:#-]\s*([^\n]+)"
)
ROMAN_NUMERALS = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
}
VARIANT_PATTERNS = (
    re.compile(r"collaboration(?:\s+and)?\s+twin(?:ning|ing)?\s+program", re.I),
    re.compile(r"working\s+professional", re.I),
    re.compile(r"lateral\s+entry", re.I),
    re.compile(r"twin(?:ning|ing)?\s+program", re.I),
    re.compile(r"regular\s+program", re.I),
)


def _session_from_parts(parts: tuple[str, ...]) -> tuple[str | None, int | None]:
    for index, part in enumerate(parts):
        if SESSION_RE.fullmatch(part.strip()):
            session = re.sub(r"[–_ ]", "-", part.strip())
            return session, index
    return None, None


def _year_from_parts(parts: tuple[str, ...]) -> int | None:
    for part in parts:
        match = YEAR_RE.fullmatch(part.strip())
        if match:
            if match.group(1):
                return int(match.group(1))
            return YEAR_NUMBERS[match.group(2).lower()]
    return None


def _year_folder_candidates(parts: tuple[str, ...]) -> list[str]:
    return [
        part
        for part in parts[:-1]
        if "year" in part.lower()
        and YEAR_RE.fullmatch(part.strip()) is None
    ]


def _variant_from_parts(parts: tuple[str, ...]) -> str | None:
    text = " / ".join(parts)
    for pattern in VARIANT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _number_candidates(values: list[str], maximum: int) -> list[int]:
    return sorted(
        {
            number
            for value in values
            for number in [
                int(value) if value.isdigit() else ROMAN_NUMERALS.get(value.upper())
            ]
            if number is not None and number <= maximum
        }
    )


def _sample_paths(files: list[zipfile.ZipInfo]) -> set[str]:
    selected: set[str] = set()
    for session in ("2025-26", "2026-27"):
        for branch in ("CSE (IoT)", "CSE", "ECE"):
            candidate = next(
                (
                    info.filename
                    for info in files
                    if info.filename.startswith(f"{session}/{branch}/")
                    and "2nd Year" in info.filename
                ),
                None,
            )
            if candidate:
                selected.add(candidate)
    if files:
        selected.add(files[0].filename)
    return selected


def inspect_archive(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        pdfs = sorted(
            (
                info
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".pdf")
            ),
            key=lambda info: info.filename.casefold(),
        )
        sample_paths = _sample_paths(pdfs)
        documents: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        text_preview: dict[str, str] = {}
        content_layout_samples: dict[str, list[dict[str, Any]]] = {}
        branch_counts: Counter[str] = Counter()
        session_counts: Counter[str] = Counter()
        year_counts: Counter[str] = Counter()
        total_pages = 0
        total_text_chars = 0
        empty_text_pages = 0
        all_course_codes: Counter[str] = Counter()
        all_labelled_codes: Counter[str] = Counter()
        all_labelled_subjects: Counter[str] = Counter()
        all_semesters: Counter[str] = Counter()
        all_units: Counter[str] = Counter()
        pages_without_text: list[dict[str, Any]] = []
        year_folder_anomalies: Counter[str] = Counter()

        for info in pdfs:
            source_path = PurePosixPath(info.filename)
            parts = source_path.parts
            session, session_index = _session_from_parts(parts)
            branch = (
                parts[session_index + 1]
                if session_index is not None and session_index + 1 < len(parts) - 1
                else None
            )
            year = _year_from_parts(parts)
            raw = archive.read(info)
            digest = hashlib.sha256(raw).hexdigest()

            try:
                document = pymupdf.open(stream=raw, filetype="pdf")
                page_texts = [page.get_text("text") for page in document]
                page_image_counts = [
                    len(page.get_images(full=True)) for page in document
                ]
                pdf_metadata = document.metadata or {}
                page_count = len(document)
                document.close()
            except Exception as exc:
                errors.append(
                    {
                        "source_path": info.filename,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            text = "\n".join(page_texts)
            codes = sorted(set(COURSE_CODE_RE.findall(text)))
            labelled_codes = sorted(
                {value.upper() for value in LABELLED_CODE_RE.findall(text)}
            )
            labelled_subjects = sorted(
                {value.strip() for value in LABELLED_SUBJECT_RE.findall(text)}
            )
            semesters = _number_candidates(SEMESTER_RE.findall(text), 8)
            units = _number_candidates(UNIT_RE.findall(text), 9)
            total_pages += page_count
            total_text_chars += len(text)
            empty_pages = [
                page_number
                for page_number, page_text in enumerate(page_texts, start=1)
                if not page_text.strip()
            ]
            empty_text_pages += len(empty_pages)
            pages_without_text.extend(
                {
                    "source_path": info.filename,
                    "page": page_number,
                    "image_count": page_image_counts[page_number - 1],
                    "likely_scanned": page_image_counts[page_number - 1] > 0,
                }
                for page_number in empty_pages
            )
            all_course_codes.update(codes)
            all_labelled_codes.update(labelled_codes)
            all_labelled_subjects.update(labelled_subjects)
            all_semesters.update(str(value) for value in semesters)
            all_units.update(units)
            if branch:
                branch_counts[branch] += 1
            if session:
                session_counts[session] += 1
            if year is not None:
                year_counts[str(year)] += 1
            year_folder_anomalies.update(_year_folder_candidates(parts))

            documents.append(
                {
                    "source": source_path.name,
                    "source_path": info.filename,
                    "academic_session": session,
                    "branch": branch,
                    "year": year,
                    "program_variant": _variant_from_parts(parts),
                    "document_sha256": digest,
                    "page_count": page_count,
                    "text_char_count": len(text),
                    "empty_text_pages": sum(not page.strip() for page in page_texts),
                    "course_code_candidates": codes[:30],
                    "labelled_course_codes": labelled_codes,
                    "labelled_subject_names": labelled_subjects,
                    "semester_candidates": semesters,
                    "unit_candidates": units,
                    "pdf_title": pdf_metadata.get("title") or None,
                    "pdf_subject": pdf_metadata.get("subject") or None,
                }
            )
            if info.filename in sample_paths:
                first_text = next((page.strip() for page in page_texts if page.strip()), "")
                text_preview[info.filename] = first_text[:1800]
                matching_pages = [
                    {"page": page_number, "text": page_text.strip()[:1600]}
                    for page_number, page_text in enumerate(page_texts, start=1)
                    if re.search(
                        r"\b(course|subject)\s+(code|name|title)\b|"
                        r"\b(unit|semester)\s+[1-9ivx]+\b",
                        page_text,
                        re.IGNORECASE,
                    )
                ]
                content_layout_samples[info.filename] = matching_pages[:4]

    return {
        "archive": archive_path.name,
        "pdf_count": len(pdfs),
        "processed_pdf_count": len(documents),
        "failed_pdf_count": len(errors),
        "total_pages": total_pages,
        "total_extracted_text_characters": total_text_chars,
        "pages_with_no_extractable_text": empty_text_pages,
        "sessions": dict(sorted(session_counts.items())),
        "branches": dict(sorted(branch_counts.items())),
        "year_counts": dict(sorted(year_counts.items())),
        "unrecognized_year_folder_names": dict(
            sorted(year_folder_anomalies.items())
        ),
        "common_course_code_candidates": all_course_codes.most_common(100),
        "labelled_course_code_counts": all_labelled_codes.most_common(100),
        "labelled_subject_name_counts": all_labelled_subjects.most_common(100),
        "semester_candidate_document_counts": dict(sorted(all_semesters.items())),
        "unit_heading_candidates": all_units.most_common(),
        "pages_without_extractable_text": pages_without_text,
        "payload_schema": {
            "text": "string; required; one searchable chunk",
            "source": "string; original PDF filename",
            "source_path": "string; archive-relative path, retained for provenance",
            "academic_session": "string or null; from a matching folder name",
            "branch": "string or null; folder immediately below the session folder",
            "year": "positive integer or null; from an explicit year folder (programs may exceed four years)",
            "program_variant": "string or null; explicit program wording from filename/folders",
            "subject": "string or null; populate only when supported by nearby PDF text",
            "course_code": "string or null; populate from an explicit course/subject code label or a validated syllabus table row",
            "semester": "integer or null; chunk-local explicit evidence, not a document-wide guess",
            "unit": "integer or null; chunk-local explicit unit heading",
            "page": "integer; 1-based PDF page containing the chunk",
            "document_id": "stable ID derived from normalized source_path so same-named files in different folders stay distinct",
            "document_sha256": "string; source content fingerprint for incremental indexing",
            "chunk_index": "integer; zero-based within source document",
            "chunk_count": "integer; expected number of chunks for a complete document version",
            "chunk_id": "string; Qdrant point ID, deterministic per document version and chunk",
        },
        "metadata_policy": (
            "Folder-derived fields are used only when their folder is explicit. "
            "Semester, subject, course_code, and unit are chunk-local because one PDF "
            "can contain multiple subjects and semesters. Unknown values remain null."
        ),
        "documents": documents,
        "errors": errors,
        "sample_first_page_text": text_preview,
        "content_layout_samples": content_layout_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PDF text statistics and metadata evidence from a syllabus ZIP."
    )
    parser.add_argument("archive", type=Path, help="Path to Syllabus_Repository.zip")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/syllabus_preprocessing_report.json"),
        help="Path for the generated JSON report.",
    )
    args = parser.parse_args()
    report = inspect_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Processed {report['processed_pdf_count']}/{report['pdf_count']} PDFs; "
        f"wrote metadata evidence to {args.output}."
    )
    if report["failed_pdf_count"]:
        print(f"{report['failed_pdf_count']} PDFs could not be read; see report errors.")


if __name__ == "__main__":
    main()
