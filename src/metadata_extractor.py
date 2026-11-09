from __future__ import annotations

import re
import uuid
from pathlib import PurePosixPath

SESSION_RE = re.compile(r"^20\d{2}[-–_ ]\d{2}$")
YEAR_RE = re.compile(
    r"^(?:(\d+)(?:st|nd|rd|th)|(first|second|third|fourth|fifth))\s+year$",
    re.IGNORECASE,
)
YEAR_NAMES = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
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
    r"\b(?:course|subject)\s+code\s*[:#-]\s*"
    r"([A-Z][A-Z0-9-]{3,15})\b",
    re.IGNORECASE,
)
SUBJECT_RE = re.compile(
    r"(?im)^\s*(?:course|subject)\s+(?:name|title)\s*[:#-]\s*([^\n]+)"
)
VARIANT_RE = re.compile(
    r"collaboration(?:\s+and)?\s+twin(?:ning|ing)?\s+program|"
    r"working\s+professional|lateral\s+entry|twin(?:ning|ing)?\s+program|"
    r"regular\s+program",
    re.IGNORECASE,
)


def _number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    return ROMAN_NUMERALS.get(value.upper())


def extract_folder_metadata(source_path: str) -> dict[str, object]:
    path = PurePosixPath(source_path.replace("\\", "/"))
    parts = path.parts
    session_index: int | None = None
    session: str | None = None
    for index, part in enumerate(parts[:-1]):
        if SESSION_RE.fullmatch(part.strip()):
            session_index = index
            session = re.sub(r"[–_ ]", "-", part.strip())
            break

    branch = (
        parts[session_index + 1]
        if session_index is not None and session_index + 1 < len(parts) - 1
        else None
    )
    year = next(
        (
            int(match.group(1))
            if match.group(1)
            else YEAR_NAMES[match.group(2).lower()]
            for part in parts[:-1]
            if (match := YEAR_RE.fullmatch(part.strip()))
        ),
        None,
    )
    variant_match = VARIANT_RE.search(" / ".join(parts))
    normalized_path = "/".join(parts)
    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, normalized_path.casefold()))
    return {
        "source": path.name,
        "source_path": normalized_path,
        "document_id": document_id,
        "academic_session": session,
        "branch": branch,
        "year": year,
        "program_variant": variant_match.group(0) if variant_match else None,
    }


def extract_chunk_metadata(text: str) -> dict[str, object]:
    course_codes = list(dict.fromkeys(code.upper() for code in COURSE_CODE_RE.findall(text)))
    subjects = list(dict.fromkeys(name.strip(" \t:;") for name in SUBJECT_RE.findall(text)))
    semesters = sorted(
        {number for value in SEMESTER_RE.findall(text) if (number := _number(value))}
    )
    units = sorted(
        {number for value in UNIT_RE.findall(text) if (number := _number(value))}
    )
    return {
        "subject": subjects[0] if len(subjects) == 1 else None,
        "course_code": course_codes[0] if len(course_codes) == 1 else None,
        "semester": semesters[0] if len(semesters) == 1 else None,
        "unit": units[0] if len(units) == 1 else None,
    }
