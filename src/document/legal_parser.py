"""Phase 7.5C.2: conservative structural analysis of extracted legal PDF pages.

This module identifies page types and structural candidates. It deliberately
does not assemble legal provisions: candidate detection must be reviewed and
validated before downstream data generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path("data/documents/Indian_Contract_Act_1872.jsonl")

# Examples: "3.Communication ... . —The" and "1. Short title.—This Act..."
_SECTION_LINE = re.compile(
    r"^\s*(?P<number>\d{1,3}[A-Za-z]?)\s*\.\s*"
    r"(?P<heading>.+?)\s*(?:[.。]\s*[—–-]|[—–])\s*(?P<body>.*)$"
)
_CHAPTER_LINE = re.compile(
    r"^\s*CHAPTER\s+(?P<number>[IVXLCDM]+)\s*$",
    re.IGNORECASE,
)
_AMENDMENT_PATTERNS = (
    re.compile(r"^\s*STATE\s+AMENDMENT\s*$", re.IGNORECASE),
    re.compile(r"^\s*CENTRAL\s+AMENDMENT\s*$", re.IGNORECASE),
    re.compile(r"^\s*AMENDMENT\s+OF\s+SECTION\b.*$", re.IGNORECASE),
    re.compile(r"^\s*\[\s*Vide\b.*$", re.IGNORECASE),
)
_FOOTNOTE_START = re.compile(
    r"^\s*\d+\.\s*(?:For\b|See\b|Subs\.\s+by\b|The words\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SectionCandidate:
    section_number: str
    heading_text: str
    page_number: int
    line_number: int
    raw_line: str
    confidence: str = "review_required"


@dataclass(frozen=True)
class PageAnalysis:
    page_number: int
    page_type: str
    chapter_headings: list[str]
    section_candidates: list[dict[str, Any]]
    amendment_markers: list[str]
    text: str


def normalize_line(line: str) -> str:
    """Collapse horizontal whitespace while retaining the original elsewhere."""
    return re.sub(r"[ \t]+", " ", line).strip()


def load_pages(path: Path = DEFAULT_SOURCE) -> list[dict[str, Any]]:
    """Load page-level JSONL and validate the fields needed by this parser."""
    if not path.is_file():
        raise FileNotFoundError(f"Source JSONL not found: {path}")

    pages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on source line {line_number}: {exc}"
                ) from exc

            missing = {"source_file", "page_number", "text"} - record.keys()
            if missing:
                raise ValueError(
                    f"Missing {sorted(missing)} on source line {line_number}"
                )
            if not isinstance(record["text"], str):
                raise ValueError(
                    f"'text' must be a string on source line {line_number}"
                )
            try:
                page_number = int(record["page_number"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid page_number on source line {line_number}"
                ) from exc
            if page_number < 1:
                raise ValueError(
                    f"page_number must be positive on source line {line_number}"
                )
            pages.append({**record, "page_number": page_number})

    pages.sort(key=lambda item: item["page_number"])
    return pages


def find_body_start_page(pages: Iterable[dict[str, Any]]) -> int | None:
    """Find the first page containing the Act's enactment/preamble transition."""
    for page in pages:
        text = page["text"]
        if re.search(
            r"IT\s+IS\s+HEREBY\s+ENACTED\s+AS\s+FOLLOWS",
            text,
            re.IGNORECASE,
        ) or re.search(r"PREAMBLE\s*[—–-]\s*WHEREAS", text, re.IGNORECASE):
            return int(page["page_number"])
    return None


def classify_page(page_number: int, text: str, body_start_page: int | None) -> str:
    """Classify a page using the detected body transition, not a fixed page count."""
    if body_start_page is None:
        return "unclassified"
    if page_number < body_start_page:
        return "table_of_contents_or_front_matter"
    if page_number == body_start_page:
        return "act_body_start"
    return "act_body_or_continuation"


def detect_chapter_headings(text: str) -> list[str]:
    """Return chapter headings; chapter headings are not legal sections."""
    found: list[str] = []
    for line in text.splitlines():
        match = _CHAPTER_LINE.match(normalize_line(line))
        if match:
            found.append(f"CHAPTER {match.group('number').upper()}")
    return found


def detect_section_candidates(
    text: str, page_number: int = 0
) -> list[dict[str, Any]]:
    """Find conservative section-heading candidates, not definitive provisions.

    Footnote-style lines and ordinary numbered subclauses are intentionally
    excluded. Candidates retain their original line for manual validation.
    """
    candidates: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = normalize_line(raw_line)
        if not line or _FOOTNOTE_START.match(line):
            continue

        match = _SECTION_LINE.match(line)
        if not match:
            continue

        number = match.group("number")
        heading = normalize_line(match.group("heading"))
        if not heading:
            continue

        candidate = SectionCandidate(
            section_number=number,
            heading_text=heading,
            page_number=page_number,
            line_number=line_number,
            raw_line=raw_line,
        )
        candidates.append(asdict(candidate))
    return candidates


def detect_amendment_markers(text: str) -> list[str]:
    """Collect amendment labels and citation lines without merging their text."""
    markers: list[str] = []
    for line in text.splitlines():
        line = normalize_line(line)
        if line and any(pattern.match(line) for pattern in _AMENDMENT_PATTERNS):
            markers.append(line)
    return markers


def analyze_page(
    record: dict[str, Any], body_start_page: int | None = None
) -> PageAnalysis:
    """Analyze one page; supply body_start_page for consistent classification."""
    page_number = int(record["page_number"])
    text = record["text"]
    return PageAnalysis(
        page_number=page_number,
        page_type=classify_page(page_number, text, body_start_page),
        chapter_headings=detect_chapter_headings(text),
        section_candidates=detect_section_candidates(text, page_number),
        amendment_markers=detect_amendment_markers(text),
        text=text,
    )


def analyze_document(path: Path = DEFAULT_SOURCE) -> list[PageAnalysis]:
    """Analyze all pages without writing or modifying any files."""
    pages = load_pages(path)
    body_start_page = find_body_start_page(pages)
    return [analyze_page(page, body_start_page) for page in pages]
