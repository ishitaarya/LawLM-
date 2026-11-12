"""Phase 7.5C.3: assemble provision candidates across extracted PDF pages.

The assembler preserves page provenance and keeps amendment and footnote text
outside the central provision body. Repeal status is assigned only when the
caller supplies verified section numbers. OCR is never silently repaired.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.document.legal_parser import (
    DEFAULT_SOURCE,
    detect_amendment_markers,
    detect_section_candidates,
    find_body_start_page,
    load_pages,
)

_FOOTNOTE_START = re.compile(
    r"^\s*\d+\.\s*(?:For\b|See\b|Subs\.\s+by\b|The words\b)", re.IGNORECASE
)
_HEADING_SEPARATOR = re.compile(r"(?:[.。]\s*[—–-]|[—–])")
_CHAPTER_LINE = re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+)\s*$", re.IGNORECASE)
# Known scan artifacts in this source: a footnote marker can be fused with
# section 151/161 (e.g. 1151./1161.). Never normalize these automatically.
_OCR_SECTION_PREFIX = re.compile(r"^11(?:5|6)1\.\s*")


@dataclass
class AssembledSection:
    section_number: str
    section_title: str
    page_number: int
    end_page_number: int
    source_pages: list[int] = field(default_factory=list)
    heading_source: dict[str, Any] = field(default_factory=dict)
    source_spans: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    chapter: str | None = None
    amendments: list[dict[str, Any]] = field(default_factory=list)
    footnotes: list[dict[str, Any]] = field(default_factory=list)
    ocr_anomalies: list[dict[str, Any]] = field(default_factory=list)
    section_status: str = "unknown"
    review_required: bool = True


def _section_body(raw_line: str) -> str:
    match = _HEADING_SEPARATOR.search(raw_line)
    return raw_line[match.end():].strip() if match else ""


def _append_text(
    section: AssembledSection,
    value: str,
    page_number: int,
    line_number: int,
    *,
    raw_line: str | None = None,
) -> None:
    value = value.strip()
    if not value:
        return
    section.text = f"{section.text}\n{value}".strip() if section.text else value
    if page_number not in section.source_pages:
        section.source_pages.append(page_number)
    section.end_page_number = max(section.end_page_number, page_number)
    section.source_spans.append({
        "page_number": page_number,
        "line_number": line_number,
        "text": value,
        "raw_line": raw_line if raw_line is not None else value,
    })

def assemble_sections(
    pages: Iterable[dict[str, Any]],
    *,
    body_start_page: int | None = None,
    repealed_sections: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Assemble provision candidates while keeping annotations separate."""
    ordered_pages = sorted(pages, key=lambda item: int(item["page_number"]))
    repealed = {str(number).upper() for number in repealed_sections}
    assembled: list[AssembledSection] = []
    current: AssembledSection | None = None
    current_chapter: str | None = None
    amendment_mode = False
    amendment_target: AssembledSection | None = None
    document_annotations: list[dict[str, Any]] = []

    for page in ordered_pages:
        page_number = int(page["page_number"])
        page_text = str(page["text"])
        if body_start_page is not None and page_number < body_start_page:
            continue

        candidates = detect_section_candidates(page_text, page_number)
        candidates_by_line = {item["line_number"]: item for item in candidates}

        for line_number, raw_line in enumerate(page_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            chapter_match = _CHAPTER_LINE.match(line)
            if chapter_match:
                current_chapter = f"CHAPTER {chapter_match.group(1).upper()}"
                continue

            if line_number in candidates_by_line:
                candidate = candidates_by_line[line_number]
                amendment_mode = False
                amendment_target = None
                number = str(candidate["section_number"]).upper()
                current = AssembledSection(
                    section_number=number,
                    section_title=candidate["heading_text"],
                    page_number=page_number,
                    end_page_number=page_number,
                    source_pages=[page_number],
                    heading_source={
                        "page_number": page_number,
                        "line_number": line_number,
                        "raw_line": raw_line,
                    },
                    chapter=current_chapter,
                    section_status="repealed" if number in repealed else "unknown",
                )
                _append_text(
                    current,
                    _section_body(raw_line),
                    page_number,
                    line_number,
                    raw_line=raw_line,
                )
                assembled.append(current)
                continue

            if _OCR_SECTION_PREFIX.match(line):
                anomaly = {
                    "page_number": page_number,
                    "line_number": line_number,
                    "text": line,
                    "reason": "Possible footnote marker fused with section number; manual review required.",
                }
                if current is not None:
                    current.ocr_anomalies.append(anomaly)
                else:
                    document_annotations.append({"type": "ocr_anomaly", **anomaly})
                continue

            if _FOOTNOTE_START.match(line):
                note = {"page_number": page_number, "line_number": line_number, "text": line}
                if current is not None:
                    current.footnotes.append(note)
                else:
                    document_annotations.append({"type": "footnote", **note})
                continue

            if detect_amendment_markers(line):
                amendment_mode = True
                amendment_target = current
                marker = {"page_number": page_number, "line_number": line_number, "text": line}
                if amendment_target is not None:
                    amendment_target.amendments.append(marker)
                else:
                    document_annotations.append({"type": "amendment", **marker})
                continue

            if amendment_mode:
                amendment = {"page_number": page_number, "line_number": line_number, "text": line}
                if amendment_target is not None:
                    amendment_target.amendments.append(amendment)
                else:
                    document_annotations.append({"type": "amendment", **amendment})
                continue

            if current is not None:
                _append_text(
                    current, raw_line, page_number, line_number, raw_line=raw_line
                )

    results = [asdict(section) for section in assembled]
    for item in results:
        item["source_pages"] = sorted(set(item["source_pages"]))
        if document_annotations:
            item["document_annotations"] = document_annotations
    return results


def assemble_document(
    path: Path = DEFAULT_SOURCE,
    *,
    repealed_sections: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Load and assemble a page JSONL document without writing any files."""
    pages = load_pages(path)
    body_start_page = find_body_start_page(pages)
    if body_start_page is None:
        raise ValueError(
            "Could not identify the Act body start; refusing to parse front matter as provisions."
        )
    return assemble_sections(
        pages,
        body_start_page=body_start_page,
        repealed_sections=repealed_sections,
    )
