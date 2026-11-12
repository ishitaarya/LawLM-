"""Phase 7.5C.3: assemble provision candidates across extracted PDF pages.

The assembler is deliberately conservative. It preserves page provenance,
keeps amendment and footnote text outside the central provision body, and
marks repeal status only when the caller supplies verified section numbers.
It does not silently repair OCR or claim that candidates are authoritative law.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.document.legal_parser import (
    DEFAULT_SOURCE,
    detect_amendment_markers,
    detect_chapter_headings,
    detect_section_candidates,
    load_pages,
)


_FOOTNOTE_START = re.compile(
    r"^\\s*\\d+\\.\\s*(?:For\\b|See\\b|Subs\\.\\s+by\\b|The words\\b)",
    re.IGNORECASE,
)
_AMENDMENT_CITATION = re.compile(
    r"^\\s*(?:\\[\\s*Vide\\b|Act\\s+No\\.|Inserted\\s+by\\b|"
    r"Subs\\.\\s+by\\b)",
    re.IGNORECASE,
)
_HEADING_SEPARATOR = re.compile(r"(?:[.。]\\s*[—–-]|[—–])")


@dataclass
class AssembledSection:
    section_number: str
    section_title: str
    page_number: int
    end_page_number: int
    source_pages: list[int] = field(default_factory=list)
    text: str = ""
    chapter: str | None = None
    amendments: list[dict[str, Any]] = field(default_factory=list)
    footnotes: list[dict[str, Any]] = field(default_factory=list)
    section_status: str = "unknown"
    review_required: bool = True


def _clean_line(value: str) -> str:
    return re.sub(r"[ \\t]+", " ", value).strip()


def _section_body(raw_line: str) -> str:
    """Extract text after the heading separator without changing its wording."""
    match = _HEADING_SEPARATOR.search(raw_line)
    return raw_line[match.end():].strip() if match else ""


def _append_text(section: AssembledSection, value: str, page_number: int) -> None:
    value = value.strip()
    if not value:
        return
    section.text = f"{section.text}\\n{value}".strip() if section.text else value
    if page_number not in section.source_pages:
        section.source_pages.append(page_number)
    section.end_page_number = max(section.end_page_number, page_number)


def assemble_sections(
    pages: Iterable[dict[str, Any]],
    *,
    body_start_page: int | None = None,
    repealed_sections: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Assemble provision candidates while preserving separate annotations.

    Input records use the page-level schema: source_file, page_number, text.
    Page order is normalized. TOC/front matter is skipped only when a body
    start page is supplied; if it is unknown, no page is silently discarded.
    """
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

        chapter_headings = detect_chapter_headings(page_text)
        if chapter_headings:
            current_chapter = chapter_headings[-1]

        candidates = detect_section_candidates(page_text, page_number)
        candidates_by_line = {item["line_number"]: item for item in candidates}

        for line_number, raw_line in enumerate(page_text.splitlines(), start=1):
            line = _clean_line(raw_line)
            if not line:
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
                    chapter=current_chapter,
                    section_status="repealed" if number in repealed else "unknown",
                )
                body = _section_body(raw_line)
                if body:
                    _append_text(current, body, page_number)
                assembled.append(current)
                continue

            if _FOOTNOTE_START.match(line):
                note = {
                    "page_number": page_number,
                    "line_number": line_number,
                    "text": line,
                }
                if current is not None:
                    current.footnotes.append(note)
                else:
                    document_annotations.append({"type": "footnote", **note})
                continue

            if detect_amendment_markers(line):
                amendment_mode = True
                amendment_target = current
                marker = {
                    "page_number": page_number,
                    "line_number": line_number,
                    "text": line,
                }
                if amendment_target is not None:
                    amendment_target.amendments.append(marker)
                else:
                    document_annotations.append({"type": "amendment", **marker})
                continue

            if amendment_mode:
                amendment = {
                    "page_number": page_number,
                    "line_number": line_number,
                    "text": line,
                }
                if amendment_target is not None:
                    amendment_target.amendments.append(amendment)
                else:
                    document_annotations.append({"type": "amendment", **amendment})
                # An explicit citation line is retained as amendment metadata.
                if _AMENDMENT_CITATION.match(line):
                    continue
                continue

            if current is not None:
                _append_text(current, raw_line, page_number)

    results = [asdict(section) for section in assembled]
    for item in results:
        item["source_pages"] = sorted(set(item["source_pages"]))
        item["amendments"] = item["amendments"]
        item["footnotes"] = item["footnotes"]
    # Keep document-level annotations discoverable without pretending they
    # belong to the first or last provision.
    if document_annotations:
        for item in results:
            item["document_annotations"] = document_annotations
    return results


def assemble_document(
    path: Path = DEFAULT_SOURCE,
    *,
    repealed_sections: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Load and assemble a page JSONL document; this function writes no files."""
    pages = load_pages(path)
    from src.document.legal_parser import find_body_start_page

    body_start_page = find_body_start_page(pages)
    return assemble_sections(
        pages,
        body_start_page=body_start_page,
        repealed_sections=repealed_sections,
    )
