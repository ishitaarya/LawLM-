"""Read-only audit of page-level legal PDF extraction JSONL.

This audit checks file structure, page coverage, empty pages, and OCR warning
signals. It does not repair text, perform OCR, or verify legal correctness.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("data/documents/Indian_Contract_Act_1872.jsonl")
_REPLACEMENT_CHAR = "\ufffd"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_PAGE_NUMBER_PATTERN = re.compile(r"^\s*(\d{1,4})\s*$")


def audit_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return page coverage and extraction-quality signals without mutation."""
    numbers = [int(page["page_number"]) for page in pages]
    counts = Counter(numbers)
    duplicates = sorted(number for number, count in counts.items() if count > 1)
    if numbers:
        expected = set(range(min(numbers), max(numbers) + 1))
        missing = sorted(expected - set(numbers))
    else:
        missing = []

    empty_pages = []
    suspiciously_short_pages = []
    replacement_char_pages = []
    control_char_pages = []
    missing_source_file_pages = []
    inconsistent_source_files = set()

    source_files = {
        str(page.get("source_file", "")).strip()
        for page in pages
        if str(page.get("source_file", "")).strip()
    }

    for page in pages:
        number = int(page["page_number"])
        text = page.get("text", "")
        if not isinstance(text, str) or not text.strip():
            empty_pages.append(number)
            continue
        if len(text.strip()) < 40:
            suspiciously_short_pages.append(number)
        if _REPLACEMENT_CHAR in text:
            replacement_char_pages.append(number)
        if _CONTROL_CHARS.search(text):
            control_char_pages.append(number)
        if not str(page.get("source_file", "")).strip():
            missing_source_file_pages.append(number)

    if len(source_files) > 1:
        inconsistent_source_files = source_files

    checks = {
        "no_duplicate_page_numbers": not duplicates,
        "no_missing_page_numbers_in_observed_range": not missing,
        "no_empty_pages": not empty_pages,
        "all_pages_have_source_file": not missing_source_file_pages,
        "single_source_file": len(source_files) <= 1,
        "no_replacement_characters": not replacement_char_pages,
        "no_control_characters": not control_char_pages,
    }
    return {
        "report_type": "legal_pdf_page_extraction_audit",
        "page_count": len(pages),
        "first_page_number": min(numbers) if numbers else None,
        "last_page_number": max(numbers) if numbers else None,
        "duplicate_page_numbers": duplicates,
        "missing_page_numbers_in_observed_range": missing,
        "empty_pages": empty_pages,
        "suspiciously_short_pages": suspiciously_short_pages,
        "pages_with_replacement_characters": replacement_char_pages,
        "pages_with_control_characters": control_char_pages,
        "pages_missing_source_file": missing_source_file_pages,
        "inconsistent_source_files": sorted(inconsistent_source_files),
        "checks": checks,
        "structural_page_integrity_passed": bool(pages) and all(checks.values()),
        "legal_accuracy_verified": False,
        "ocr_quality_verified": False,
        "ready_for_training": False,
        "note": (
            "A passing page audit only confirms basic extraction structure. "
            "Short pages require visual inspection; this audit cannot detect all "
            "OCR substitutions, missing columns, or incorrect legal text."
        ),
    }


def load_pages_with_parse_errors(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load valid records while retaining JSONL syntax errors in the report."""
    if not path.is_file():
        raise FileNotFoundError(f"Source JSONL not found: {path}")
    valid_lines: list[str] = []
    parse_errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                valid_lines.append(json.dumps(record, ensure_ascii=False))
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_number, "error": str(exc)})

    # Reuse the canonical page schema checks without creating a temporary file.
    pages: list[dict[str, Any]] = []
    for line_number, raw in enumerate(valid_lines, start=1):
        record = json.loads(raw)
        missing = {"source_file", "page_number", "text"} - record.keys()
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in valid JSON record {line_number}")
        if not isinstance(record["text"], str):
            raise ValueError(f"'text' must be a string in valid JSON record {line_number}")
        try:
            page_number = int(record["page_number"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid page_number in valid JSON record {line_number}") from exc
        if page_number < 1:
            raise ValueError(f"page_number must be positive in valid JSON record {line_number}")
        pages.append({**record, "page_number": page_number})
    pages.sort(key=lambda item: item["page_number"])
    return pages, parse_errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pages, parse_errors = load_pages_with_parse_errors(args.source)
    report = audit_pages(pages)
    report["json_parse_error_count"] = len(parse_errors)
    report["json_parse_errors"] = parse_errors
    if parse_errors:
        report["structural_page_integrity_passed"] = False
    rendered = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output is None:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(f"Page extraction audit created: {args.output}")


if __name__ == "__main__":
    main()
