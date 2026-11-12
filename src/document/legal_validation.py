"""Read-only structural validation for assembled legal provisions.

This checks extraction quality signals, not the legal correctness or current
force of any provision. It does not write or modify source files.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.document.legal_section_assembler import assemble_document


def validate_sections(sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a deterministic validation summary for assembled section records."""
    numbers = [str(item.get("section_number", "")).upper() for item in sections]
    counts = Counter(numbers)
    duplicates = sorted(number for number, count in counts.items() if number and count > 1)
    empty_numbers = sum(not number for number in numbers)
    empty_text = [
        number or "<missing>"
        for number, item in zip(numbers, sections)
        if not str(item.get("text", "")).strip()
    ]
    invalid_ranges = [
        {
            "section_number": item.get("section_number"),
            "start_page": item.get("page_number"),
            "end_page": item.get("end_page_number"),
        }
        for item in sections
        if int(item.get("end_page_number", 0)) < int(item.get("page_number", 0))
    ]
    missing_page_provenance = [
        item.get("section_number")
        for item in sections
        if not item.get("source_pages")
    ]
    page_order_errors = [
        item.get("section_number")
        for item in sections
        if item.get("source_pages")
        and item["source_pages"] != sorted(set(item["source_pages"]))
    ]
    amendment_count = sum(len(item.get("amendments", [])) for item in sections)
    footnote_count = sum(len(item.get("footnotes", [])) for item in sections)
    ocr_anomaly_count = sum(len(item.get("ocr_anomalies", [])) for item in sections)
    review_required_count = sum(bool(item.get("review_required", True)) for item in sections)

    return {
        "report_type": "structural_extraction_validation",
        "legal_accuracy_verified": False,
        "section_count": len(sections),
        "unique_section_number_count": len(set(number for number in numbers if number)),
        "duplicate_section_numbers": duplicates,
        "missing_section_number_count": empty_numbers,
        "empty_text_section_numbers": empty_text,
        "invalid_page_ranges": invalid_ranges,
        "sections_missing_page_provenance": missing_page_provenance,
        "sections_with_unsorted_or_duplicate_page_refs": page_order_errors,
        "amendment_annotation_count": amendment_count,
        "footnote_annotation_count": footnote_count,
        "ocr_anomaly_count": ocr_anomaly_count,
        "sections_requiring_review": review_required_count,
        "checks": {
            "no_duplicate_section_numbers": not duplicates,
            "all_sections_have_numbers": empty_numbers == 0,
            "all_sections_have_text": not empty_text,
            "all_page_ranges_valid": not invalid_ranges,
            "all_sections_have_page_provenance": not missing_page_provenance,
            "page_references_sorted_and_unique": not page_order_errors,
        },
        "ready_for_training": False,
        "note": (
            "Passing structural checks does not establish legal correctness, "
            "completeness, amendment currency, or suitability for model training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/documents/Indian_Contract_Act_1872.jsonl"),
        help="Page-level source JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional report path. Existing files are never overwritten.",
    )
    args = parser.parse_args()

    sections = assemble_document(args.source)
    report = validate_sections(sections)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output is None:
        print(rendered)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(f"Validation report created: {args.output}")


if __name__ == "__main__":
    main()
