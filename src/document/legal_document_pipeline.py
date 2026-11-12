"""Safe document-artifact pipeline for legal PDF page JSONL.

Flow: page JSONL -> section assembly -> structural validation -> legal chunks.
All outputs are new files; existing files are never overwritten. The output
chunks retain page/line provenance and are not automatically training-ready.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.document.legal_chunker import create_legal_chunks
from src.document.legal_parser import find_body_start_page, load_pages
from src.document.legal_section_assembler import assemble_sections
from src.document.legal_validation import validate_sections


def prepare_document_artifacts(
    pages: list[dict[str, Any]],
    *,
    repealed_sections: tuple[str, ...] = (),
    body_start_page: int | None = None,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Build artifacts in memory and stop if structural validation fails."""
    body_start = (
        body_start_page
        if body_start_page is not None
        else find_body_start_page(pages)
    )
    if body_start is not None and (
        body_start < 1
        or not any(int(page["page_number"]) == body_start for page in pages)
    ):
        raise ValueError("body_start_page must match an existing positive source page.")
    if body_start is None:
        raise ValueError(
            "Could not identify the Act body start. Supply a manually verified "
            "body_start_page rather than parsing the table of contents."
        )
    sections = assemble_sections(
        pages,
        body_start_page=body_start,
        repealed_sections=repealed_sections,
    )
    if not sections:
        raise ValueError(
            "No legal section candidates were assembled; refusing to create empty outputs."
        )
    report = validate_sections(sections)

    failed_checks = [
        name for name, passed in report["checks"].items() if not passed
    ]
    if failed_checks:
        raise ValueError(
            "Document failed structural validation: " + ", ".join(failed_checks)
        )

    chunks = create_legal_chunks(
        sections, chunk_size=chunk_size, overlap=overlap
    )
    return sections, report, chunks


def _write_jsonl_new(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json_new(value: dict[str, Any], path: Path) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/documents/Indian_Contract_Act_1872.jsonl"),
        help="Page-level JSONL created by pdf_loader.py.",
    )
    parser.add_argument(
        "--sections-output",
        type=Path,
        default=Path("data/processed/Indian_Contract_Act_1872_sections.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/Indian_Contract_Act_1872_validation.json"),
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=Path("data/processed/Indian_Contract_Act_1872_legal_chunks.jsonl"),
    )
    parser.add_argument(
        "--body-start-page",
        type=int,
        default=None,
        help="Manually verified first page of the Act body, if automatic detection fails.",
    )
    parser.add_argument(
        "--repealed-section",
        action="append",
        default=[],
        help="Verified repealed section number; repeat this option for multiple sections.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    pages = load_pages(args.source)
    sections, report, chunks = prepare_document_artifacts(
        pages,
        body_start_page=args.body_start_page,
        repealed_sections=tuple(args.repealed_section),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    outputs = [args.sections_output, args.report_output, args.chunks_output]
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Sections, report, and chunks outputs must be different paths.")
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(existing)
        )

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    _write_jsonl_new(sections, args.sections_output)
    _write_json_new(report, args.report_output)
    _write_jsonl_new(chunks, args.chunks_output)

    print(f"Source pages: {len(pages)}")
    print(f"Assembled sections: {len(sections)}")
    print(f"Legal chunks: {len(chunks)}")
    print(f"Structural report: {args.report_output}")
    print("Training readiness: NOT VERIFIED")
    print("Legal accuracy: NOT VERIFIED")


if __name__ == "__main__":
    main()
