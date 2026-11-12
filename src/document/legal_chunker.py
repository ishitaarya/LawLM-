"""Section-aware chunking with page/line citations for assembled legal text.

This module consumes assembled section JSONL, never raw PDF pages. It keeps
chunks inside a single provision and carries the exact source lines forward.
It does not modify the input or any existing chunk file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.document.legal_review import section_text_hash, validate_review_manifest

DEFAULT_INPUT = Path("data/processed/Indian_Contract_Act_1872_sections.jsonl")
DEFAULT_OUTPUT = Path("data/processed/Indian_Contract_Act_1872_legal_chunks.jsonl")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def _section_text_and_offsets(
    section: dict[str, Any],
) -> tuple[str, list[tuple[int, int, dict[str, Any]]]]:
    """Rebuild text exactly as the assembler does and index each source line."""
    spans = section.get("source_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError(
            f"Section {section.get('section_number', '<missing>')} has no source_spans."
        )

    parts: list[str] = []
    offsets: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for span in spans:
        value = str(span.get("text", "")).strip()
        if not value:
            continue
        if parts:
            cursor += 1  # newline inserted between source lines
        start = cursor
        parts.append(value)
        cursor += len(value)
        offsets.append((start, cursor, span))

    return "\n".join(parts), offsets


def _chunk_ranges(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    if not text:
        return []

    ranges = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        ranges.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return ranges


def _trace_range(
    start: int,
    end: int,
    offsets: list[tuple[int, int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return source line references intersecting a chunk character range."""
    traced = []
    for line_start, line_end, source in offsets:
        overlap_start = max(start, line_start)
        overlap_end = min(end, line_end)
        if overlap_start >= overlap_end:
            continue
        original = str(source["text"]).strip()
        traced.append({
            "page_number": int(source["page_number"]),
            "line_number": int(source["line_number"]),
            "text": original[overlap_start - line_start:overlap_end - line_start],
            "raw_line": source.get("raw_line", original),
        })
    return traced


def create_legal_chunks(
    sections: list[dict[str, Any]],
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    review_manifest: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create provision-bounded chunks with source references and optional review history."""
    review_by_index: dict[int, dict[str, Any]] = {}
    if review_manifest is not None:
        report = validate_review_manifest(sections, review_manifest)
        if not report["all_sections_have_valid_review_records"]:
            raise ValueError("Review manifest failed integrity validation; refusing to attach review history.")
        review_by_index = {int(record["section_index"]): record for record in review_manifest}
    chunks: list[dict[str, Any]] = []
    counter = 1

    for section_index, section in enumerate(sections):
        number = str(section.get("section_number", "")).strip()
        if not number:
            raise ValueError("Cannot chunk a section without section_number.")
        text, offsets = _section_text_and_offsets(section)
        if not text.strip():
            raise ValueError(f"Section {number} has no extractable body text.")

        for start, end in _chunk_ranges(text, chunk_size, overlap):
            chunk_text = text[start:end].strip()
            source_spans = _trace_range(start, end, offsets)
            if not source_spans:
                raise ValueError(f"Chunk for section {number} has no source trace.")

            pages = sorted({span["page_number"] for span in source_spans})
            review = review_by_index.get(section_index)
            chunks.append({
                "chunk_id": f"legal_{counter:06d}",
                "section_number": number,
                "section_title": section.get("section_title", ""),
                "chapter": section.get("chapter"),
                "page_number": pages[0],
                "end_page_number": pages[-1],
                "source_pages": pages,
                "source_spans": source_spans,
                "section_status": section.get("section_status", "unknown"),
                "review_required": bool(section.get("review_required", True)),
                "text": chunk_text,
                "character_count": len(chunk_text),
                "amendment_annotations": list(section.get("amendments", [])),
                "footnote_annotations": list(section.get("footnotes", [])),
                "ocr_anomalies": list(section.get("ocr_anomalies", [])),
                "has_amendment_annotations": bool(section.get("amendments")),
                "has_footnote_annotations": bool(section.get("footnotes")),
                "has_ocr_anomalies": bool(section.get("ocr_anomalies")),
                "source_section_text_sha256": section_text_hash(str(section.get("text", ""))),
                "review_record_id": review.get("review_record_id") if review else None,
                "review_status": review.get("review_status", "pending") if review else "pending",
                "reviewed_by": review.get("reviewed_by") if review else None,
                "reviewed_at": review.get("reviewed_at") if review else None,
                "reviewed_text_sha256": review.get("reviewed_text_sha256") if review else None,
                "legal_content_review_status": review.get("legal_content_review_status", "pending") if review else "pending",
                "legal_content_reviewed_by": review.get("legal_content_reviewed_by") if review else None,
                "legal_content_reviewed_at": review.get("legal_content_reviewed_at") if review else None,
                "legal_content_reviewed_text_sha256": review.get("legal_content_reviewed_text_sha256") if review else None,
                "legal_authority_reference": review.get("legal_authority_reference") if review else None,
                "legal_review_resolution_notes": review.get("legal_review_resolution_notes") if review else None,
                "resolved_review_reasons": list(review.get("resolved_review_reasons", [])) if review else [],
            })
            counter += 1

    return chunks


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def write_jsonl_new(records: list[dict[str, Any]], path: Path) -> None:
    """Create a new output only; never overwrite an existing dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    sections = load_jsonl(args.input)
    chunks = create_legal_chunks(
        sections, chunk_size=args.chunk_size, overlap=args.overlap
    )
    write_jsonl_new(chunks, args.output)
    print(f"Sections: {len(sections)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
