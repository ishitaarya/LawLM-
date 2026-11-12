"""Audit legal chunk provenance and carried human-review history.

Read-only: this module never writes datasets, filters chunks, or starts training.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.document.legal_review import load_jsonl, section_text_hash
from src.training.legal_training_gate import evaluate_training_candidates

REVIEW_FIELDS = (
    "review_record_id",
    "review_status",
    "reviewed_by",
    "reviewed_at",
    "reviewed_text_sha256",
    "legal_content_review_status",
    "legal_content_reviewed_by",
    "legal_content_reviewed_at",
    "legal_content_reviewed_text_sha256",
    "legal_authority_reference",
    "legal_review_resolution_notes",
    "resolved_review_reasons",
)


def audit_chunk_provenance(
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check candidate chunks against source sections and the human-review manifest."""
    gate = evaluate_training_candidates(sections, manifest)
    issues: list[dict[str, Any]] = []
    if gate["issues"] and gate["candidate_count"] == 0:
        issues.append({
            "scope": "review_gate",
            "issue": "no_valid_training_candidates",
            "details": gate["issues"],
        })

    by_number: dict[str, list[int]] = defaultdict(list)
    for index, section in enumerate(sections):
        number = str(section.get("section_number", "")).strip()
        if number:
            by_number[number].append(index)
    candidate_indices = set(gate["candidate_section_indices"])
    seen_candidate_sections: set[int] = set()
    candidate_chunk_ids: list[str] = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"chunk-index-{chunk_index}")
        number = str(chunk.get("section_number", "")).strip()
        matches = by_number.get(number, [])
        if len(matches) != 1:
            issues.append({"chunk_id": chunk_id, "issue": "section_number_missing_or_ambiguous"})
            continue
        section_index = matches[0]
        section = sections[section_index]
        if section_index not in candidate_indices:
            issues.append({"chunk_id": chunk_id, "issue": "parent_section_not_training_candidate"})
            continue

        seen_candidate_sections.add(section_index)
        candidate_chunk_ids.append(str(chunk_id))
        current_hash = section_text_hash(str(section.get("text", "")))
        if chunk.get("source_section_text_sha256") != current_hash:
            issues.append({"chunk_id": chunk_id, "issue": "source_section_hash_missing_or_mismatched"})

        section_spans = {
            (int(span.get("page_number", -1)), int(span.get("line_number", -1))): str(span.get("text", "")).strip()
            for span in section.get("source_spans", [])
            if isinstance(span, dict)
        }
        chunk_spans = chunk.get("source_spans")
        if not isinstance(chunk_spans, list) or not chunk_spans:
            issues.append({"chunk_id": chunk_id, "issue": "chunk_source_spans_missing"})
            continue

        traced_pages = set()
        reconstructed_parts = []
        for span in chunk_spans:
            if not isinstance(span, dict):
                issues.append({"chunk_id": chunk_id, "issue": "invalid_chunk_source_span"})
                continue
            try:
                key = (int(span.get("page_number")), int(span.get("line_number")))
            except (TypeError, ValueError):
                issues.append({"chunk_id": chunk_id, "issue": "invalid_chunk_source_coordinates"})
                continue
            original = section_spans.get(key)
            excerpt = span.get("text")
            if original is None or not isinstance(excerpt, str) or not excerpt or excerpt not in original:
                issues.append({"chunk_id": chunk_id, "issue": "source_line_excerpt_not_found"})
            if not isinstance(span.get("raw_line"), str) or not span.get("raw_line"):
                issues.append({"chunk_id": chunk_id, "issue": "raw_source_line_missing"})
            traced_pages.add(key[0])
            reconstructed_parts.append(excerpt if isinstance(excerpt, str) else "")

        expected_pages = sorted(traced_pages)
        if chunk.get("source_pages") != expected_pages:
            issues.append({"chunk_id": chunk_id, "issue": "source_page_list_mismatch"})
        if expected_pages and (
            chunk.get("page_number") != expected_pages[0]
            or chunk.get("end_page_number") != expected_pages[-1]
        ):
            issues.append({"chunk_id": chunk_id, "issue": "source_page_range_mismatch"})
        if "\n".join(reconstructed_parts).strip() != str(chunk.get("text", "")).strip():
            issues.append({"chunk_id": chunk_id, "issue": "chunk_text_does_not_match_traced_source"})

        expected_annotations = {
            "amendment_annotations": section.get("amendments", []),
            "footnote_annotations": section.get("footnotes", []),
            "ocr_anomalies": section.get("ocr_anomalies", []),
        }
        for field, expected in expected_annotations.items():
            if chunk.get(field) != expected:
                issues.append({"chunk_id": chunk_id, "issue": f"{field}_mismatch"})
        for field in REVIEW_FIELDS:
            expected = None
            if field == "resolved_review_reasons":
                expected = []
            if field in {"review_record_id", "review_status", "reviewed_by", "reviewed_at",
                         "reviewed_text_sha256", "legal_content_review_status",
                         "legal_content_reviewed_by", "legal_content_reviewed_at",
                         "legal_content_reviewed_text_sha256", "legal_authority_reference",
                         "legal_review_resolution_notes", "resolved_review_reasons"}:
                record = next(
                    (item for item in manifest if item.get("section_index") == section_index),
                    None,
                )
                if record is not None:
                    expected = record.get(field, "pending" if field in {
                        "review_status", "legal_content_review_status"
                    } else ([] if field == "resolved_review_reasons" else None))
            if chunk.get(field) != expected:
                issues.append({"chunk_id": chunk_id, "issue": f"{field}_review_history_mismatch"})

        if chunk.get("section_title") != section.get("section_title", ""):
            issues.append({"chunk_id": chunk_id, "issue": "section_title_mismatch"})
        if chunk.get("section_status") != section.get("section_status", "unknown"):
            issues.append({"chunk_id": chunk_id, "issue": "section_status_mismatch"})

    missing = sorted(candidate_indices - seen_candidate_sections)
    for index in missing:
        issues.append({"section_index": index, "issue": "candidate_section_has_no_chunk"})

    provenance_issue_count = len(issues)
    return {
        "report_type": "legal_chunk_provenance_audit",
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "training_candidate_section_count": gate["candidate_count"],
        "candidate_chunk_count": len(candidate_chunk_ids),
        "candidate_chunk_ids": candidate_chunk_ids,
        "candidate_sections_without_chunks": missing,
        "issue_count": provenance_issue_count,
        "issues": issues,
        "provenance_integrity_passed": provenance_issue_count == 0 and gate["candidate_count"] > 0,
        "dataset_exported": False,
        "training_started": False,
        "legal_accuracy_verified": False,
        "ready_for_training": False,
        "note": "This is a read-only provenance audit, not legal verification or training authorization.",
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_chunk_provenance(
        load_jsonl(args.sections),
        load_jsonl(args.chunks),
        load_jsonl(args.review_manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "chunk_count": report["chunk_count"],
        "candidate_chunk_count": report["candidate_chunk_count"],
        "issue_count": report["issue_count"],
        "provenance_integrity_passed": report["provenance_integrity_passed"],
        "report": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
