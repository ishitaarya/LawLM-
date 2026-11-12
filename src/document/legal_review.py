"""Human-review manifest generation for assembled legal sections.

Approval here means a human checked extraction/structure against the source.
It is not legal advice, legal correctness verification, or training approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SHORT_TEXT_WARNING_CHARS = 80


def section_text_hash(text: str) -> str:
    """Return a stable SHA-256 digest of the exact UTF-8 section text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _review_reasons(section: dict[str, Any], duplicate_numbers: set[str]) -> list[str]:
    reasons: list[str] = []
    number = str(section.get("section_number", "")).strip()
    text = str(section.get("text", ""))
    if not number:
        reasons.append("missing_section_number")
    if number.upper() in duplicate_numbers:
        reasons.append("duplicate_section_number")
    if len(text.strip()) < SHORT_TEXT_WARNING_CHARS:
        reasons.append("unusually_short_section_text")
    if section.get("ocr_anomalies"):
        reasons.append("ocr_anomalies_present")
    if section.get("amendments"):
        reasons.append("amendment_annotations_present")
    if section.get("footnotes"):
        reasons.append("footnote_annotations_present")
    heading = section.get("heading_source")
    if (
        not isinstance(heading, dict)
        or not heading.get("page_number")
        or not heading.get("line_number")
        or not heading.get("raw_line")
    ):
        reasons.append("missing_heading_provenance")
    spans = section.get("source_spans")
    if not isinstance(spans, list) or not spans:
        reasons.append("missing_body_source_spans")
    if str(section.get("section_status", "unknown")).lower() == "unknown":
        reasons.append("section_status_unverified")
    if section.get("review_required", True):
        reasons.append("upstream_review_flagged")
    return reasons


def build_review_manifest(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one pending review record per section without changing source data."""
    counts: dict[str, int] = {}
    for section in sections:
        number = str(section.get("section_number", "")).strip().upper()
        counts[number] = counts.get(number, 0) + 1
    duplicates = {number for number, count in counts.items() if number and count > 1}

    manifest = []
    occurrences: dict[str, int] = {}
    for index, section in enumerate(sections):
        number = str(section.get("section_number", "")).strip()
        key = number.upper()
        occurrences[key] = occurrences.get(key, 0) + 1
        text = str(section.get("text", ""))
        manifest.append({
            "review_record_id": f"section-{index + 1:06d}",
            "section_index": index,
            "section_number": number or None,
            "section_occurrence": occurrences[key],
            "section_title": str(section.get("section_title", "")),
            "chapter": section.get("chapter"),
            "source_page_start": section.get("page_number"),
            "source_page_end": section.get("end_page_number"),
            "source_pages": list(section.get("source_pages", [])),
            "heading_source": section.get("heading_source"),
            "source_spans": list(section.get("source_spans", [])),
            "amendments": list(section.get("amendments", [])),
            "footnotes": list(section.get("footnotes", [])),
            "ocr_anomalies": list(section.get("ocr_anomalies", [])),
            "source_text_sha256": section_text_hash(text),
            "source_text_character_count": len(text),
            "review_reasons": _review_reasons(section, duplicates),
            "review_status": "pending",
            "review_scope": "source_extraction_and_structure_only",
            "reviewed_by": None,
            "reviewed_at": None,
            "reviewed_text_sha256": None,
            "reviewer_notes": None,
            "legal_accuracy_verified": False,
            "ready_for_training": False,
        })
    return manifest


def approve_review_record(
    record: dict[str, Any],
    *,
    reviewer: str,
    reviewed_at: str,
    reviewed_text_sha256: str,
    reviewer_notes: str | None = None,
) -> dict[str, Any]:
    """Return an approved copy only when reviewer and exact source hash are supplied."""
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("A non-empty reviewer identity is required.")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ValueError("A review timestamp is required.")
    expected_hash = record.get("source_text_sha256")
    if not expected_hash or reviewed_text_sha256 != expected_hash:
        raise ValueError("Reviewed text hash does not match the current source text.")
    approved = dict(record)
    approved.update({
        "review_status": "extraction_review_approved",
        "reviewed_by": reviewer.strip(),
        "reviewed_at": reviewed_at.strip(),
        "reviewed_text_sha256": reviewed_text_sha256,
        "reviewer_notes": reviewer_notes,
        "legal_accuracy_verified": False,
        "ready_for_training": False,
    })
    return approved



def approve_legal_content_record(
    record: dict[str, Any],
    *,
    reviewer: str,
    reviewed_at: str,
    reviewed_text_sha256: str,
    authority_reference: str,
    resolution_notes: str,
    resolved_review_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Record a separate human legal-content review against an identified authority.

    This is a reviewer attestation, not an independent guarantee of correctness.
    """
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("A non-empty legal reviewer identity is required.")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ValueError("A legal review timestamp is required.")
    if not isinstance(authority_reference, str) or not authority_reference.strip():
        raise ValueError("An authoritative source reference is required.")
    if not isinstance(resolution_notes, str) or not resolution_notes.strip():
        raise ValueError("Legal review notes are required.")
    expected_hash = record.get("source_text_sha256")
    if not expected_hash or reviewed_text_sha256 != expected_hash:
        raise ValueError("Reviewed text hash does not match the current source text.")
    approved = dict(record)
    approved.update({
        "legal_content_review_status": "legal_content_review_approved",
        "legal_content_reviewed_by": reviewer.strip(),
        "legal_content_reviewed_at": reviewed_at.strip(),
        "legal_content_reviewed_text_sha256": reviewed_text_sha256,
        "legal_authority_reference": authority_reference.strip(),
        "legal_review_resolution_notes": resolution_notes.strip(),
        "resolved_review_reasons": sorted(set(resolved_review_reasons or [])),
        "legal_accuracy_verified": False,
        "ready_for_training": False,
    })
    return approved


def validate_review_manifest(
    sections: list[dict[str, Any]], manifest: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate coverage and approval integrity against the current source sections."""
    issues: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    approved_count = 0
    pending_count = 0
    for record in manifest:
        try:
            index = int(record.get("section_index", -1))
        except (TypeError, ValueError):
            index = -1
        if index < 0 or index >= len(sections):
            issues.append({"review_record_id": record.get("review_record_id"), "issue": "invalid_section_index"})
            continue
        if index in seen_indices:
            issues.append({"review_record_id": record.get("review_record_id"), "issue": "duplicate_section_index"})
            continue
        seen_indices.add(index)
        current_hash = section_text_hash(str(sections[index].get("text", "")))
        if record.get("source_text_sha256") != current_hash:
            issues.append({"review_record_id": record.get("review_record_id"), "issue": "source_text_hash_mismatch"})
            continue
        status = record.get("review_status", "pending")
        if status == "extraction_review_approved":
            if (
                not record.get("reviewed_by")
                or not record.get("reviewed_at")
                or record.get("reviewed_text_sha256") != current_hash
            ):
                issues.append({"review_record_id": record.get("review_record_id"), "issue": "invalid_approval_metadata"})
                pending_count += 1
            else:
                approved_count += 1
        elif status == "pending":
            pending_count += 1
        else:
            issues.append({"review_record_id": record.get("review_record_id"), "issue": "unknown_review_status"})
            pending_count += 1

    missing_indices = sorted(set(range(len(sections))) - seen_indices)
    for index in missing_indices:
        issues.append({"section_index": index, "issue": "missing_review_record"})
    return {
        "report_type": "human_extraction_review_manifest_validation",
        "section_count": len(sections),
        "manifest_record_count": len(manifest),
        "approved_count": approved_count,
        "pending_count": pending_count + len(missing_indices),
        "missing_section_indices": missing_indices,
        "issues": issues,
        "all_sections_have_valid_review_records": not issues and len(seen_indices) == len(sections),
        "legal_accuracy_verified": False,
        "ready_for_training": False,
        "note": "Extraction-review approval is not legal accuracy verification or authorization to train.",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}.")
            records.append(record)
    return records


def write_jsonl_new(records: list[dict[str, Any]], path: Path) -> None:
    """Create a new manifest only; never overwrite an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Assembled section JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="New review-manifest JSONL path.")
    args = parser.parse_args()
    sections = load_jsonl(args.source)
    manifest = build_review_manifest(sections)
    write_jsonl_new(manifest, args.output)
    print(f"Sections: {len(sections)}")
    print(f"Pending review records: {len(manifest)}")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
