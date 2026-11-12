"""Conservative gate for identifying reviewed legal-section training candidates.

This module does not export datasets, change chunks, or launch training.
"""
from __future__ import annotations

from typing import Any

from src.document.legal_review import section_text_hash, validate_review_manifest

TRAINABLE_STATUSES = {"in_force", "active"}


def evaluate_training_candidates(
    sections: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """List sections that passed both human review stages and hard safety checks."""
    manifest_report = validate_review_manifest(sections, manifest)
    issues: list[dict[str, Any]] = []
    candidates: list[int] = []
    by_index = {
        int(record["section_index"]): record
        for record in manifest
        if isinstance(record.get("section_index"), int)
        and 0 <= record["section_index"] < len(sections)
    }

    if manifest_report["issues"]:
        issues.append({
            "scope": "manifest",
            "issue": "manifest_integrity_failed",
            "details": manifest_report["issues"],
        })

    for index, section in enumerate(sections):
        record = by_index.get(index)
        if record is None:
            issues.append({"section_index": index, "issue": "review_record_missing"})
            continue

        current_hash = section_text_hash(str(section.get("text", "")))
        if record.get("source_text_sha256") != current_hash:
            issues.append({"section_index": index, "issue": "source_text_hash_mismatch"})
            continue

        if record.get("review_status") != "extraction_review_approved":
            issues.append({"section_index": index, "issue": "extraction_review_not_approved"})
            continue
        if (
            not record.get("reviewed_by")
            or not record.get("reviewed_at")
            or record.get("reviewed_text_sha256") != current_hash
        ):
            issues.append({"section_index": index, "issue": "extraction_approval_invalid"})
            continue

        if record.get("legal_content_review_status") != "legal_content_review_approved":
            issues.append({"section_index": index, "issue": "legal_content_review_not_approved"})
            continue
        if (
            not record.get("legal_content_reviewed_by")
            or not record.get("legal_content_reviewed_at")
            or record.get("legal_content_reviewed_text_sha256") != current_hash
            or not record.get("legal_authority_reference")
            or not record.get("legal_review_resolution_notes")
        ):
            issues.append({"section_index": index, "issue": "legal_content_approval_invalid"})
            continue

        status = str(section.get("section_status", "unknown")).casefold()
        if status not in TRAINABLE_STATUSES:
            issues.append({"section_index": index, "issue": "legal_status_not_verified_in_force"})
            continue

        reasons = set(record.get("review_reasons", []))
        resolved = set(record.get("resolved_review_reasons", []))
        unresolved = sorted(reasons - resolved)
        if unresolved:
            issues.append({
                "section_index": index,
                "issue": "review_reasons_unresolved",
                "reasons": unresolved,
            })
            continue
        if "missing_section_number" in reasons or "duplicate_section_number" in reasons:
            issues.append({"section_index": index, "issue": "section_identity_integrity_failed"})
            continue

        candidates.append(index)

    return {
        "report_type": "legal_training_candidate_gate",
        "section_count": len(sections),
        "candidate_section_indices": candidates,
        "candidate_count": len(candidates),
        "blocked_count": len(sections) - len(candidates),
        "issues": issues,
        "all_sections_passed": len(candidates) == len(sections) and not issues,
        "dataset_exported": False,
        "training_started": False,
        "legal_accuracy_verified": False,
        "ready_for_training": False,
        "note": (
            "Candidates have matching extraction and legal-content reviewer attestations "
            "and a cited authority reference. This gate does not independently verify "
            "the authority, create a training dataset, or authorize model training."
        ),
    }
