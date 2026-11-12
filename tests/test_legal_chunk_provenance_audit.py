"""Tests for legal chunk provenance and review-history auditing."""
import unittest

from src.document.legal_chunker import create_legal_chunks
from src.document.legal_review import (
    approve_legal_content_record,
    approve_review_record,
    build_review_manifest,
    section_text_hash,
)
from src.training.legal_chunk_provenance_audit import audit_chunk_provenance


class LegalChunkProvenanceAuditTests(unittest.TestCase):
    def section(self):
        lines = [
            {"page_number": 12, "line_number": 3, "text": "A valid agreement is enforceable by law.", "raw_line": "3 A valid agreement is enforceable by law."},
            {"page_number": 12, "line_number": 4, "text": "This is the second sentence of the provision.", "raw_line": "4 This is the second sentence of the provision."},
        ]
        return {
            "section_number": "10",
            "section_title": "What agreements are contracts",
            "chapter": "CHAPTER II",
            "page_number": 12,
            "end_page_number": 12,
            "source_pages": [12],
            "source_spans": lines,
            "text": "\n".join(line["text"] for line in lines),
            "section_status": "in_force",
            "review_required": False,
            "amendments": [],
            "footnotes": [],
            "ocr_anomalies": [],
        }

    def approved_manifest(self, section):
        record = build_review_manifest([section])[0]
        digest = section_text_hash(section["text"])
        record = approve_review_record(
            record, reviewer="Extraction reviewer",
            reviewed_at="2026-11-12T12:00:00Z",
            reviewed_text_sha256=digest,
        )
        return [approve_legal_content_record(
            record, reviewer="Legal reviewer",
            reviewed_at="2026-11-12T13:00:00Z",
            reviewed_text_sha256=digest,
            authority_reference="Official Gazette, Section 10",
            resolution_notes="Compared against the authoritative text.",
            resolved_review_reasons=record["review_reasons"],
        )]

    def test_clean_candidate_chunks_pass_with_review_history(self):
        section = self.section()
        manifest = self.approved_manifest(section)
        chunks = create_legal_chunks([section], chunk_size=200, overlap=0, review_manifest=manifest)
        report = audit_chunk_provenance([section], chunks, manifest)
        self.assertTrue(report["provenance_integrity_passed"])
        self.assertEqual(report["candidate_chunk_count"], len(chunks))
        self.assertEqual(chunks[0]["reviewed_by"], "Extraction reviewer")
        self.assertEqual(chunks[0]["legal_authority_reference"], "Official Gazette, Section 10")

    def test_missing_or_changed_source_span_is_reported(self):
        section = self.section()
        manifest = self.approved_manifest(section)
        chunks = create_legal_chunks([section], chunk_size=200, overlap=0, review_manifest=manifest)
        chunks[0]["source_spans"][0]["text"] = "invented text"
        report = audit_chunk_provenance([section], chunks, manifest)
        self.assertFalse(report["provenance_integrity_passed"])
        self.assertIn("source_line_excerpt_not_found", [item["issue"] for item in report["issues"]])

    def test_annotation_tampering_is_reported(self):
        section = self.section()
        manifest = self.approved_manifest(section)
        chunks = create_legal_chunks([section], chunk_size=200, overlap=0, review_manifest=manifest)
        chunks[0]["ocr_anomalies"] = [{"text": "hidden"}]
        report = audit_chunk_provenance([section], chunks, manifest)
        self.assertIn("ocr_anomalies_mismatch", [item["issue"] for item in report["issues"]])

    def test_unreviewed_section_cannot_pass(self):
        section = self.section()
        manifest = build_review_manifest([section])
        chunks = create_legal_chunks([section], chunk_size=200, overlap=0, review_manifest=manifest)
        report = audit_chunk_provenance([section], chunks, manifest)
        self.assertFalse(report["provenance_integrity_passed"])
        self.assertIn("no_valid_training_candidates", [item["issue"] for item in report["issues"]])

    def test_chunk_without_candidate_section_is_blocked(self):
        section = self.section()
        manifest = self.approved_manifest(section)
        section["section_status"] = "repealed"
        chunks = create_legal_chunks([section], chunk_size=200, overlap=0, review_manifest=manifest)
        report = audit_chunk_provenance([section], chunks, manifest)
        self.assertIn("parent_section_not_training_candidate", [item["issue"] for item in report["issues"]])


if __name__ == "__main__":
    unittest.main()
