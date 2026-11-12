"""Tests for the human legal extraction-review manifest."""
import unittest

from src.document.legal_review import (
    approve_review_record,
    build_review_manifest,
    section_text_hash,
    validate_review_manifest,
)


class LegalReviewTests(unittest.TestCase):
    def section(self, number="1", text="This is a sufficiently detailed provision text for review."):
        return {
            "section_number": number,
            "section_title": "Test provision",
            "page_number": 10,
            "end_page_number": 10,
            "source_pages": [10],
            "heading_source": {"page_number": 10, "line_number": 1, "raw_line": "1. Test.—"},
            "source_spans": [{"page_number": 10, "line_number": 2, "text": text}],
            "text": text,
            "chapter": "CHAPTER I",
            "amendments": [],
            "footnotes": [],
            "ocr_anomalies": [],
            "section_status": "unknown",
            "review_required": False,
        }

    def test_every_section_starts_pending_and_not_training_ready(self):
        manifest = build_review_manifest([self.section()])
        self.assertEqual(manifest[0]["review_status"], "pending")
        self.assertIsNone(manifest[0]["reviewed_by"])
        self.assertFalse(manifest[0]["ready_for_training"])
        self.assertFalse(manifest[0]["legal_accuracy_verified"])

    def test_review_reasons_preserve_flags_and_annotations(self):
        section = self.section()
        section["ocr_anomalies"] = [{"text": "1151."}]
        section["amendments"] = [{"text": "Amendment"}]
        section["footnotes"] = [{"text": "Footnote"}]
        manifest = build_review_manifest([section])
        reasons = manifest[0]["review_reasons"]
        self.assertIn("ocr_anomalies_present", reasons)
        self.assertIn("amendment_annotations_present", reasons)
        self.assertIn("footnote_annotations_present", reasons)
        self.assertIn("section_status_unverified", reasons)
        self.assertEqual(manifest[0]["ocr_anomalies"], section["ocr_anomalies"])

    def test_missing_provenance_and_short_text_are_flagged(self):
        section = self.section(text="Short")
        section.pop("heading_source")
        section["source_spans"] = []
        reasons = build_review_manifest([section])[0]["review_reasons"]
        self.assertIn("missing_heading_provenance", reasons)
        self.assertIn("missing_body_source_spans", reasons)
        self.assertIn("unusually_short_section_text", reasons)

    def test_duplicate_section_numbers_are_flagged(self):
        manifest = build_review_manifest([self.section(), self.section()])
        self.assertIn("duplicate_section_number", manifest[0]["review_reasons"])
        self.assertIn("duplicate_section_number", manifest[1]["review_reasons"])

    def test_approval_requires_reviewer_timestamp_and_exact_hash(self):
        record = build_review_manifest([self.section()])[0]
        digest = record["source_text_sha256"]
        with self.assertRaises(ValueError):
            approve_review_record(record, reviewer="", reviewed_at="2026-11-12T12:00:00Z", reviewed_text_sha256=digest)
        with self.assertRaises(ValueError):
            approve_review_record(record, reviewer="Reviewer", reviewed_at="", reviewed_text_sha256=digest)
        with self.assertRaises(ValueError):
            approve_review_record(record, reviewer="Reviewer", reviewed_at="2026-11-12T12:00:00Z", reviewed_text_sha256="wrong")

    def test_valid_approval_is_scoped_and_validated(self):
        section = self.section()
        record = build_review_manifest([section])[0]
        approved = approve_review_record(
            record,
            reviewer="Human reviewer",
            reviewed_at="2026-11-12T12:00:00Z",
            reviewed_text_sha256=section_text_hash(section["text"]),
            reviewer_notes="Compared with source PDF.",
        )
        report = validate_review_manifest([section], [approved])
        self.assertEqual(report["approved_count"], 1)
        self.assertTrue(report["all_sections_have_valid_review_records"])
        self.assertFalse(approved["legal_accuracy_verified"])
        self.assertFalse(approved["ready_for_training"])

    def test_changed_source_text_invalidates_existing_approval(self):
        section = self.section()
        record = build_review_manifest([section])[0]
        approved = approve_review_record(
            record,
            reviewer="Human reviewer",
            reviewed_at="2026-11-12T12:00:00Z",
            reviewed_text_sha256=record["source_text_sha256"],
        )
        changed = dict(section, text=section["text"] + " changed")
        report = validate_review_manifest([changed], [approved])
        self.assertFalse(report["all_sections_have_valid_review_records"])
        self.assertEqual(report["issues"][0]["issue"], "source_text_hash_mismatch")

    def test_missing_manifest_record_is_reported(self):
        report = validate_review_manifest([self.section()], [])
        self.assertEqual(report["missing_section_indices"], [0])
        self.assertFalse(report["all_sections_have_valid_review_records"])


if __name__ == "__main__":
    unittest.main()
