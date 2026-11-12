"""Tests for the conservative legal training-candidate gate."""
import unittest

from src.document.legal_review import (
    approve_legal_content_record,
    approve_review_record,
    build_review_manifest,
    section_text_hash,
)
from src.training.legal_training_gate import evaluate_training_candidates


class LegalTrainingGateTests(unittest.TestCase):
    def section(self):
        text = "Section provision text long enough for a normal extraction review."
        return {
            "section_number": "1",
            "section_title": "Test section",
            "text": text,
            "section_status": "in_force",
        }

    def fully_reviewed(self, section):
        record = build_review_manifest([section])[0]
        # These fields are clean in this fixture; status is supplied from an
        # explicit source review rather than inferred by the gate.
        record = approve_review_record(
            record,
            reviewer="Extraction reviewer",
            reviewed_at="2026-11-12T12:00:00Z",
            reviewed_text_sha256=section_text_hash(section["text"]),
        )
        return approve_legal_content_record(
            record,
            reviewer="Legal reviewer",
            reviewed_at="2026-11-12T13:00:00Z",
            reviewed_text_sha256=section_text_hash(section["text"]),
            authority_reference="Official Gazette, Act text, Section 1",
            resolution_notes="Compared provision and current status with cited source.",
            resolved_review_reasons=record["review_reasons"],
        )

    def test_requires_both_review_stages(self):
        section = self.section()
        manifest = build_review_manifest([section])
        report = evaluate_training_candidates([section], manifest)
        self.assertEqual(report["candidate_count"], 0)
        self.assertIn("extraction_review_not_approved", [i["issue"] for i in report["issues"]])

    def test_fully_reviewed_in_force_section_becomes_candidate_only(self):
        section = self.section()
        manifest = [self.fully_reviewed(section)]
        report = evaluate_training_candidates([section], manifest)
        self.assertEqual(report["candidate_section_indices"], [0])
        self.assertFalse(report["ready_for_training"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["dataset_exported"])

    def test_unknown_or_repealed_status_is_blocked(self):
        for status in ("unknown", "repealed"):
            section = dict(self.section(), section_status=status)
            manifest = [self.fully_reviewed(section)]
            report = evaluate_training_candidates([section], manifest)
            self.assertEqual(report["candidate_count"], 0)
            self.assertIn("legal_status_not_verified_in_force", [i["issue"] for i in report["issues"]])

    def test_unresolved_review_reason_blocks_candidate(self):
        section = dict(self.section(), ocr_anomalies=[{"text": "ambiguous"}])
        manifest = [self.fully_reviewed(section)]
        manifest[0]["resolved_review_reasons"] = []
        report = evaluate_training_candidates([section], manifest)
        self.assertIn("review_reasons_unresolved", [i["issue"] for i in report["issues"]])

    def test_source_change_invalidates_candidate(self):
        section = self.section()
        manifest = [self.fully_reviewed(section)]
        changed = dict(section, text=section["text"] + " edited")
        report = evaluate_training_candidates([changed], manifest)
        self.assertEqual(report["candidate_count"], 0)
        self.assertIn("source_text_hash_mismatch", [i["issue"] for i in report["issues"]])

    def test_missing_authority_reference_blocks_candidate(self):
        section = self.section()
        record = self.fully_reviewed(section)
        record["legal_authority_reference"] = ""
        report = evaluate_training_candidates([section], [record])
        self.assertIn("legal_content_approval_invalid", [i["issue"] for i in report["issues"]])


if __name__ == "__main__":
    unittest.main()
