"""Tests for page-level legal PDF extraction audit."""
import unittest

from src.document.legal_source_audit import audit_pages


class LegalSourceAuditTests(unittest.TestCase):
    def page(self, number, text="This is a sufficiently long extracted legal page text."):
        return {
            "source_file": "Indian_Contract_Act_1872.pdf",
            "page_number": number,
            "text": text,
        }

    def test_clean_page_sequence_passes_structural_checks_only(self):
        report = audit_pages([self.page(1), self.page(2)])
        self.assertTrue(report["structural_page_integrity_passed"])
        self.assertFalse(report["legal_accuracy_verified"])
        self.assertFalse(report["ocr_quality_verified"])
        self.assertFalse(report["ready_for_training"])

    def test_duplicate_and_missing_page_numbers_are_reported(self):
        report = audit_pages([self.page(1), self.page(1), self.page(3)])
        self.assertEqual(report["duplicate_page_numbers"], [1])
        self.assertEqual(report["missing_page_numbers_in_observed_range"], [2])
        self.assertFalse(report["structural_page_integrity_passed"])

    def test_empty_short_and_ocr_warning_pages_are_reported(self):
        report = audit_pages([
            self.page(1, ""),
            self.page(2, "Short"),
            self.page(3, "Broken " + chr(0xFFFD) + " extraction text " * 5),
        ])
        self.assertEqual(report["empty_pages"], [1])
        self.assertEqual(report["suspiciously_short_pages"], [2])
        self.assertEqual(report["pages_with_replacement_characters"], [3])

    def test_inconsistent_source_files_are_reported(self):
        first = self.page(1)
        second = self.page(2)
        second["source_file"] = "different.pdf"
        report = audit_pages([first, second])
        self.assertEqual(
            report["inconsistent_source_files"],
            ["Indian_Contract_Act_1872.pdf", "different.pdf"],
        )
        self.assertFalse(report["checks"]["single_source_file"])


if __name__ == "__main__":
    unittest.main()
