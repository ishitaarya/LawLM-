"""Tests for read-only legal extraction validation."""

import unittest

from src.document.legal_validation import validate_sections


class LegalValidationTests(unittest.TestCase):
    def _section(self, number="1", text="Provision text.", pages=None):
        pages = [10] if pages is None else pages
        return {
            "section_number": number,
            "section_title": "Test heading",
            "page_number": pages[0] if pages else 10,
            "end_page_number": pages[-1] if pages else 10,
            "source_pages": pages,
            "heading_source": {
                "page_number": pages[0] if pages else 10,
                "line_number": 1,
                "raw_line": f"{number}. Test heading.—{text}",
            },
            "source_spans": (
                [{"page_number": page, "line_number": 2, "text": text, "raw_line": text}
                 for page in pages] if text else []
            ),
            "text": text,
            "amendments": [],
            "footnotes": [],
            "ocr_anomalies": [],
            "review_required": True,
        }

    def test_clean_structure_is_reported_without_claiming_legal_accuracy(self):
        report = validate_sections([self._section()])
        self.assertTrue(report["checks"]["no_duplicate_section_numbers"])
        self.assertFalse(report["legal_accuracy_verified"])
        self.assertFalse(report["ready_for_training"])

    def test_duplicate_section_numbers_are_reported(self):
        report = validate_sections([self._section(), self._section()])
        self.assertEqual(report["duplicate_section_numbers"], ["1"])

    def test_missing_text_and_page_provenance_are_reported(self):
        section = self._section(text="", pages=[])
        report = validate_sections([section])
        self.assertFalse(report["checks"]["all_sections_have_text"])
        self.assertFalse(report["checks"]["all_sections_have_page_provenance"])

    def test_annotation_and_ocr_counts_are_reported(self):
        section = self._section()
        section["amendments"] = [{"text": "State amendment"}]
        section["footnotes"] = [{"text": "Footnote"}]
        section["ocr_anomalies"] = [{"text": "1151."}]
        report = validate_sections([section])
        self.assertEqual(report["amendment_annotation_count"], 1)
        self.assertEqual(report["footnote_annotation_count"], 1)
        self.assertEqual(report["ocr_anomaly_count"], 1)

    def test_invalid_page_range_is_reported(self):
        section = self._section()
        section["end_page_number"] = 9
        report = validate_sections([section])
        self.assertEqual(len(report["invalid_page_ranges"]), 1)

    def test_missing_section_number_is_reported(self):
        report = validate_sections([self._section(number="")])
        self.assertEqual(report["missing_section_number_count"], 1)
        self.assertFalse(report["checks"]["all_sections_have_numbers"])

    def test_missing_heading_and_body_line_provenance_are_reported(self):
        section = self._section()
        section.pop("heading_source")
        section["source_spans"] = [
            {"page_number": 99, "line_number": 0, "text": ""}
        ]
        report = validate_sections([section])
        self.assertEqual(report["sections_missing_heading_line_provenance"], ["1"])
        self.assertEqual(report["sections_with_invalid_body_line_provenance"], ["1"])
        self.assertFalse(report["checks"]["all_sections_have_heading_line_provenance"])
        self.assertFalse(report["checks"]["all_body_spans_have_valid_page_line_provenance"])


if __name__ == "__main__":
    unittest.main()
