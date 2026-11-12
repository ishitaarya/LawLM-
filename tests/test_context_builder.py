"""Tests for citation-aware legal context construction."""
import unittest

from src.document.context_builder import build_context


class ContextBuilderTests(unittest.TestCase):
    def test_context_includes_page_range_and_source_line_citations(self):
        context = build_context(
            "Explain Section 10.",
            [{
                "section_number": "10",
                "section_title": "What agreements are contracts",
                "page_number": 12,
                "end_page_number": 13,
                "section_status": "unknown",
                "amendment_annotations": [
                    {"page_number": 13, "line_number": 5, "text": "State amendment wording."}
                ],
                "footnote_annotations": [
                    {"page_number": 13, "line_number": 6, "text": "Footnote explanation."}
                ],
                "ocr_anomalies": [{"text": "1151."}],
                "source_spans": [
                    {"page_number": 12, "line_number": 4, "text": "An agreement."},
                    {"page_number": 13, "line_number": 2, "text": "A contract."},
                ],
                "text": "An agreement. A contract.",
            }],
        )
        self.assertIn("Pages: 12-13", context)
        self.assertIn("p12:L4; p13:L2", context)
        self.assertIn("Section status: unknown", context)
        self.assertIn("Review required: True", context)
        self.assertIn("OCR anomalies=True", context)
        self.assertIn("Separate amendment annotation", context)
        self.assertIn("State amendment wording.", context)
        self.assertIn("Separate footnote", context)
        self.assertIn("OCR anomaly requiring manual review: 1151.", context)

    def test_empty_results_are_not_presented_as_evidence(self):
        context = build_context("Question", [])
        self.assertIn("No relevant legal context was retrieved.", context)


if __name__ == "__main__":
    unittest.main()
