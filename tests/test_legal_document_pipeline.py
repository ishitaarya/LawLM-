"""Tests for the in-memory legal document artifact pipeline."""
import unittest

from src.document.legal_document_pipeline import prepare_document_artifacts


class LegalDocumentPipelineTests(unittest.TestCase):
    def test_pages_flow_to_sections_report_and_traceable_chunks(self):
        pages = [{
            "source_file": "sample.pdf",
            "page_number": 10,
            "text": (
                "It is hereby enacted as follows:\n"
                "1. Short title.—This Act may be called the Sample Act.\n"
                "It extends to the whole of India."
            ),
        }]
        sections, report, chunks = prepare_document_artifacts(
            pages, chunk_size=200, overlap=20
        )
        self.assertEqual(len(sections), 1)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(report["checks"]["all_sections_have_heading_line_provenance"])
        self.assertEqual(chunks[0]["section_number"], "1")
        self.assertEqual(chunks[0]["source_pages"], [10])
        self.assertTrue(chunks[0]["source_spans"])

    def test_empty_assembly_is_rejected(self):
        pages = [{
            "source_file": "sample.pdf",
            "page_number": 1,
            "text": "A table of contents with no enactment marker.",
        }]
        with self.assertRaises(ValueError):
            prepare_document_artifacts(pages)

    def test_manually_verified_body_start_can_be_supplied(self):
        pages = [{
            "source_file": "sample.pdf",
            "page_number": 10,
            "text": "1. Short title.—This Act may be called the Sample Act.",
        }]
        sections, report, chunks = prepare_document_artifacts(
            pages, body_start_page=10, chunk_size=200, overlap=20
        )
        self.assertEqual(sections[0]["section_number"], "1")
        self.assertEqual(len(chunks), 1)
        self.assertFalse(report["legal_accuracy_verified"])


if __name__ == "__main__":
    unittest.main()
