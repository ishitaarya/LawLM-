"""Tests for traceable, section-aware legal chunking."""
import unittest

from src.document.legal_chunker import create_legal_chunks


def section(number="10", text_lines=None, status="unknown"):
    text_lines = text_lines or [
        {"page_number": 20, "line_number": 4, "text": "A" * 30, "raw_line": "raw A"},
        {"page_number": 21, "line_number": 1, "text": "B" * 30, "raw_line": "raw B"},
    ]
    return {
        "section_number": number,
        "section_title": "Test provision",
        "chapter": "CHAPTER II",
        "source_spans": text_lines,
        "section_status": status,
        "review_required": True,
        "amendments": [],
        "footnotes": [],
        "ocr_anomalies": [],
    }


class LegalChunkerTests(unittest.TestCase):
    def test_chunks_keep_exact_source_page_and_line_references(self):
        chunks = create_legal_chunks([section()], chunk_size=40, overlap=5)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section_number"], "10")
        self.assertEqual(chunks[0]["source_pages"], [20, 21])
        self.assertEqual(
            [(s["page_number"], s["line_number"]) for s in chunks[0]["source_spans"]],
            [(20, 4), (21, 1)],
        )
        self.assertEqual(chunks[0]["page_number"], 20)
        self.assertEqual(chunks[0]["end_page_number"], 21)

    def test_chunks_never_cross_section_boundaries(self):
        chunks = create_legal_chunks(
            [section("10"), section("11")], chunk_size=100, overlap=10
        )
        self.assertEqual({chunk["section_number"] for chunk in chunks}, {"10", "11"})

    def test_empty_or_missing_source_provenance_is_rejected(self):
        broken = section()
        broken["source_spans"] = []
        with self.assertRaises(ValueError):
            create_legal_chunks([broken])

    def test_chunking_preserves_review_and_annotation_flags(self):
        item = section(status="repealed")
        item["ocr_anomalies"] = [{"text": "1151."}]
        item["amendments"] = [{"text": "State amendment text"}]
        item["footnotes"] = [{"text": "Footnote text"}]
        chunks = create_legal_chunks([item], chunk_size=100, overlap=0)
        self.assertTrue(chunks[0]["review_required"])
        self.assertTrue(chunks[0]["has_ocr_anomalies"])
        self.assertEqual(chunks[0]["section_status"], "repealed")
        self.assertEqual(chunks[0]["amendment_annotations"][0]["text"], "State amendment text")
        self.assertEqual(chunks[0]["footnote_annotations"][0]["text"], "Footnote text")
        self.assertEqual(chunks[0]["ocr_anomalies"][0]["text"], "1151.")

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            create_legal_chunks([section()], chunk_size=20, overlap=20)


if __name__ == "__main__":
    unittest.main()
