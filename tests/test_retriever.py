"""Tests for legal retrieval safeguards."""
import unittest

from src.document.retriever import build_index, search


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "chunk_id": "active-1",
                "section_number": "10",
                "section_title": "Valid agreements",
                "section_status": "unknown",
                "text": "An agreement is a contract when it meets the legal conditions.",
            },
            {
                "chunk_id": "repealed-1",
                "section_number": "76",
                "section_title": "Repealed provision",
                "section_status": "repealed",
                "text": "An agreement is a contract under this historical provision.",
            },
        ]
        self.idf, self.vectors = build_index(self.chunks)

    def test_repealed_sections_are_excluded_by_default(self):
        results = search("agreement contract", self.chunks, self.idf, self.vectors)
        self.assertNotIn("repealed-1", [item["chunk_id"] for item in results])

    def test_repealed_sections_can_be_requested_for_historical_research(self):
        results = search(
            "agreement contract", self.chunks, self.idf, self.vectors,
            include_repealed=True,
        )
        self.assertIn("repealed-1", [item["chunk_id"] for item in results])


if __name__ == "__main__":
    unittest.main()
