"""Tests for inference-to-evaluation run logging."""
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.legal_run_logger import (
    build_run_record,
    write_run_record_new,
)


class LegalRunLoggerTests(unittest.TestCase):
    def test_records_answer_and_retrieved_citations(self):
        chunks = [{
            "section_number": "10",
            "text": "A provision text.",
            "page_number": 12,
            "source_spans": [{"page_number": 12, "line_number": 4}],
        }]
        record = build_run_record("case-1", "Generated answer", chunks, False)
        self.assertEqual(record["answer"], "Generated answer")
        self.assertEqual(record["retrieved_chunks"], chunks)
        self.assertFalse(record["abstained"])

    def test_abstention_keeps_displayed_refusal_but_no_substantive_answer(self):
        record = build_run_record("case-2", "No evidence found.", [], True)
        self.assertEqual(record["answer"], "")
        self.assertEqual(record["displayed_response"], "No evidence found.")
        self.assertTrue(record["abstained"])

    def test_invalid_query_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "query_id"):
            build_run_record(" ", "answer", [], False)

    def test_non_object_chunks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "list of objects"):
            build_run_record("case-1", "answer", ["bad"], False)

    def test_write_creates_jsonl_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            record = build_run_record("case-1", "answer", [], False)
            write_run_record_new(record, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, record)
            with self.assertRaises(FileExistsError):
                write_run_record_new(record, path)


if __name__ == "__main__":
    unittest.main()
