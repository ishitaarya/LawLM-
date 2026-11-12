"""Integration tests for retrieval-to-answer-evaluation orchestration."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.document.qa_orchestrator import NO_EVIDENCE_RESPONSE, run_legal_qa
from src.evaluation.legal_answer_evaluation import evaluate_answers, gold_review_hash


def approved_gold(
    query_id: str,
    answerable: bool,
    expected_sections: list[str],
    evidence_quote: str,
) -> dict:
    gold = {
        "query_id": query_id,
        "question": "Synthetic integration-test question?",
        "expected_answer": "Synthetic expected answer for human comparison.",
        "expected_sections": expected_sections,
        "evidence_quote": evidence_quote,
        "answerable": answerable,
        "review_status": "answer_benchmark_approved",
        "reviewed_by": "CI synthetic fixture",
        "reviewed_at": "2026-11-12T00:00:00Z",
    }
    gold["reviewed_content_sha256"] = gold_review_hash(gold)
    return gold


class LegalQAOrchestrationIntegrationTests(unittest.TestCase):
    def test_retrieval_generation_logging_and_evaluation_preserve_citations(self):
        chunk = {
            "chunk_id": "synthetic-section-10",
            "section_number": "10",
            "section_title": "Synthetic agreement provision",
            "page_number": 12,
            "end_page_number": 13,
            "text": "Synthetic evidence quote: an agreement enforceable by law.",
            "source_spans": [
                {"page_number": 12, "line_number": 8, "text": "Synthetic evidence quote."}
            ],
        }
        generated_calls = []

        def retrieve(query: str, top_k: int) -> list[dict]:
            self.assertEqual(query, "Synthetic question?")
            self.assertEqual(top_k, 2)
            return [chunk]

        def generate(query: str, context: str) -> str:
            generated_calls.append((query, context))
            self.assertIn("Section: 10", context)
            self.assertIn("p12:L8", context)
            return "Synthetic answer based on the supplied test evidence."

        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "run.jsonl"
            answer, retrieved, context = run_legal_qa(
                "Synthetic question?", retrieve, generate, top_k=2,
                query_id="synthetic-answerable",
                evaluation_record_path=run_path,
            )
            saved = json.loads(run_path.read_text(encoding="utf-8").strip())
            report = evaluate_answers(
                [approved_gold(
                    "synthetic-answerable", True, ["10"],
                    "Synthetic evidence quote: an agreement enforceable by law."
                )],
                [saved],
            )

        self.assertEqual(answer, "Synthetic answer based on the supplied test evidence.")
        self.assertEqual(retrieved[0]["source_spans"], chunk["source_spans"])
        self.assertIn("Section: 10", context)
        self.assertEqual(len(generated_calls), 1)
        self.assertEqual(saved["retrieved_chunks"][0]["source_spans"], chunk["source_spans"])
        self.assertFalse(saved["abstained"])
        checks = report["outcomes"][0]["checks"]
        self.assertTrue(checks["answer_present"])
        self.assertTrue(checks["expected_section_retrieved"])
        self.assertTrue(checks["evidence_quote_retrieved"])
        self.assertTrue(checks["citation_metadata_present"])
        self.assertFalse(checks["unsupported_answer_risk"])
        self.assertFalse(report["legal_answer_accuracy_verified"])

    def test_no_evidence_abstains_without_calling_generator_and_logs_refusal(self):
        generator_called = []

        def retrieve(query: str, top_k: int) -> list[dict]:
            return []

        def generate(query: str, context: str) -> str:
            generator_called.append(True)
            return "This must never run."

        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "abstention.jsonl"
            answer, retrieved, context = run_legal_qa(
                "Unmatched synthetic question?", retrieve, generate,
                query_id="synthetic-unanswerable",
                evaluation_record_path=run_path,
            )
            saved = json.loads(run_path.read_text(encoding="utf-8").strip())
            report = evaluate_answers(
                [approved_gold(
                    "synthetic-unanswerable", False, ["99"],
                    "Evidence deliberately absent from the synthetic fixture."
                )],
                [saved],
            )

        self.assertEqual(answer, NO_EVIDENCE_RESPONSE)
        self.assertEqual(retrieved, [])
        self.assertIn("No relevant legal context", context)
        self.assertEqual(generator_called, [])
        self.assertEqual(saved["answer"], "")
        self.assertEqual(saved["displayed_response"], NO_EVIDENCE_RESPONSE)
        self.assertTrue(saved["abstained"])
        self.assertTrue(report["outcomes"][0]["checks"]["correct_abstention_behavior"])
        self.assertFalse(report["legal_answer_accuracy_verified"])

    def test_empty_generated_text_is_logged_as_abstention(self):
        chunk = {"section_number": "10", "text": "Synthetic evidence.", "page_number": 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty-answer.jsonl"
            answer, _, _ = run_legal_qa(
                "Synthetic question?",
                lambda query, top_k: [chunk],
                lambda query, context: "",
                query_id="synthetic-empty",
                evaluation_record_path=path,
            )
            saved = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(answer, "")
        self.assertTrue(saved["abstained"])
        self.assertEqual(saved["answer"], "")

    def test_logger_refuses_to_overwrite_existing_run_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.jsonl"
            path.write_text("keep this file", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_legal_qa(
                    "Synthetic question?",
                    lambda query, top_k: [],
                    lambda query, context: "",
                    query_id="synthetic-no-overwrite",
                    evaluation_record_path=path,
                )
            self.assertEqual(path.read_text(encoding="utf-8"), "keep this file")


if __name__ == "__main__":
    unittest.main()
