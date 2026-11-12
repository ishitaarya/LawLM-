"""Tests for human-reviewed legal answer evaluation."""
import unittest

from src.evaluation.legal_answer_evaluation import (
    evaluate_answers,
    gold_review_hash,
)


class LegalAnswerEvaluationTests(unittest.TestCase):
    def make_gold(self, **changes):
        gold = {
            "query_id": "q1",
            "question": "When is an agreement a contract?",
            "expected_answer": "An agreement enforceable by law is a contract.",
            "expected_sections": ["10"],
            "evidence_quote": "All agreements are contracts if made by free consent.",
            "answerable": True,
            "review_status": "answer_benchmark_approved",
            "reviewed_by": "legal reviewer",
            "reviewed_at": "2026-11-12T10:00:00Z",
        }
        gold.update(changes)
        gold["reviewed_content_sha256"] = gold_review_hash(gold)
        return gold

    def test_supported_answer_has_evidence_and_citation_signals(self):
        report = evaluate_answers(
            [self.make_gold()],
            [{
                "query_id": "q1",
                "answer": "An agreement made by free consent can become a contract.",
                "retrieved_chunks": [{
                    "section_number": "10",
                    "text": "All agreements are contracts if made by free consent.",
                    "page_number": 10,
                    "source_spans": [{"page_number": 10, "line_number": 4}],
                }],
                "abstained": False,
            }],
        )
        self.assertEqual(report["metrics"]["expected_section_retrieved"], 1.0)
        self.assertEqual(report["metrics"]["evidence_quote_retrieved"], 1.0)
        self.assertEqual(report["metrics"]["citation_metadata_present"], 1.0)
        self.assertFalse(report["legal_answer_accuracy_verified"])
        self.assertTrue(report["human_semantic_review_required"])

    def test_answer_without_retrieved_gold_evidence_is_flagged(self):
        report = evaluate_answers(
            [self.make_gold()],
            [{
                "query_id": "q1",
                "answer": "A contract is always valid.",
                "retrieved_chunks": [{
                    "section_number": "99",
                    "text": "Unrelated provision.",
                }],
                "abstained": False,
            }],
        )
        self.assertEqual(report["metrics"]["unsupported_answer_risk"], 1.0)
        self.assertEqual(report["metrics"]["expected_section_retrieved"], 0.0)

    def test_unanswerable_case_requires_empty_abstention(self):
        gold = self.make_gold(answerable=False, expected_sections=["10"])
        report = evaluate_answers(
            [gold],
            [{
                "query_id": "q1", "answer": "",
                "retrieved_chunks": [], "abstained": True,
            }],
        )
        self.assertEqual(report["metrics"]["correct_abstention_behavior"], 1.0)

    def test_modified_gold_requires_new_human_approval_hash(self):
        gold = self.make_gold()
        gold["expected_answer"] = "Changed after approval."
        with self.assertRaisesRegex(ValueError, "hash"):
            evaluate_answers([gold], [])

    def test_missing_run_is_counted_without_fabricating_output(self):
        report = evaluate_answers([self.make_gold()], [])
        self.assertEqual(report["missing_run_count"], 1)
        self.assertEqual(report["evaluated_case_count"], 0)

    def test_unknown_run_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent from gold"):
            evaluate_answers([self.make_gold()], [{
                "query_id": "unknown", "answer": "",
                "retrieved_chunks": [], "abstained": True,
            }])


if __name__ == "__main__":
    unittest.main()
