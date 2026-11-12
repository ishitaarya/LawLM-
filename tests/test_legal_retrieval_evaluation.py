"""Tests for legal retrieval ranking evaluation."""
import unittest

from src.evaluation.legal_retrieval_evaluation import evaluate_retrieval


class LegalRetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "chunk_id": "section-10",
                "section_number": "10",
                "section_title": "What agreements are contracts",
                "section_status": "unknown",
                "text": "All agreements are contracts if made by free consent of parties competent to contract.",
            },
            {
                "chunk_id": "section-23",
                "section_number": "23",
                "section_title": "What consideration and objects are lawful",
                "section_status": "unknown",
                "text": "The consideration or object of an agreement is lawful unless forbidden by law.",
            },
            {
                "chunk_id": "section-76",
                "section_number": "76",
                "section_title": "Historical repealed provision",
                "section_status": "repealed",
                "text": "A valid contract is described in this historical provision.",
            },
        ]

    def test_expected_section_metrics_are_calculated(self):
        report = evaluate_retrieval(
            [{"query": "What makes an agreement a valid contract?",
              "expected_sections": ["10"]}],
            self.chunks,
        )
        self.assertEqual(report["valid_query_count"], 1)
        self.assertEqual(report["metrics"]["hit_rate_at_1"], 1.0)
        self.assertEqual(report["metrics"]["mean_reciprocal_rank"], 1.0)
        self.assertFalse(report["legal_answer_accuracy_verified"])

    def test_repealed_section_is_not_retrieved_by_default(self):
        report = evaluate_retrieval(
            [{"query": "valid contract", "expected_sections": ["76"]}],
            self.chunks,
        )
        self.assertIsNone(report["query_results"][0]["first_relevant_rank"])

    def test_missing_expected_section_is_reported_as_invalid_query(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval([{"query": "A legal question"}], self.chunks)


if __name__ == "__main__":
    unittest.main()
