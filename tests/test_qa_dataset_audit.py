"""Tests for read-only legal QA dataset auditing."""
import unittest

from src.training.qa_dataset_audit import audit_records


def record(question="When does an agreement become a contract?"):
    return {
        "question": question,
        "context": "A" * 120,
        "answer": "An agreement becomes a contract when the legal requirements are satisfied.",
        "section": "10",
    }


class QADatasetAuditTests(unittest.TestCase):
    def test_clean_record_is_counted_but_not_declared_training_ready(self):
        report = audit_records([record()])
        self.assertEqual(report["structurally_valid_record_count"], 1)
        self.assertFalse(report["ready_for_training"])
        self.assertFalse(report["legal_accuracy_verified"])

    def test_sentence_truncation_and_hard_token_flags_are_reported(self):
        item = record()
        item["answer"] = "An agreement becomes a contract when"
        item["hard_truncated"] = True
        report = audit_records([item])
        issue_codes = report["issues"][0]["issues"]
        self.assertIn("answer_may_be_sentence_truncated", issue_codes)
        self.assertIn("hard_truncation_flagged", issue_codes)
        self.assertEqual(report["hard_truncated_record_indices"], [1])

    def test_duplicate_questions_are_flagged_for_manual_review(self):
        report = audit_records([record(), record()])
        self.assertEqual(len(report["duplicate_questions"]), 1)
        self.assertEqual(report["records_requiring_review_count"], 2)

    def test_missing_and_empty_fields_are_reported(self):
        report = audit_records([{"question": "", "answer": "Some answer."}])
        codes = report["issues"][0]["issues"]
        self.assertIn("missing_context", codes)
        self.assertIn("missing_section", codes)
        self.assertIn("empty_question", codes)


if __name__ == "__main__":
    unittest.main()
