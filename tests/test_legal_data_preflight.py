"""Tests for the read-only real-data preflight checker."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.legal_data_preflight import run_preflight


def page_record(number: int) -> dict:
    return {
        "source_file": "synthetic_act.pdf",
        "page_number": number,
        "text": "Synthetic extracted legal text with sufficient length for structural audit.",
    }


def qa_record(question: str, section: str = "10") -> dict:
    return {
        "question": question,
        "context": "Synthetic context " * 12,
        "answer": "Synthetic answer with enough words to pass structural checks.",
        "section": section,
    }


class LegalDataPreflightTests(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in records),
            encoding="utf-8",
        )

    def make_inputs(self, root: Path):
        pdf = root / "act.pdf"
        pdf.write_bytes(b"%PDF- synthetic test fixture")
        pages = root / "pages.jsonl"
        train = root / "train.jsonl"
        validation = root / "validation.jsonl"
        test = root / "test.jsonl"
        self.write_jsonl(pages, [page_record(1), page_record(2)])
        self.write_jsonl(train, [qa_record("Train split synthetic question?")])
        self.write_jsonl(validation, [qa_record("Validation split synthetic question?", "11")])
        self.write_jsonl(test, [qa_record("Test split synthetic question?", "12")])
        return pages, pdf, train, validation, test

    def test_clean_synthetic_inputs_pass_structural_preflight_only(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_inputs(Path(directory))
            report = run_preflight(*paths)
        self.assertTrue(report["all_required_files_available"])
        self.assertTrue(report["structural_preflight_passed"])
        self.assertFalse(report["legal_accuracy_verified"])
        self.assertFalse(report["ready_for_training"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["source_data_modified"])

    def test_missing_real_inputs_are_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_preflight(
                root / "missing-pages.jsonl",
                root / "missing.pdf",
                root / "missing-train.jsonl",
                root / "missing-validation.jsonl",
                root / "missing-test.jsonl",
            )
        self.assertFalse(report["all_required_files_available"])
        self.assertFalse(report["structural_preflight_passed"])
        self.assertEqual(
            report["inputs"]["qa_train"]["error"], "file_missing"
        )

    def test_duplicate_questions_across_splits_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = list(self.make_inputs(root))
            self.write_jsonl(paths[2], [qa_record("Same question across splits?")])
            self.write_jsonl(paths[3], [qa_record("  SAME   QUESTION across splits?")])
            report = run_preflight(*paths)
        self.assertFalse(report["cross_split_checks"]["question_split_isolation_passed"])
        self.assertEqual(report["cross_split_checks"]["duplicate_question_count"], 1)
        self.assertFalse(report["structural_preflight_passed"])

    def test_section_overlap_is_reported_for_human_review_not_silently_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = list(self.make_inputs(root))
            self.write_jsonl(paths[2], [qa_record("Train question?", "10")])
            self.write_jsonl(paths[3], [qa_record("Validation question?", "10")])
            report = run_preflight(*paths)
        self.assertTrue(report["shared_section_overlap_requires_review"])
        self.assertEqual(report["cross_split_checks"]["shared_section_count"], 1)

    def test_malformed_page_jsonl_fails_structural_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = list(self.make_inputs(Path(directory)))
            paths[0].write_text('{"broken":\n', encoding="utf-8")
            report = run_preflight(*paths)
        self.assertFalse(report["structural_preflight_passed"])
        page_report = report["inputs"]["source_pages_jsonl"]["audit"]
        self.assertEqual(page_report["json_parse_error_count"], 1)
        self.assertFalse(page_report["structural_page_integrity_passed"])

    def test_source_files_are_fingerprinted_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_inputs(Path(directory))
            before = [path.read_bytes() for path in paths]
            run_preflight(*paths)
            after = [path.read_bytes() for path in paths]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
