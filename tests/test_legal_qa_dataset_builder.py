"""Tests for safe construction of human-reviewed legal QA splits."""
import tempfile
import unittest
from pathlib import Path

from src.document.legal_review import (
    approve_legal_content_record,
    approve_review_record,
    build_review_manifest,
    section_text_hash,
)
from src.document.legal_chunker import create_legal_chunks
from src.training.legal_qa_dataset_builder import (
    build_legal_qa_splits,
    qa_review_hash,
    write_split_files_new,
)


class LegalQADatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.lines = [
            {"page_number": 12, "line_number": 3, "text": "A valid agreement is enforceable by law.", "raw_line": "3 A valid agreement is enforceable by law."},
            {"page_number": 12, "line_number": 4, "text": "This is the second sentence of the provision.", "raw_line": "4 This is the second sentence of the provision."},
        ]
        self.sections = [{
            "section_number": "10",
            "section_title": "What agreements are contracts",
            "chapter": "CHAPTER II",
            "page_number": 12,
            "end_page_number": 12,
            "source_pages": [12],
            "source_spans": self.lines,
            "text": "\n".join(line["text"] for line in self.lines),
            "section_status": "in_force",
            "review_required": False,
            "heading_source": {"page_number": 12, "line_number": 2, "raw_line": "10. What agreements are contracts"},
            "amendments": [],
            "footnotes": [],
            "ocr_anomalies": [],
        }]
        record = build_review_manifest(self.sections)[0]
        digest = section_text_hash(self.sections[0]["text"])
        record = approve_review_record(
            record, reviewer="Extraction reviewer", reviewed_at="2026-11-12T12:00:00Z",
            reviewed_text_sha256=digest,
        )
        self.manifest = [approve_legal_content_record(
            record, reviewer="Legal reviewer", reviewed_at="2026-11-12T13:00:00Z",
            reviewed_text_sha256=digest, authority_reference="Official Gazette, Section 10",
            resolution_notes="Compared with the authoritative publication.",
            resolved_review_reasons=record["review_reasons"],
        )]
        self.chunks = create_legal_chunks(
            self.sections, chunk_size=200, overlap=0, review_manifest=self.manifest
        )

    def draft(self, question="When is an agreement a contract?"):
        item = {
            "question": question,
            "answer": "An agreement becomes a contract when it is enforceable by law.",
            "evidence_quote": "A valid agreement is enforceable by law.",
            "chunk_id": self.chunks[0]["chunk_id"],
            "qa_review_status": "qa_review_approved",
            "qa_reviewed_by": "QA reviewer",
            "qa_reviewed_at": "2026-11-12T14:00:00Z",
        }
        item["qa_reviewed_content_sha256"] = qa_review_hash(item)
        return item

    def empty_splits(self):
        return {"train": [], "validation": [], "test": []}

    def test_builds_citation_preserving_reviewed_record(self):
        drafts = self.empty_splits()
        drafts["train"] = [self.draft()]
        result = build_legal_qa_splits(self.sections, self.chunks, self.manifest, drafts)
        record = result["splits"]["train"][0]
        self.assertEqual(record["section"], "10")
        self.assertEqual(record["context"], self.chunks[0]["text"])
        self.assertEqual(record["source_pages"], [12])
        self.assertEqual(record["qa_reviewed_by"], "QA reviewer")
        self.assertFalse(result["ready_for_training"])

    def test_unreviewed_or_modified_qa_draft_is_rejected(self):
        drafts = self.empty_splits()
        item = self.draft()
        item["answer"] = "A changed answer."
        drafts["train"] = [item]
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            build_legal_qa_splits(self.sections, self.chunks, self.manifest, drafts)

    def test_evidence_must_be_an_exact_source_excerpt(self):
        drafts = self.empty_splits()
        item = self.draft()
        item["evidence_quote"] = "This claim is not in the Act."
        item["qa_reviewed_content_sha256"] = qa_review_hash(item)
        drafts["train"] = [item]
        with self.assertRaisesRegex(ValueError, "exact excerpt"):
            build_legal_qa_splits(self.sections, self.chunks, self.manifest, drafts)

    def test_same_section_cannot_leak_across_splits(self):
        drafts = self.empty_splits()
        drafts["train"] = [self.draft()]
        drafts["test"] = [self.draft("Explain the rule in Section 10.")]
        with self.assertRaisesRegex(ValueError, "appears in both"):
            build_legal_qa_splits(self.sections, self.chunks, self.manifest, drafts)

    def test_duplicate_question_across_splits_is_rejected(self):
        drafts = self.empty_splits()
        drafts["train"] = [self.draft()]
        drafts["validation"] = [self.draft()]
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            build_legal_qa_splits(self.sections, self.chunks, self.manifest, drafts)

    def test_only_new_output_files_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "qa"
            written = write_split_files_new(self.empty_splits(), output_dir)
            self.assertTrue(Path(written["train"]).is_file())
            with self.assertRaises(FileExistsError):
                write_split_files_new(self.empty_splits(), output_dir)


if __name__ == "__main__":
    unittest.main()
