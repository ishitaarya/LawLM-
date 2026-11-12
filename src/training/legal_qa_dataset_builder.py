"""Build supervised legal QA splits only from human-reviewed, provenance-checked evidence.

This builder does not invent questions or answers. Draft examples must be
written and approved by a human reviewer, tied to exact question/answer/evidence
text, and linked to a candidate legal chunk. Existing split files are never
overwritten and splits are supplied separately rather than randomly generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.document.legal_review import load_jsonl
from src.training.legal_chunk_provenance_audit import audit_chunk_provenance
from src.training.legal_training_gate import evaluate_training_candidates

SPLITS = ("train", "validation", "test")
DRAFT_FIELDS = ("question", "answer", "evidence_quote", "chunk_id")


def qa_review_hash(record: dict[str, Any]) -> str:
    """Hash the exact reviewed QA payload in a stable field order."""
    payload = {field: record.get(field) for field in DRAFT_FIELDS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalise_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def build_legal_qa_splits(
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    drafts_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Validate reviewed QA drafts and return citation-preserving split records."""
    if set(drafts_by_split) != set(SPLITS):
        raise ValueError("Provide exactly three draft splits: train, validation, and test.")

    gate = evaluate_training_candidates(sections, manifest)
    if gate["candidate_count"] == 0:
        raise ValueError("No sections passed the two-stage legal review candidate gate.")

    candidate_indices = set(gate["candidate_section_indices"])
    section_by_number: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, section in enumerate(sections):
        number = str(section.get("section_number", "")).strip()
        if index in candidate_indices:
            if not number or number in section_by_number:
                raise ValueError("Candidate section identity is missing or duplicated.")
            section_by_number[number] = (index, section)

    candidate_chunks = [
        chunk for chunk in chunks
        if str(chunk.get("section_number", "")).strip() in section_by_number
    ]
    provenance = audit_chunk_provenance(sections, candidate_chunks, manifest)
    if not provenance["provenance_integrity_passed"]:
        raise ValueError(
            "Candidate chunks failed provenance audit; refusing QA dataset construction. "
            f"Issues: {provenance['issues'][:5]}"
        )

    chunk_by_id: dict[str, dict[str, Any]] = {}
    for chunk in candidate_chunks:
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Every candidate chunk must have a non-empty chunk_id.")
        if chunk_id in chunk_by_id:
            raise ValueError(f"Duplicate candidate chunk_id: {chunk_id}")
        chunk_by_id[chunk_id] = chunk

    output: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    question_splits: dict[str, str] = {}
    section_splits: dict[str, str] = {}

    for split in SPLITS:
        for draft_index, draft in enumerate(drafts_by_split[split], start=1):
            if not isinstance(draft, dict):
                raise ValueError(f"{split} draft {draft_index} is not a JSON object.")
            for field in DRAFT_FIELDS:
                if not isinstance(draft.get(field), str) or not draft[field].strip():
                    raise ValueError(f"{split} draft {draft_index} has missing/empty {field}.")

            if draft.get("qa_review_status") != "qa_review_approved":
                raise ValueError(f"{split} draft {draft_index} has not passed human QA review.")
            if not isinstance(draft.get("qa_reviewed_by"), str) or not draft["qa_reviewed_by"].strip():
                raise ValueError(f"{split} draft {draft_index} is missing QA reviewer identity.")
            if not isinstance(draft.get("qa_reviewed_at"), str) or not draft["qa_reviewed_at"].strip():
                raise ValueError(f"{split} draft {draft_index} is missing QA review timestamp.")
            if draft.get("qa_reviewed_content_sha256") != qa_review_hash(draft):
                raise ValueError(f"{split} draft {draft_index} QA review hash does not match its content.")

            question = draft["question"].strip()
            answer = draft["answer"].strip()
            quote = draft["evidence_quote"].strip()
            chunk_id = draft["chunk_id"].strip()
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"{split} draft {draft_index} references a non-candidate or unknown chunk.")
            if quote not in str(chunk.get("text", "")):
                raise ValueError(f"{split} draft {draft_index} evidence_quote is not an exact excerpt of its chunk.")

            section_number = str(chunk.get("section_number", "")).strip()
            section_entry = section_by_number.get(section_number)
            if section_entry is None:
                raise ValueError(f"{split} draft {draft_index} references an unknown candidate section.")
            _, section = section_entry
            if str(chunk.get("section_status", "")).casefold() not in {"in_force", "active"}:
                raise ValueError(f"{split} draft {draft_index} is not linked to an in-force section.")

            normalized = _normalise_question(question)
            if normalized in question_splits:
                raise ValueError(
                    f"Duplicate question appears in {question_splits[normalized]} and {split} splits."
                )
            question_splits[normalized] = split

            previous_split = section_splits.get(section_number)
            if previous_split and previous_split != split:
                raise ValueError(
                    f"Section {section_number} appears in both {previous_split} and {split}; "
                    "keep all examples from one section in a single split."
                )
            section_splits[section_number] = split

            output[split].append({
                "question": question,
                "context": chunk["text"],
                "answer": answer,
                "section": section_number,
                "chunk_id": chunk_id,
                "section_title": chunk.get("section_title", ""),
                "legal_status": chunk.get("section_status"),
                "source_pages": list(chunk.get("source_pages", [])),
                "source_spans": list(chunk.get("source_spans", [])),
                "evidence_quote": quote,
                "legal_authority_reference": chunk.get("legal_authority_reference"),
                "extraction_reviewed_by": chunk.get("reviewed_by"),
                "legal_content_reviewed_by": chunk.get("legal_content_reviewed_by"),
                "qa_reviewed_by": draft["qa_reviewed_by"].strip(),
                "qa_reviewed_at": draft["qa_reviewed_at"].strip(),
                "qa_reviewed_content_sha256": draft["qa_reviewed_content_sha256"],
                "split": split,
            })

    return {
        "report_type": "human_reviewed_legal_qa_dataset_build",
        "split_counts": {split: len(output[split]) for split in SPLITS},
        "total_records": sum(len(output[split]) for split in SPLITS),
        "candidate_section_count": gate["candidate_count"],
        "candidate_chunk_count": len(candidate_chunks),
        "source_provenance_integrity_passed": provenance["provenance_integrity_passed"],
        "splits": output,
        "dataset_exported": False,
        "training_started": False,
        "legal_accuracy_verified": False,
        "ready_for_training": False,
        "note": (
            "Examples and answers are human-authored and human-reviewed; the builder checks "
            "their hashes, evidence excerpts, candidate source, citations, and split isolation. "
            "It does not independently establish legal correctness or authorize training."
        ),
    }


def write_split_files_new(
    splits: dict[str, list[dict[str, Any]]], output_dir: Path
) -> dict[str, str]:
    """Write three new JSONL files without overwriting any existing file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: output_dir / f"{split}.jsonl" for split in SPLITS}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing split files: {existing}")
    for split, path in paths.items():
        with path.open("x", encoding="utf-8") as handle:
            for record in splits[split]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {split: str(path) for split, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--train-drafts", type=Path, required=True)
    parser.add_argument("--validation-drafts", type=Path, required=True)
    parser.add_argument("--test-drafts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = build_legal_qa_splits(
        load_jsonl(args.sections),
        load_jsonl(args.chunks),
        load_jsonl(args.review_manifest),
        {
            "train": load_jsonl(args.train_drafts),
            "validation": load_jsonl(args.validation_drafts),
            "test": load_jsonl(args.test_drafts),
        },
    )
    paths = write_split_files_new(result["splits"], args.output_dir)
    print(json.dumps({
        "split_counts": result["split_counts"],
        "total_records": result["total_records"],
        "outputs": paths,
        "ready_for_training": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
