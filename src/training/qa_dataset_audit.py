"""Read-only quality audit for supervised legal QA JSONL records.

Expected canonical fields: question, context, answer, and section. This tool
never edits, filters, or rewrites the source dataset. Its report is an audit
aid, not proof of legal correctness or training readiness.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("data/qa_v2/train.jsonl")
REQUIRED_FIELDS = ("question", "context", "answer", "section")


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _looks_sentence_complete(value: str) -> bool:
    return bool(re.search(r"""[.!?]["')\]]?\s*$""", value.strip()))


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return record-level quality signals without changing any record."""
    issues: list[dict[str, Any]] = []
    question_locations: dict[str, list[int]] = defaultdict(list)
    hard_truncated: list[int] = []
    review_required: set[int] = set()
    structurally_valid: set[int] = set()

    for index, record in enumerate(records, start=1):
        record_issues: list[str] = []
        if not isinstance(record, dict):
            issues.append({"record": index, "issues": ["record_not_object"]})
            review_required.add(index)
            continue

        for field in REQUIRED_FIELDS:
            if field not in record:
                record_issues.append(f"missing_{field}")
            elif not isinstance(record[field], str):
                record_issues.append(f"non_string_{field}")
            elif not record[field].strip():
                record_issues.append(f"empty_{field}")

        question = record.get("question")
        context = record.get("context")
        answer = record.get("answer")
        if isinstance(question, str) and question.strip():
            question_locations[_normalize_question(question)].append(index)
            if len(question.strip()) < 10:
                record_issues.append("question_suspiciously_short")
        if isinstance(context, str) and context.strip() and len(context.strip()) < 80:
            record_issues.append("context_suspiciously_short")
        if isinstance(answer, str) and answer.strip():
            if len(answer.strip()) < 20:
                record_issues.append("answer_suspiciously_short")
            if not _looks_sentence_complete(answer):
                record_issues.append("answer_may_be_sentence_truncated")

        truncation_flags = (
            record.get("hard_truncated") is True
            or record.get("token_truncated") is True
            or str(record.get("truncation_type", "")).casefold()
            in {"hard", "hard_token", "token_limit"}
        )
        if truncation_flags:
            hard_truncated.append(index)
            record_issues.append("hard_truncation_flagged")

        if record_issues:
            review_required.add(index)
            issues.append({"record": index, "issues": sorted(set(record_issues))})
        else:
            structurally_valid.add(index)

    duplicate_questions = []
    for question, indices in question_locations.items():
        if len(indices) > 1:
            duplicate_questions.append({
                "normalized_question": question,
                "record_indices": indices,
            })
            review_required.update(indices)

    return {
        "report_type": "legal_qa_dataset_quality_audit",
        "record_count": len(records),
        "structurally_valid_record_count": len(structurally_valid),
        "records_requiring_review_count": len(review_required),
        "hard_truncated_record_indices": hard_truncated,
        "duplicate_questions": duplicate_questions,
        "issues": issues,
        "legal_accuracy_verified": False,
        "ready_for_training": False,
        "note": (
            "This read-only audit detects basic schema and text-quality signals. "
            "It does not verify legal accuracy, source authority, answer completeness, "
            "or whether records are safe for training."
        ),
    }


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"QA JSONL not found: {path}")
    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                parse_errors.append({
                    "line": line_number,
                    "error": str(exc),
                })
    return records, parse_errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records, parse_errors = load_jsonl(args.source)
    report = audit_records(records)
    report["json_parse_error_count"] = len(parse_errors)
    report["json_parse_errors"] = parse_errors
    rendered = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output is None:
        print(rendered)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(f"QA audit report created: {args.output}")


if __name__ == "__main__":
    main()
