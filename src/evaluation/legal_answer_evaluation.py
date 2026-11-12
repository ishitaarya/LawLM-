"""Read-only evaluation of generated legal answers against human-reviewed cases.

This evaluator measures evidence and citation signals, not legal semantics.
Every gold case is human-authored, approved, and hash-bound. Model answers
must be supplied as a separate JSONL run file; no model or training is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

GOLD_FIELDS = (
    "query_id", "question", "expected_answer", "expected_sections",
    "evidence_quote", "answerable",
)
RUN_FIELDS = ("query_id", "answer", "retrieved_chunks", "abstained")


def gold_review_hash(record: dict[str, Any]) -> str:
    """Hash the exact gold benchmark content and answerability decision."""
    payload = {field: record.get(field) for field in GOLD_FIELDS}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _has_citation(chunk: dict[str, Any]) -> bool:
    spans = chunk.get("source_spans")
    if isinstance(spans, list) and any(
        isinstance(span, dict) and span.get("page_number") is not None
        for span in spans
    ):
        return True
    return chunk.get("page_number") is not None


def evaluate_answers(
    gold_cases: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare model outputs with approved gold cases using auditable signals."""
    if not gold_cases:
        raise ValueError("Gold benchmark must contain at least one case.")
    gold_by_id: dict[str, dict[str, Any]] = {}
    for index, gold in enumerate(gold_cases, start=1):
        if not isinstance(gold, dict):
            raise ValueError(f"Gold case {index} must be a JSON object.")
        for field in GOLD_FIELDS:
            if field not in gold:
                raise ValueError(f"Gold case {index} is missing {field}.")
        query_id = str(gold["query_id"]).strip()
        if not query_id or query_id in gold_by_id:
            raise ValueError("Gold query_id values must be non-empty and unique.")
        if not isinstance(gold["expected_sections"], list) or not gold["expected_sections"]:
            raise ValueError(f"Gold case {index} expected_sections must be a non-empty list.")
        if not isinstance(gold["answerable"], bool):
            raise ValueError(f"Gold case {index} answerable must be boolean.")
        for field in ("question", "expected_answer", "evidence_quote"):
            if not isinstance(gold[field], str) or not gold[field].strip():
                raise ValueError(f"Gold case {index} has empty {field}.")
        if gold.get("review_status") != "answer_benchmark_approved":
            raise ValueError(f"Gold case {index} is not human-approved.")
        if not str(gold.get("reviewed_by", "")).strip():
            raise ValueError(f"Gold case {index} is missing reviewer identity.")
        if not str(gold.get("reviewed_at", "")).strip():
            raise ValueError(f"Gold case {index} is missing review timestamp.")
        if gold.get("reviewed_content_sha256") != gold_review_hash(gold):
            raise ValueError(f"Gold case {index} review hash does not match its content.")
        gold_by_id[query_id] = gold

    run_by_id: dict[str, dict[str, Any]] = {}
    for index, run in enumerate(run_records, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"Run record {index} must be a JSON object.")
        for field in RUN_FIELDS:
            if field not in run:
                raise ValueError(f"Run record {index} is missing {field}.")
        query_id = str(run["query_id"]).strip()
        if not query_id or query_id in run_by_id:
            raise ValueError("Run query_id values must be non-empty and unique.")
        if not isinstance(run["retrieved_chunks"], list):
            raise ValueError(f"Run record {index} retrieved_chunks must be a list.")
        if not isinstance(run["answer"], str) or not isinstance(run["abstained"], bool):
            raise ValueError(f"Run record {index} has invalid answer/abstained types.")
        run_by_id[query_id] = run

    unknown_ids = sorted(set(run_by_id) - set(gold_by_id))
    if unknown_ids:
        raise ValueError(f"Run contains query_id values absent from gold benchmark: {unknown_ids}")

    outcomes = []
    counters = {
        "answer_present": 0,
        "expected_section_retrieved": 0,
        "evidence_quote_retrieved": 0,
        "citation_metadata_present": 0,
        "correct_abstention_behavior": 0,
        "unsupported_answer_risk": 0,
    }

    for query_id, gold in gold_by_id.items():
        run = run_by_id.get(query_id)
        if run is None:
            outcomes.append({"query_id": query_id, "error": "missing_run_record"})
            continue
        answer = run["answer"].strip()
        chunks = run["retrieved_chunks"]
        if any(not isinstance(chunk, dict) for chunk in chunks):
            raise ValueError(f"Run {query_id} contains a non-object retrieved chunk.")

        expected = {str(value).strip().upper() for value in gold["expected_sections"]}
        retrieved_sections = {
            str(chunk.get("section_number", "")).strip().upper()
            for chunk in chunks if str(chunk.get("section_number", "")).strip()
        }
        section_hit = bool(expected & retrieved_sections)
        quote = gold["evidence_quote"]
        quote_found = any(quote in str(chunk.get("text", "")) for chunk in chunks)
        citation_present = any(_has_citation(chunk) for chunk in chunks)
        answer_present = bool(answer)
        abstention_correct = (
            bool(run["abstained"]) and not answer_present
            if not gold["answerable"]
            else answer_present and not bool(run["abstained"])
        )
        unsupported_risk = answer_present and not quote_found

        checks = {
            "answer_present": answer_present,
            "expected_section_retrieved": section_hit,
            "evidence_quote_retrieved": quote_found,
            "citation_metadata_present": citation_present,
            "correct_abstention_behavior": abstention_correct,
            "unsupported_answer_risk": unsupported_risk,
        }
        for key, passed in checks.items():
            if passed:
                counters[key] += 1

        outcomes.append({
            "query_id": query_id,
            "answer": answer,
            "expected_answer_for_human_comparison": gold["expected_answer"],
            "answerable": gold["answerable"],
            "retrieved_sections": sorted(retrieved_sections),
            "checks": checks,
            "human_semantic_review_required": True,
        })

    evaluated = len([item for item in outcomes if "error" not in item])
    denominator = evaluated or 1
    return {
        "report_type": "human_reviewed_legal_answer_evaluation",
        "gold_case_count": len(gold_cases),
        "run_record_count": len(run_records),
        "evaluated_case_count": evaluated,
        "missing_run_count": len(gold_cases) - evaluated,
        "metrics": {
            key: counters[key] / denominator
            for key in counters
        },
        "outcomes": outcomes,
        "human_semantic_review_required": True,
        "legal_answer_accuracy_verified": False,
        "ready_for_training": False,
        "note": (
            "These are structural evidence, retrieval, citation-metadata, and abstention "
            "signals only. Exact evidence presence does not prove that every generated "
            "claim is supported. A qualified human must compare each answer with the "
            "approved expected answer and authoritative law before claiming correctness."
        ),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_answers(_load_jsonl(args.gold), _load_jsonl(args.run))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is None:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(f"Legal answer evaluation report created: {args.output}")


if __name__ == "__main__":
    main()
