"""Create immutable JSONL records from LawSuit LLM inference runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_run_record(
    query_id: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    abstained: bool,
) -> dict[str, Any]:
    """Build the evaluator input while retaining the user-visible response."""
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError("query_id must be a non-empty string.")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string.")
    if not isinstance(retrieved_chunks, list) or any(
        not isinstance(chunk, dict) for chunk in retrieved_chunks
    ):
        raise ValueError("retrieved_chunks must be a list of objects.")
    if not isinstance(abstained, bool):
        raise ValueError("abstained must be boolean.")

    # The evaluator expects no substantive answer for an abstention. Keep the
    # exact refusal shown to the user separately for auditability.
    return {
        "query_id": query_id.strip(),
        "answer": "" if abstained else answer.strip(),
        "displayed_response": answer.strip(),
        "retrieved_chunks": retrieved_chunks,
        "abstained": abstained,
    }


def write_run_record_new(record: dict[str, Any], output_path: Path) -> None:
    """Write one record to a new JSONL file; never append or overwrite."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
