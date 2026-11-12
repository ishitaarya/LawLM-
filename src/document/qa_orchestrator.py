"""Dependency-injected orchestration for the complete legal QA run.

This module is intentionally free of model-framework imports so the retrieval,
abstention, logging, and evaluation hand-off can be integration-tested on CI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.document.context_builder import build_context
from src.evaluation.legal_run_logger import build_run_record, write_run_record_new

RetrieveFunction = Callable[[str, int], list[dict[str, Any]]]
GenerateFunction = Callable[[str, str], str]

NO_EVIDENCE_RESPONSE = (
    "I could not retrieve a relevant provision from the selected legal "
    "document. I cannot provide an answer without supporting source text."
)


def run_legal_qa(
    query: str,
    retrieve: RetrieveFunction,
    generate: GenerateFunction,
    top_k: int = 3,
    query_id: str | None = None,
    evaluation_record_path: Path | None = None,
) -> tuple[str, list[dict[str, Any]], str]:
    """Execute retrieval -> context -> generation/abstention -> optional log."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if (query_id is None) != (evaluation_record_path is None):
        raise ValueError("query_id and evaluation_record_path must be supplied together.")

    retrieved = retrieve(query, top_k)
    if not isinstance(retrieved, list) or any(
        not isinstance(chunk, dict) for chunk in retrieved
    ):
        raise ValueError("retrieve must return a list of chunk objects.")

    context = build_context(query, retrieved, max_chunks=top_k)
    if retrieved:
        answer = generate(query, context)
        if not isinstance(answer, str):
            raise ValueError("generate must return a string.")
    else:
        answer = NO_EVIDENCE_RESPONSE

    abstained = not retrieved or not answer.strip()
    if evaluation_record_path is not None:
        record = build_run_record(query_id, answer, retrieved, abstained)
        write_run_record_new(record, evaluation_record_path)

    return answer, retrieved, context
