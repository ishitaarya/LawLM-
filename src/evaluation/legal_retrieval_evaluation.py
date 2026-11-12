"""Evaluate legal retrieval against a human-reviewed JSONL benchmark.

Benchmark record schema:
{"query": "...", "expected_sections": ["10", "14"]}
This evaluates retrieval ranking only. It does not evaluate legal correctness
of generated answers or the legal authority/currentness of the source corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.document.retriever import build_index, load_chunks, search

DEFAULT_CHUNKS = Path("data/processed/Indian_Contract_Act_1872_legal_chunks.jsonl")
DEFAULT_BENCHMARK = Path("data/evaluation/legal_retrieval_benchmark.jsonl")


def evaluate_retrieval(
    benchmark: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    cutoffs: tuple[int, ...] = (1, 3, 5),
    include_repealed: bool = False,
) -> dict[str, Any]:
    if not benchmark:
        raise ValueError("Benchmark must contain at least one query.")
    if not chunks:
        raise ValueError("Legal chunk collection must not be empty.")
    if not cutoffs or any(k < 1 for k in cutoffs):
        raise ValueError("All retrieval cutoffs must be positive integers.")

    idf, vectors = build_index(chunks)
    hit_counts = {k: 0 for k in cutoffs}
    reciprocal_ranks: list[float] = []
    query_results = []
    valid_queries = 0

    for index, case in enumerate(benchmark, start=1):
        if not isinstance(case, dict):
            query_results.append({"case": index, "error": "benchmark_record_not_object"})
            continue
        query = str(case.get("query", "")).strip()
        expected = case.get("expected_sections", [])
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(expected, list):
            expected = []
        expected_set = {str(value).upper() for value in expected if str(value).strip()}
        if not query or not expected_set:
            query_results.append({
                "case": index,
                "error": "query_or_expected_sections_missing",
            })
            continue

        valid_queries += 1
        max_k = max(cutoffs)
        results = search(
            query, chunks, idf, vectors,
            top_k=max_k, include_repealed=include_repealed,
        )
        ranked_sections = [str(item.get("section_number", "")).upper() for item in results]
        first_rank = next(
            (rank for rank, section in enumerate(ranked_sections, start=1)
             if section in expected_set),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

        for cutoff in cutoffs:
            if any(section in expected_set for section in ranked_sections[:cutoff]):
                hit_counts[cutoff] += 1

        query_results.append({
            "case": index,
            "query": query,
            "expected_sections": sorted(expected_set),
            "retrieved_sections": ranked_sections,
            "first_relevant_rank": first_rank,
        })

    if valid_queries == 0:
        raise ValueError("No valid benchmark queries were found.")

    return {
        "report_type": "legal_retrieval_ranking_evaluation",
        "benchmark_query_count": len(benchmark),
        "valid_query_count": valid_queries,
        "invalid_query_count": len(benchmark) - valid_queries,
        "metrics": {
            f"hit_rate_at_{k}": hit_counts[k] / valid_queries
            for k in cutoffs
        } | {
            "mean_reciprocal_rank": sum(reciprocal_ranks) / valid_queries,
        },
        "include_repealed": include_repealed,
        "query_results": query_results,
        "legal_answer_accuracy_verified": False,
        "note": (
            "Metrics measure whether expected section numbers are retrieved. "
            "They do not prove answer correctness, legal currency, or source authority."
        ),
    }


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark JSONL not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid benchmark JSON on line {line_number}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-repealed", action="store_true")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    benchmark = load_benchmark(args.benchmark)
    report = evaluate_retrieval(
        benchmark, chunks, include_repealed=args.include_repealed
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output is None:
        print(rendered)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(f"Retrieval evaluation report created: {args.output}")


if __name__ == "__main__":
    main()
