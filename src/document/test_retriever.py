"""
LawSuit LLM — Phase 6B Retrieval Evaluation

Evaluates retrieval against a small manually defined benchmark.
This is a retrieval test, not a legal-answer evaluation.
"""

from pathlib import Path

from src.document.retriever import (
    DEFAULT_INPUT,
    build_index,
    load_chunks,
    search,
)


TEST_CASES = [
    {
        "query": "What are the essentials of a valid contract?",
        "expected_sections": {"10"},
    },
    {
        "query": "What makes an agreement a contract?",
        "expected_sections": {"10"},
    },
    {
        "query": "What considerations and objects are lawful?",
        "expected_sections": {"23"},
    },
    {
        "query": "When is an agreement without consideration valid?",
        "expected_sections": {"25"},
    },
    {
        "query": "What is free consent?",
        "expected_sections": {"13", "14"},
    },
]


def evaluate_case(case: dict, chunks: list[dict], idf: dict, vectors: list[dict]):
    results = search(
        case["query"],
        chunks,
        idf,
        vectors,
        top_k=5,
    )

    retrieved_sections = [
        str(result.get("section_number"))
        for result in results
        if result.get("section_number") is not None
    ]

    expected = case["expected_sections"]

    top1_hit = bool(
        retrieved_sections
        and retrieved_sections[0] in expected
    )

    top3_hit = any(
        section in expected
        for section in retrieved_sections[:3]
    )

    top5_hit = any(
        section in expected
        for section in retrieved_sections[:5]
    )

    return {
        "query": case["query"],
        "expected": expected,
        "retrieved": retrieved_sections,
        "top1": top1_hit,
        "top3": top3_hit,
        "top5": top5_hit,
    }


def main() -> None:
    if not Path(DEFAULT_INPUT).exists():
        raise FileNotFoundError(
            f"Chunk file not found: {DEFAULT_INPUT}"
        )

    chunks = load_chunks(DEFAULT_INPUT)

    if not chunks:
        raise RuntimeError("No chunks found.")

    idf, vectors = build_index(chunks)

    results = [
        evaluate_case(case, chunks, idf, vectors)
        for case in TEST_CASES
    ]

    top1 = sum(result["top1"] for result in results)
    top3 = sum(result["top3"] for result in results)
    top5 = sum(result["top5"] for result in results)

    total = len(results)

    print("=" * 72)
    print("LawSuit LLM — Phase 6B Retrieval Evaluation")
    print("=" * 72)
    print(f"Test cases: {total}")
    print()

    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result['query']}")
        print(f"    Expected section(s): {', '.join(sorted(result['expected']))}")
        print(f"    Retrieved sections:  {', '.join(result['retrieved'])}")
        print(f"    Top-1: {'PASS' if result['top1'] else 'FAIL'}")
        print(f"    Top-3: {'PASS' if result['top3'] else 'FAIL'}")
        print(f"    Top-5: {'PASS' if result['top5'] else 'FAIL'}")
        print()

    print("-" * 72)
    print(f"Top-1 accuracy: {top1}/{total} = {top1 / total:.2%}")
    print(f"Top-3 accuracy: {top3}/{total} = {top3 / total:.2%}")
    print(f"Top-5 accuracy: {top5}/{total} = {top5 / total:.2%}")
    print("=" * 72)


if __name__ == "__main__":
    main()
