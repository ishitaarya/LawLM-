"""Phase 7.3 evaluation of retrieval grounding for the legal QA pipeline.

This benchmark measures whether the retriever returns the expected statutory
sections for a small, explicitly labeled legal-question set. It does not judge
the legal correctness of free-form generated answers.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.document.retriever import build_index, load_chunks, search

CHUNKS = Path("data/documents/Indian_Contract_Act_1872_chunks.jsonl")
OUTPUT = Path("logs/grounding_retrieval_evaluation.json")

CASES = [
    {
        "id": "valid_contract",
        "question": "What are the essentials of a valid contract?",
        "expected_sections": {"10"},
    },
    {
        "id": "lawful_consideration",
        "question": "What makes consideration and object lawful?",
        "expected_sections": {"23"},
    },
    {
        "id": "without_consideration",
        "question": "When is an agreement without consideration valid?",
        "expected_sections": {"25"},
    },
    {
        "id": "free_consent",
        "question": "What is free consent?",
        "expected_sections": {"13", "14"},
    },
]


def main() -> None:
    if not CHUNKS.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS}")

    chunks = load_chunks(CHUNKS)
    idf, vectors = build_index(chunks)

    results = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    print("=" * 72)
    print("LawSuit LLM - Phase 7.3 Grounding / Retrieval Evaluation")
    print("=" * 72)

    for case in CASES:
        ranked = search(case["question"], chunks, idf, vectors, top_k=5)
        section_numbers = [str(item.get("section_number")) for item in ranked]
        expected = set(case["expected_sections"])

        top1 = bool(section_numbers[:1] and set(section_numbers[:1]) & expected)
        top3 = bool(set(section_numbers[:3]) & expected)
        top5 = bool(set(section_numbers[:5]) & expected)

        top1_hits += int(top1)
        top3_hits += int(top3)
        top5_hits += int(top5)

        item = {
            "id": case["id"],
            "question": case["question"],
            "expected_sections": sorted(expected),
            "retrieved_sections_top5": section_numbers,
            "top1_hit": top1,
            "top3_hit": top3,
            "top5_hit": top5,
        }
        results.append(item)

        print(f"\n[{case['id']}]")
        print(f"Question: {case['question']}")
        print(f"Expected: {sorted(expected)}")
        print(f"Top-5:    {section_numbers}")
        print(f"Hits: top1={top1} | top3={top3} | top5={top5}")

    total = len(CASES)
    report = {
        "phase": "7.3",
        "dataset": str(CHUNKS),
        "cases": total,
        "top1_accuracy": top1_hits / total,
        "top3_accuracy": top3_hits / total,
        "top5_accuracy": top5_hits / total,
        "results": results,
        "note": (
            "This is a small targeted retrieval benchmark over Indian Contract "
            "Act sections. It is not a general legal QA accuracy estimate and "
            "does not certify generated legal advice."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print(f"Top-1 accuracy: {top1_hits}/{total} ({top1_hits / total:.1%})")
    print(f"Top-3 accuracy: {top3_hits}/{total} ({top3_hits / total:.1%})")
    print(f"Top-5 accuracy: {top5_hits}/{total} ({top5_hits / total:.1%})")
    print(f"Report: {OUTPUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
