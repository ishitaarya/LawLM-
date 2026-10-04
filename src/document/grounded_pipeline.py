"""
LawSuit LLM — Phase 7.4 Grounded Document Pipeline

Question -> retrieval -> section deduplication -> source-grounded answer.

The Transformer is not used to fabricate a legal answer in this mode.
It remains part of the research system and can be evaluated separately.

Educational/research use only; not legal advice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.document.grounded_answer import build_grounded_answer
from src.document.retriever import build_index, load_chunks, search

DEFAULT_CHUNKS = Path("data/documents/Indian_Contract_Act_1872_chunks.jsonl")
DEFAULT_OUTPUT = Path("logs/grounded_answers.jsonl")


def answer_question(query: str, top_k: int = 5) -> dict:
    chunks = load_chunks(DEFAULT_CHUNKS)
    if not chunks:
        raise RuntimeError("No legal chunks found.")

    idf, vectors = build_index(chunks)
    retrieved = search(query, chunks, idf, vectors, top_k=top_k)

    answer = build_grounded_answer(query, retrieved)
    answer["retrieved_count"] = len(retrieved)
    answer["unique_source_count"] = len(answer["sources"])
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grounded legal document answering."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="When is an agreement without consideration valid?",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    if not DEFAULT_CHUNKS.exists():
        raise FileNotFoundError(f"Chunk file not found: {DEFAULT_CHUNKS}")

    result = answer_question(args.query, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("=" * 72)
    print("LawSuit LLM - Phase 7.4 Grounded Legal Answer")
    print("=" * 72)
    print(f"Question: {result['question']}")
    print(f"Grounded: {result['grounded']}")
    print(f"Retrieved chunks: {result['retrieved_count']}")
    print(f"Unique source sections: {result['unique_source_count']}")
    print()
    print("Answer:")
    print(result["answer"])
    print()
    print("Sources:")
    for source in result["sources"]:
        print(
            f"  Section {source['section']} | page {source['page']} "
            f"| score {source['score']:.4f}"
        )
        print(f"  Title: {source['title']}")
        print(f"  Text: {source['text'][:500]}")
        print()
    print("Educational/research output only; not legal advice.")
    print(f"Saved: {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
