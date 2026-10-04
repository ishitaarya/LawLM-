"""Tests for the Phase 7.4 grounded legal answer engine."""

from src.document.grounded_answer import (
    build_grounded_answer,
    deduplicate_results,
)


def sample_results():
    return [
        {
            "section_number": "25",
            "section_title": "Agreement without consideration, void",
            "page_number": 17,
            "score": 3.4562,
            "text": (
                "Agreement without consideration, void, unless it is in writing "
                "and registered."
            ),
        },
        {
            "section_number": "25",
            "section_title": "Agreement without consideration, void",
            "page_number": 18,
            "score": 3.3605,
            "text": "Another overlapping chunk from section 25.",
        },
        {
            "section_number": "19",
            "section_title": "Voidability",
            "page_number": 12,
            "score": 1.1,
            "text": "A different provision.",
        },
    ]


def main() -> None:
    results = sample_results()

    unique = deduplicate_results(results)
    assert [item["section_number"] for item in unique] == ["25", "19"]
    assert unique[0]["score"] == 3.4562

    answer = build_grounded_answer(
        "When is an agreement without consideration valid?",
        results,
    )

    assert answer["grounded"] is True
    assert "Section 25" in answer["answer"]
    assert "writing" in answer["answer"]
    assert len(answer["sources"]) == 2

    empty = build_grounded_answer("Unknown question", [])
    assert empty["grounded"] is False

    print("=" * 72)
    print("LawSuit LLM - Phase 7.4 Grounded Answer Tests")
    print("=" * 72)
    print("Duplicate section chunks: passed")
    print("Grounded answer construction: passed")
    print("Empty retrieval handling: passed")
    print("All Phase 7.4 tests passed!")
    print("=" * 72)


if __name__ == "__main__":
    main()
