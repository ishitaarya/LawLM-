"""
LawSuit LLM — Phase 6C End-to-End Pipeline Test
"""

from src.document.pipeline import answer_question


def main() -> None:
    query = "When is an agreement without consideration valid?"

    answer, results, context = answer_question(
        query,
        top_k=3,
        max_new_tokens=30,
        temperature=0.7,
    )

    assert results
    assert str(results[0].get("section_number")) == "25"
    assert "LEGAL QUESTION" in context
    assert "RETRIEVED LEGAL CONTEXT" in context
    assert answer.strip()

    print("=" * 72)
    print("Phase 6C Pipeline Test")
    print("=" * 72)
    print("Retrieval: PASS")
    print(f"Top section: {results[0].get('section_number')}")
    print("Context builder: PASS")
    print("LawSuit LLM generation: PASS")
    print(f"Generated answer: {answer}")
    print("=" * 72)


if __name__ == "__main__":
    main()
