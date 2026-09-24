"""
LawSuit LLM — Phase 6B Context Builder Test
"""

from src.document.context_builder import build_context


def main() -> None:
    query = "When is an agreement without consideration valid?"

    results = [
        {
            "chunk_id": "contract_00069",
            "section_number": "25",
            "section_title": "Agreement without consideration",
            "page_number": 17,
            "text": (
                "25. Agreement without consideration, void, unless it is "
                "in writing and registered, or is a promise to compensate "
                "for something done."
            ),
            "score": 2.50,
        },
        {
            "chunk_id": "contract_00070",
            "section_number": "25",
            "section_title": "Agreement without consideration",
            "page_number": 17,
            "text": "An agreement made without consideration is void, unless...",
            "score": 2.20,
        },
    ]

    context = build_context(query, results, max_chunks=2)

    assert "LEGAL QUESTION" in context
    assert query in context
    assert "Section: 25" in context
    assert "Agreement without consideration" in context
    assert "RETRIEVED LEGAL CONTEXT" in context
    assert "Do not invent legal provisions" in context

    print("=" * 72)
    print("Context Builder Test")
    print("=" * 72)
    print("PASS — context was built correctly.")
    print(context)
    print("=" * 72)


if __name__ == "__main__":
    main()
