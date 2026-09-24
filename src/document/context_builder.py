"""
LawSuit LLM — Phase 6B Context Builder

Builds a clean, traceable legal context from retrieved chunks.
No pretrained model or external LLM is used.
"""

from __future__ import annotations

from typing import Any


def build_context(
    query: str,
    results: list[dict[str, Any]],
    max_chunks: int = 3,
) -> str:
    """Build structured context for the downstream LawSuit LLM."""
    if not query or not query.strip():
        raise ValueError("query must not be empty.")
    if max_chunks < 1:
        raise ValueError("max_chunks must be at least 1.")

    selected_results = results[:max_chunks]

    lines = [
        "LEGAL QUESTION",
        query.strip(),
        "",
        "RETRIEVED LEGAL CONTEXT",
    ]

    if not selected_results:
        lines.extend(["", "No relevant legal context was retrieved."])
    else:
        for index, result in enumerate(selected_results, start=1):
            section = result.get("section_number") or "N/A"
            title = result.get("section_title") or "Untitled section"
            page = result.get("page_number") or "N/A"
            text = " ".join(str(result.get("text", "")).split())

            lines.extend(
                [
                    "",
                    f"[Source {index}]",
                    f"Section: {section}",
                    f"Title: {title}",
                    f"Page: {page}",
                    f"Text: {text}",
                ]
            )

    lines.extend(
        [
            "",
            "INSTRUCTION",
            "Answer using the retrieved legal context.",
            "Do not invent legal provisions that are not present in the context.",
        ]
    )

    return "\n".join(lines)


def print_context(context: str) -> None:
    """Print a generated context for manual inspection."""
    print("=" * 72)
    print("LawSuit LLM — Phase 6B Context Builder")
    print("=" * 72)
    print(context)
    print("=" * 72)


if __name__ == "__main__":
    example = build_context(
        "When is an agreement without consideration valid?",
        [
            {
                "section_number": "25",
                "section_title": "Agreement without consideration",
                "page_number": 17,
                "text": "25. Agreement without consideration, void, unless it is in writing and registered.",
            }
        ],
    )
    print_context(example)
