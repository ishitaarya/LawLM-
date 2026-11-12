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
            end_page = result.get("end_page_number") or page
            text = " ".join(str(result.get("text", "")).split())
            source_spans = result.get("source_spans") or []
            source_lines = sorted({
                (int(span["page_number"]), int(span["line_number"]))
                for span in source_spans
                if span.get("page_number") and span.get("line_number")
            })
            line_reference = (
                "; ".join(f"p{source_page}:L{line_number}"
                          for source_page, line_number in source_lines)
                if source_lines else "Line-level reference unavailable"
            )
            status = result.get("section_status", "unknown")
            review_required = bool(result.get("review_required", True))
            annotations = (
                f"amendments={bool(result.get('has_amendment_annotations'))}, "
                f"footnotes={bool(result.get('has_footnote_annotations'))}, "
                f"OCR anomalies={bool(result.get('has_ocr_anomalies'))}"
            )

            lines.extend(
                [
                    "",
                    f"[Source {index}]",
                    f"Section: {section}",
                    f"Title: {title}",
                    f"Pages: {page}-{end_page}",
                    f"Source lines: {line_reference}",
                    f"Section status: {status}",
                    f"Review required: {review_required}",
                    f"Annotations: {annotations}",
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
