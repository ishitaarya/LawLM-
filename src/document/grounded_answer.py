"""
LawSuit LLM — Phase 7.4 Grounded Legal Answer Engine

Builds a conservative, source-traceable answer from retrieved statutory
context. The from-scratch Transformer remains available for generation, but
this layer avoids presenting unconstrained generated text as a legal answer.

Educational/research use only; not legal advice.
"""

from __future__ import annotations

import re
from typing import Any


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-scoring chunk for each section number."""
    best: dict[str, dict[str, Any]] = {}

    for result in results:
        section = str(result.get("section_number") or "N/A")
        current = best.get(section)
        if current is None or result.get("score", 0.0) > current.get("score", 0.0):
            best[section] = result

    return sorted(
        best.values(),
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )


def clean_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def extract_section_facts(result: dict[str, Any]) -> list[str]:
    """Extract sentence-sized facts without inventing content."""
    text = clean_text(result.get("text", ""))
    if not text:
        return []

    sentences = re.split(r"(?<=[.;])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def build_grounded_answer(
    query: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a conservative answer using only retrieved statutory text."""
    unique_results = deduplicate_results(results)

    if not unique_results:
        return {
            "question": query,
            "answer": "No relevant legal provision was retrieved.",
            "sources": [],
            "grounded": False,
        }

    primary = unique_results[0]
    section = str(primary.get("section_number") or "N/A")
    title = clean_text(primary.get("section_title") or "Untitled section")
    text = clean_text(primary.get("text", ""))

    facts = extract_section_facts(primary)

    if facts:
        answer = (
            f"Based on Section {section} ({title}), the relevant provision states: "
            f"{facts[0]}"
        )
    else:
        answer = (
            f"Section {section} ({title}) was retrieved, but no readable "
            "provision text was available for a grounded summary."
        )

    sources = [
        {
            "section": str(item.get("section_number") or "N/A"),
            "title": clean_text(item.get("section_title") or "Untitled section"),
            "page": item.get("page_number"),
            "score": item.get("score", 0.0),
            "text": clean_text(item.get("text", "")),
        }
        for item in unique_results
    ]

    return {
        "question": query,
        "answer": answer,
        "sources": sources,
        "grounded": True,
    }
