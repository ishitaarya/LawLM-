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


def _is_index_like(result: dict[str, Any]) -> bool:
    """Identify table-of-contents/index chunks that are not the provision."""
    title = " ".join(str(result.get("section_title") or "").lower().split())
    text = " ".join(str(result.get("text") or "").lower().split())

    return title in {"interpretation", "contents", "table of contents", "index"} or (
        title == "interpretation" and text.count("25.") > 1
    )


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-scoring useful chunk for each section number."""
    best: dict[str, dict[str, Any]] = {}

    for result in results:
        if _is_index_like(result):
            continue

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
    """Normalize whitespace and common PDF extraction spacing artifacts."""
    text = str(text).replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    replacements = {
        "l i mitation": "limitation",
        "i t": "it",
        "forthe": "for the",
        "cons ideration": "consideration",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def extract_section_facts(result: dict[str, Any]) -> list[str]:
    """Extract meaningful statutory clauses without inventing content."""
    text = clean_text(result.get("text", ""))
    if not text:
        return []

    section = str(result.get("section_number") or "").strip()
    if section:
        text = re.sub(rf"^\s*{re.escape(section)}\.\s*", "", text, count=1)

    # The PDF chunk contains the section heading followed by an em dash and
    # then the actual operative provision.
    if "—" in text:
        text = text.split("—", 1)[1].strip()

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

    facts = extract_section_facts(primary)

    if facts:
        answer = f"Section {section} ({title}) provides: {facts[0]}"
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
