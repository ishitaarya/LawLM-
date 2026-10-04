"""
LawSuit LLM — Phase 7.5 QA dataset utilities.

Creates supervised question/context/answer examples from local legal
documents. No pretrained model or external LLM is used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalQAExample:
    question: str
    context: str
    answer: str
    section: str


def build_section25_examples() -> list[LegalQAExample]:
    """Small seed set for validating the QA-training pipeline."""
    context = (
        "Section 25 — Agreement without consideration, void, unless it is "
        "in writing and registered, or is a promise to compensate for "
        "something done, or is a promise to pay a debt barred by limitation law. "
        "An agreement made without consideration is void, unless it is expressed "
        "in writing and registered under the law for the time being in force for "
        "the registration of documents, and is made on account of natural love "
        "and affection between parties standing in a near relation to each other."
    )

    return [
        LegalQAExample(
            question="When is an agreement without consideration valid?",
            context=context,
            answer=(
                "An agreement without consideration is generally void, subject "
                "to the exceptions stated in Section 25."
            ),
            section="25",
        ),
        LegalQAExample(
            question="What section deals with agreements without consideration?",
            context=context,
            answer="Section 25 deals with agreements without consideration.",
            section="25",
        ),
        LegalQAExample(
            question="What is the general rule for an agreement without consideration?",
            context=context,
            answer="The general rule is that an agreement without consideration is void.",
            section="25",
        ),
    ]
