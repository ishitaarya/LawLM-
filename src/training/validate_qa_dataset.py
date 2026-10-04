"""Validate deterministic legal QA dataset integrity and split isolation."""

from __future__ import annotations

import json
from pathlib import Path

QA_DIR = Path("data/qa")


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    splits = {
        name: load(QA_DIR / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }

    seen_questions: set[str] = set()
    seen_sources: dict[str, str] = {}

    for split, rows in splits.items():
        for row in rows:
            required = ("question", "context", "answer", "section", "act_id", "source_id")
            assert all(str(row.get(key, "")).strip() for key in required)
            assert row["question"] not in seen_questions
            seen_questions.add(row["question"])

            source = row["source_id"]
            previous = seen_sources.get(source)
            assert previous in (None, split), (
                f"Source leakage: {source} appears in {previous} and {split}"
            )
            seen_sources[source] = split

    total = sum(len(rows) for rows in splits.values())
    assert total > 0
    assert total <= 20_000

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5A QA Dataset Validation")
    print("=" * 72)
    for split, rows in splits.items():
        print(f"{split:>12}: {len(rows):,}")
    print(f"{'TOTAL':>12}: {total:,}")
    print(f"Unique questions: {len(seen_questions):,}")
    print("Required fields: passed")
    print("Duplicate questions: passed")
    print("Source split isolation: passed")
    print("Dataset size cap: passed")
    print("QA dataset validation passed!")
    print("=" * 72)


if __name__ == "__main__":
    main()
