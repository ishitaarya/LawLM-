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

    seen_records: set[tuple[str, str]] = set()
    seen_sources: dict[str, str] = {}
    unique_questions: set[str] = set()

    for split, rows in splits.items():
        for row in rows:
            required = (
                "question",
                "context",
                "answer",
                "section",
                "act_id",
                "source_id",
            )
            assert all(str(row.get(key, "")).strip() for key in required)

            # The same template question may legitimately occur for different
            # Acts/sections. A duplicate is only invalid when both the source
            # and question are identical.
            record_key = (row["source_id"], row["question"])
            assert record_key not in seen_records, (
                f"Duplicate QA record: {record_key}"
            )
            seen_records.add(record_key)
            unique_questions.add(row["question"])

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
    print(f"Unique question templates: {len(unique_questions):,}")
    print(f"Unique QA records: {len(seen_records):,}")
    print("Required fields: passed")
    print("Duplicate QA records: passed")
    print("Source split isolation: passed")
    print("Dataset size cap: passed")
    print("QA dataset validation passed!")
    print("=" * 72)


if __name__ == "__main__":
    main()
