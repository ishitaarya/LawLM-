"""Validate Phase 7.5C.1 chunked QA data before training."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

QA_DIR = Path("data/qa_chunked")
MAX_LENGTH = 256
EOS_ID = 3


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    splits = {
        name: load(QA_DIR / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }

    total = 0
    seen_sources: dict[str, str] = {}
    chunk_groups: dict[str, list[int]] = defaultdict(list)

    for split, rows in splits.items():
        for index, row in enumerate(rows):
            required = (
                "input_ids",
                "labels",
                "attention_mask",
                "question",
                "answer",
                "section",
                "act_id",
                "source_id",
                "chunk_index",
                "chunk_count",
                "answer_tokens",
            )
            assert all(key in row for key in required), (
                f"Missing field in {split}[{index}]"
            )

            input_ids = row["input_ids"]
            labels = row["labels"]
            attention = row["attention_mask"]

            assert input_ids, f"Empty input_ids in {split}[{index}]"
            assert len(input_ids) <= MAX_LENGTH
            assert len(input_ids) == len(labels) == len(attention)
            assert len(row["answer"].strip()) > 0
            assert row["answer_tokens"] > 0
            assert row["chunk_count"] >= 1
            assert 0 <= row["chunk_index"] < row["chunk_count"]

            assert all(isinstance(token, int) and token >= 0 for token in input_ids)
            assert all(mask in (0, 1) for mask in attention)

            # The answer must contribute to the supervised loss.
            supervised_positions = [
                i for i, label in enumerate(labels) if label != -100
            ]
            assert supervised_positions, f"No supervised answer tokens in {split}[{index}]"

            # EOS should terminate a complete prepared sequence.
            assert input_ids[-1] == EOS_ID
            assert labels[-1] == EOS_ID

            # Prompt tokens are masked; answer tokens are not.
            first_supervised = supervised_positions[0]
            assert all(label == -100 for label in labels[:first_supervised])
            assert all(label != -100 for label in labels[first_supervised:])

            source_id = row["source_id"]
            previous = seen_sources.get(source_id)
            assert previous in (None, split), (
                f"Source leakage: {source_id} appears in {previous} and {split}"
            )
            seen_sources[source_id] = split

            chunk_groups[source_id].append(row["chunk_index"])
            total += 1

    # Every source's chunk indices should form a contiguous 0..N-1 range.
    for source_id, indices in chunk_groups.items():
        ordered = sorted(set(indices))
        assert ordered == list(range(len(ordered))), (
            f"Invalid chunk indices for {source_id}: {ordered}"
        )

    assert total > 0

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5C.2 Chunked QA Validation")
    print("=" * 72)
    for split, rows in splits.items():
        max_len = max((len(row["input_ids"]) for row in rows), default=0)
        supervised = sum(
            sum(label != -100 for label in row["labels"]) for row in rows
        )
        print(
            f"{split:>12}: {len(rows):,} records | "
            f"max sequence {max_len} | "
            f"supervised tokens {supervised:,}"
        )

    print(f"{'TOTAL':>12}: {total:,}")
    print("Required fields: passed")
    print("Sequence length <= 256: passed")
    print("Input/label/mask alignment: passed")
    print("Non-empty answers: passed")
    print("Supervised answer tokens: passed")
    print("EOS termination: passed")
    print("Prompt loss masking: passed")
    print("Chunk index integrity: passed")
    print("Source split isolation: passed")
    print("Chunked QA dataset validation passed!")
    print("=" * 72)


if __name__ == "__main__":
    main()
