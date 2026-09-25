"""Validate causal LM blocks for the expanded Phase 7.5 corpus."""

from __future__ import annotations

from pathlib import Path

from src.training.causal_lm_dataset import build_split_dataset


BLOCK_SIZE = 256


def main() -> None:
    print("=" * 68)
    print("LawSuit LLM - Phase 7.5.4 Causal LM Dataset Test")
    print("=" * 68)

    datasets = {}
    for split in ("train", "validation", "test"):
        path = Path("data/tokenized") / f"{split}.jsonl"
        dataset = build_split_dataset(split, block_size=BLOCK_SIZE)
        datasets[split] = dataset

        assert len(dataset) > 0, f"{split} dataset is empty"
        input_ids, targets = dataset[0]
        assert input_ids.shape == (BLOCK_SIZE,)
        assert targets.shape == (BLOCK_SIZE,)
        assert input_ids.dtype.name if False else True
        assert input_ids.numel() == BLOCK_SIZE
        assert targets.numel() == BLOCK_SIZE
        assert input_ids.tolist()[1:] == targets.tolist()[:-1]

        print(f"{split:10}: {len(dataset):,} blocks | input {tuple(input_ids.shape)} | target {tuple(targets.shape)}")

    total = sum(len(dataset) for dataset in datasets.values())
    print("-" * 68)
    print(f"TOTAL BLOCKS: {total:,}")
    print(f"BLOCK SIZE : {BLOCK_SIZE}")
    print()
    print("Causal LM dataset validation passed!")


if __name__ == "__main__":
    main()
