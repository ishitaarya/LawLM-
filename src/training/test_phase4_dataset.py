"""Smoke tests for the Phase 4 causal LM dataset."""

from pathlib import Path

from src.training.causal_lm_dataset import CausalLMDataset


def test_dataset_builds_from_encoded_split() -> None:
    dataset = CausalLMDataset(
        Path("data/tokenized/train.jsonl"),
        block_size=256,
        eos_id=3,
    )

    assert len(dataset) > 0

    input_ids, targets = dataset[0]
    assert input_ids.shape == targets.shape
    assert input_ids.shape[0] == 256
    assert input_ids.dtype == targets.dtype


def test_causal_shift_is_correct() -> None:
    dataset = CausalLMDataset(
        Path("data/tokenized/train.jsonl"),
        block_size=16,
        eos_id=3,
    )
    input_ids, targets = dataset[0]

    assert input_ids.shape == targets.shape
    assert targets[0].item() == input_ids[1].item()


if __name__ == "__main__":
    test_dataset_builds_from_encoded_split()
    test_causal_shift_is_correct()
    print("=" * 60)
    print("LawSuit LLM - Phase 4 Dataset Test")
    print("=" * 60)
    print("Dataset: data/tokenized/train.jsonl")
    print("Block size: 256")
    print("Causal shift: OK")
    print("Dataset loading: OK")
    print("Phase 4 dataset test passed!")
    print("=" * 60)
