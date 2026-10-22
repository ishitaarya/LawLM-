"""Validate QA V2 sequence alignment, answer metadata, and unknown-token usage."""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path("data/qa_v2")
EOS_ID = 3
UNK_ID = 1
MAX_LENGTH = 256
TRUNCATION_TYPES = {"complete", "sentence_boundary", "hard_token_cap"}


def main():
    for split in ("train", "validation", "test"):
        path = DATA / f"{split}.jsonl"
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert rows, f"{split} split is empty"

        truncation_counts = {name: 0 for name in sorted(TRUNCATION_TYPES)}
        for i, r in enumerate(rows):
            ids = r["input_ids"]
            labels = r["labels"]
            prompt_len = r["prompt_length"]
            answer_tokens = r["answer_tokens"]

            assert len(ids) == len(labels) <= MAX_LENGTH, (split, i, "length")
            assert 1 <= prompt_len < len(ids), (split, i, "prompt_length")
            assert answer_tokens >= 8, (split, i, "short answer")
            assert len(ids) == prompt_len + answer_tokens + 1, (
                split, i, "answer token count does not match sequence"
            )
            assert len(r.get("attention_mask", [])) == len(ids), (
                split, i, "attention mask length"
            )
            assert all(x == 1 for x in r["attention_mask"]), (
                split, i, "unexpected attention mask value"
            )

            assert all(x == -100 for x in labels[:prompt_len - 1]), (
                split, i, "prompt tokens must be masked"
            )
            # The final prompt token predicts the first answer token.
            assert labels[prompt_len - 1:-1] == ids[prompt_len:], (
                split, i, "causal target alignment"
            )
            assert labels[-1] == -100, (split, i, "final label must be masked")
            assert ids[-1] == EOS_ID, (split, i, "missing EOS")
            assert sum(x == UNK_ID for x in labels) == 0, (
                split, i, "UNK token in supervised targets"
            )

            truncation = r.get("answer_truncation")
            assert truncation in TRUNCATION_TYPES, (
                split, i, "missing or invalid answer_truncation"
            )
            truncation_counts[truncation] += 1
            assert isinstance(r.get("answer"), str) and r["answer"].strip(), (
                split, i, "empty answer text"
            )

        print(
            f"{split}: {len(rows):,} valid; causal alignment passed; "
            f"answer truncation={truncation_counts}"
        )
    print("QA V2 validation passed.")


if __name__ == "__main__":
    main()
