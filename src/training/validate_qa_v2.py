"""Validate causally aligned QA V2 sequences and unknown-token usage."""
from __future__ import annotations
import json
from pathlib import Path
DATA = Path("data/qa_v2")
EOS_ID = 3
UNK_ID = 1
MAX_LENGTH = 256

def main():
    for split in ("train", "validation", "test"):
        rows = [json.loads(x) for x in (DATA / f"{split}.jsonl").open(encoding="utf-8") if x.strip()]
        assert rows, split
        for i, r in enumerate(rows):
            ids, labels, prompt_len = r["input_ids"], r["labels"], r["prompt_length"]
            assert len(ids) == len(labels) <= MAX_LENGTH
            assert 1 <= prompt_len < len(ids)
            assert all(x == -100 for x in labels[:prompt_len - 1])
            # Last prompt token predicts first answer token; labels are shifted.
            assert labels[prompt_len - 1:-1] == ids[prompt_len:]
            assert labels[-1] == -100
            assert ids[-1] == EOS_ID
            assert sum(x == UNK_ID for x in labels) == 0
        print(f"{split}: {len(rows):,} valid (causal target alignment passed)")
    print("QA V2 validation passed.")

if __name__ == "__main__": main()
