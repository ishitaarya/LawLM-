"""Validate Phase 7.5F QA V2 sequences and unknown-token usage."""
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
            ids, labels = r["input_ids"], r["labels"]
            assert len(ids) == len(labels) <= MAX_LENGTH
            first = next((j for j, x in enumerate(labels) if x != -100), None)
            assert first is not None and first > 0
            assert all(x == -100 for x in labels[:first])
            assert ids[first:] == [x for x in labels[first:]]
            assert labels[-1] == EOS_ID
            assert sum(x == UNK_ID for x in labels) == 0
        print(f"{split}: {len(rows):,} valid")
    print("QA V2 validation passed.")

if __name__ == "__main__":
    main()
