import json
import random
from pathlib import Path

INPUT_FILE = Path("data/processed/legal_corpus.jsonl")
OUTPUT_DIR = Path("data/splits")

SEED = 42
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Clean dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in INPUT_FILE.open("r", encoding="utf-8") if line.strip()]
    random.Random(SEED).shuffle(rows)

    n = len(rows)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": rows[:train_end],
        "validation": rows[train_end:val_end],
        "test": rows[val_end:],
    }

    for name, data in splits.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in data:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("LawLM - Dataset Split")
    print("=" * 60)
    print(f"Total:      {n}")
    print(f"Train:      {len(splits['train'])}")
    print(f"Validation: {len(splits['validation'])}")
    print(f"Test:       {len(splits['test'])}")
    print(f"Seed:       {SEED}")
    print("=" * 60)


if __name__ == "__main__":
    main()
