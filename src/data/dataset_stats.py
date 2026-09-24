"""Phase 2: corpus statistics."""

import json
from pathlib import Path

FILES = {
    "train": Path("data/splits/train.jsonl"),
    "validation": Path("data/splits/validation.jsonl"),
    "test": Path("data/splits/test.jsonl"),
}


def analyze(path: Path) -> dict:
    count = chars = words = 0
    min_chars = None
    max_chars = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            length = len(text)
            count += 1
            chars += length
            words += len(text.split())
            min_chars = length if min_chars is None else min(min_chars, length)
            max_chars = max(max_chars, length)

    return {
        "records": count,
        "characters": chars,
        "words": words,
        "min_chars": min_chars or 0,
        "max_chars": max_chars,
        "avg_chars": round(chars / count, 2) if count else 0,
    }


def main() -> None:
    print("=" * 64)
    print("LawSuit LLM — Phase 2 Corpus Statistics")
    print("=" * 64)

    for name, path in FILES.items():
        if not path.exists():
            print(f"{name}: missing ({path})")
            continue

        data = analyze(path)
        print(f"\n{name.upper()}")
        for key, value in data.items():
            print(f"{key:>14}: {value:,}" if isinstance(value, int) else f"{key:>14}: {value}")


if __name__ == "__main__":
    main()
