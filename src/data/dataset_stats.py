import json
from pathlib import Path

FILES = {
    "train": Path("data/splits/train.jsonl"),
    "validation": Path("data/splits/validation.jsonl"),
    "test": Path("data/splits/test.jsonl"),
}


def analyze(path):
    count = 0
    chars = 0
    words = 0
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

    return count, chars, words, min_chars or 0, max_chars


def main():
    print("=" * 60)
    print("LawLM - Dataset Statistics")
    print("=" * 60)

    total = 0
    for name, path in FILES.items():
        if not path.exists():
            print(f"{name}: missing ({path})")
            continue

        count, chars, words, min_chars, max_chars = analyze(path)
        total += count
        print(f"{name.title():<12} documents: {count:,}")
        print(f"{'':<12} characters: {chars:,}")
        print(f"{'':<12} words:      {words:,}")
        print(f"{'':<12} min chars:  {min_chars:,}")
        print(f"{'':<12} max chars:  {max_chars:,}")
        print()

    print(f"Total documents: {total:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
