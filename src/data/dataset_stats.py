"""Phase 7.5.2: detailed corpus validation statistics."""

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
    jurisdictions = {}
    acts = set()
    sections = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            length = len(text)
            count += 1
            chars += length
            words += len(text.split())
            min_chars = length if min_chars is None else min(min_chars, length)
            max_chars = max(max_chars, length)

            jurisdiction = str(row.get("state") or "unknown").strip() or "unknown"
            jurisdictions[jurisdiction] = jurisdictions.get(jurisdiction, 0) + 1

            act_id = str(row.get("act_id") or "").strip()
            section = str(row.get("section_number") or "").strip()
            if act_id:
                acts.add(act_id)
            if act_id or section:
                sections.add(f"{act_id}:{section}")

    return {
        "records": count,
        "characters": chars,
        "words": words,
        "estimated_tokens": round(words * 1.33),
        "size_mb_decimal": round(chars / 1_000_000, 2),
        "min_chars": min_chars or 0,
        "max_chars": max_chars,
        "avg_chars": round(chars / count, 2) if count else 0,
        "jurisdictions": jurisdictions,
        "unique_acts": len(acts),
        "unique_sections": len(sections),
    }


def main() -> None:
    print("=" * 72)
    print("LawSuit LLM — Phase 7.5.2 Corpus Validation Statistics")
    print("=" * 72)

    combined = {
        "records": 0, "characters": 0, "words": 0, "estimated_tokens": 0,
        "size_mb_decimal": 0.0, "unique_acts": set(), "unique_sections": set(),
        "jurisdictions": {},
    }

    for name, path in FILES.items():
        if not path.exists():
            print(f"{name}: missing ({path})")
            continue

        data = analyze(path)
        combined["records"] += data["records"]
        combined["characters"] += data["characters"]
        combined["words"] += data["words"]
        combined["estimated_tokens"] += data["estimated_tokens"]
        combined["unique_acts"].add(data["unique_acts"])
        combined["unique_sections"].add(data["unique_sections"])
        for jurisdiction, count in data["jurisdictions"].items():
            combined["jurisdictions"][jurisdiction] = (
                combined["jurisdictions"].get(jurisdiction, 0) + count
            )

        print(f"\n{name.upper()}")
        for key in (
            "records", "characters", "words", "estimated_tokens",
            "size_mb_decimal", "min_chars", "max_chars", "avg_chars",
            "unique_acts", "unique_sections",
        ):
            print(f"{key:>20}: {data[key]:,}" if isinstance(data[key], int) else f"{key:>20}: {data[key]}")
        print(f"{'jurisdictions':>20}: {data['jurisdictions']}")

    print("\nTOTAL")
    print(f"{'records':>20}: {combined['records']:,}")
    print(f"{'characters':>20}: {combined['characters']:,}")
    print(f"{'words':>20}: {combined['words']:,}")
    print(f"{'estimated_tokens':>20}: {combined['estimated_tokens']:,}")
    print(f"{'size_mb_decimal':>20}: {combined['characters'] / 1_000_000:.2f}")
    print(f"{'jurisdictions':>20}: {combined['jurisdictions']}")
    print()
    print("Note: token count is an estimate from whitespace words; the exact")
    print("SentencePiece token count will be measured after tokenizer training.")


if __name__ == "__main__":
    main()
