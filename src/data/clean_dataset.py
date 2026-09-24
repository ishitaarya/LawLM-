"""Phase 2: clean, normalize, deduplicate and preserve legal provenance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

INPUT_FILE = Path("data/raw/open_india_law_legislation.jsonl")
OUTPUT_FILE = Path("data/cleaned/legal_corpus.jsonl")
MIN_CHARS = 200


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    stats = {
        "input": 0,
        "kept": 0,
        "duplicate_id": 0,
        "duplicate_text": 0,
        "too_short": 0,
        "missing_source": 0,
    }

    with INPUT_FILE.open("r", encoding="utf-8") as src, OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue

            stats["input"] += 1
            row = json.loads(line)
            text = clean_text(row.get("text", ""))

            if len(text) < MIN_CHARS:
                stats["too_short"] += 1
                continue

            act_id = str(row.get("act_id", "")).strip()
            section = str(row.get("section_number", "")).strip()
            unique_id = f"{act_id}:{section}"

            if unique_id in seen_ids:
                stats["duplicate_id"] += 1
                continue

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                stats["duplicate_text"] += 1
                continue

            source_url = str(row.get("source_url", "")).strip()
            if not source_url:
                stats["missing_source"] += 1

            cleaned = dict(row)
            cleaned["text"] = text
            cleaned["text_sha256"] = text_hash

            seen_ids.add(unique_id)
            seen_hashes.add(text_hash)

            dst.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    print("=" * 64)
    print("LawSuit LLM — Phase 2 Cleaning")
    print("=" * 64)
    for key, value in stats.items():
        print(f"{key:>18}: {value:,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
