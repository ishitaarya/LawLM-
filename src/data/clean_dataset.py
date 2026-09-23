import json
import re
from pathlib import Path

INPUT_FILE = Path("data/raw/open_india_law_legislation.jsonl")
OUTPUT_FILE = Path("data/processed/legal_corpus.jsonl")
MIN_CHARS = 200


def clean_text(text):
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    seen_text = set()
    total = kept = duplicate_id = duplicate_text = too_short = 0

    with INPUT_FILE.open("r", encoding="utf-8") as src, OUTPUT_FILE.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue

            total += 1
            row = json.loads(line)

            doc_id = str(row.get("act_id", "")).strip()
            section_number = str(row.get("section_number", "")).strip()
            text = clean_text(row.get("text", ""))

            if len(text) < MIN_CHARS:
                too_short += 1
                continue

            unique_id = f"{doc_id}:{section_number}"
            if unique_id in seen_ids:
                duplicate_id += 1
                continue

            if text in seen_text:
                duplicate_text += 1
                continue

            seen_ids.add(unique_id)
            seen_text.add(text)

            clean_row = {
                "act_id": doc_id,
                "title": str(row.get("title", "")).strip(),
                "section_number": section_number,
                "section_title": str(row.get("section_title", "")).strip(),
                "state": str(row.get("state", "")).strip(),
                "year": row.get("year"),
                "act_status": str(row.get("act_status", "")).strip(),
                "section_status": str(row.get("section_status", "")).strip(),
                "source_url": str(row.get("source_url", "")).strip(),
                "source_publisher": str(row.get("source_publisher", "")).strip(),
                "text": text,
            }

            dst.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            kept += 1

    print("=" * 60)
    print("LawLM - Open India Law Cleaning")
    print("=" * 60)
    print(f"Input provisions:     {total}")
    print(f"Kept provisions:      {kept}")
    print(f"Duplicate IDs:        {duplicate_id}")
    print(f"Duplicate texts:      {duplicate_text}")
    print(f"Too short:            {too_short}")
    print(f"Output:               {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
