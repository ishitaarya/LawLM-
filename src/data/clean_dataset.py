import json
import re
from pathlib import Path

INPUT_FILE = Path("data/raw/legal_documents.jsonl")
OUTPUT_FILE = Path("data/processed/legal_corpus.jsonl")
MIN_CHARS = 500


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
            total += 1
            row = json.loads(line)

            doc_id = str(row.get("doc_id", "")).strip()
            text = clean_text(row.get("text", ""))

            if len(text) < MIN_CHARS:
                too_short += 1
                continue
            if doc_id and doc_id in seen_ids:
                duplicate_id += 1
                continue
            if text in seen_text:
                duplicate_text += 1
                continue

            if doc_id:
                seen_ids.add(doc_id)
            seen_text.add(text)

            clean_row = {
                "doc_id": doc_id,
                "document_title": str(row.get("document_title", "")).strip(),
                "document_type": str(row.get("document_type", "")).strip(),
                "document_jurisdiction": str(row.get("document_jurisdiction", "")).strip(),
                "issuing_authority": str(row.get("issuing_authority", "")).strip(),
                "issue_date": str(row.get("issue_date", "")).strip(),
                "text": text,
            }
            dst.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            kept += 1

    print("=" * 60)
    print("LawLM - Dataset Cleaning")
    print("=" * 60)
    print(f"Input documents:       {total}")
    print(f"Kept documents:        {kept}")
    print(f"Duplicate IDs:         {duplicate_id}")
    print(f"Duplicate texts:       {duplicate_text}")
    print(f"Too short:             {too_short}")
    print(f"Output:                {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
