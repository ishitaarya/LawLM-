"""Phase 2: stream a small, reproducible Indian legislation corpus.

The Open India Law snapshot publishes legislation as jurisdiction-specific
Parquet files. We intentionally start with a small number of records so the
pipeline can be inspected before scaling.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

BASE_URL = "https://oss-data-in.vaquill.ai"
VERSION = "v2026.08.1"
JURISDICTIONS = ("central", "madhya-pradesh")
MAX_RECORDS = 5000
OUTPUT_FILE = Path("data/raw/open_india_law_legislation.jsonl")


def source_url(jurisdiction: str) -> str:
    return f"{BASE_URL}/{VERSION}/in_{jurisdiction}_legislation.parquet"


def normalize_row(row: dict, jurisdiction: str) -> dict:
    return {
        "act_id": str(row.get("act_id") or "").strip(),
        "title": str(row.get("title") or "").strip(),
        "chapter": str(row.get("chapter") or "").strip(),
        "section_number": str(row.get("section_number") or "").strip(),
        "section_title": str(row.get("section_title") or "").strip(),
        "state": str(row.get("state") or jurisdiction).strip(),
        "year": row.get("year"),
        "amendment_count": row.get("amendment_count"),
        "act_status": str(row.get("act_status") or "").strip(),
        "section_status": str(row.get("section_status") or "").strip(),
        "source_url": str(row.get("source_url") or "").strip(),
        "source_publisher": str(row.get("source_publisher") or "").strip(),
        "source_snapshot": VERSION,
        "source_dataset": "Vaquill-AI/open-india-law",
        "text": str(row.get("text") or "").strip(),
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen = set()

    print("=" * 64)
    print("LawSuit LLM — Phase 2 Legal Corpus Ingestion")
    print("=" * 64)
    print(f"Snapshot:    {VERSION}")
    print(f"Jurisdictions: {', '.join(JURISDICTIONS)}")
    print(f"Record cap:  {MAX_RECORDS:,}")
    print(f"Output:      {OUTPUT_FILE}")
    print()

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for jurisdiction in JURISDICTIONS:
            url = source_url(jurisdiction)
            print(f"Streaming: {jurisdiction} -> {url}")

            dataset = load_dataset(
                "parquet",
                data_files={"train": url},
                split="train",
                streaming=True,
            )

            for row in dataset:
                normalized = normalize_row(row, jurisdiction)
                text = normalized["text"]
                key = (
                    normalized["act_id"],
                    normalized["section_number"],
                    text,
                )

                if not text or key in seen:
                    continue

                seen.add(key)
                file.write(
                    json.dumps(normalized, ensure_ascii=False) + "\n"
                )
                count += 1

                if count % 500 == 0:
                    print(f"  collected: {count:,}")

                if count >= MAX_RECORDS:
                    break

            if count >= MAX_RECORDS:
                break

    if count == 0:
        raise RuntimeError("No legal provisions were collected.")

    print()
    print(f"Collected: {count:,}")
    print(f"Saved to:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
