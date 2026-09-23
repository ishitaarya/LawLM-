import json
from pathlib import Path

from datasets import load_dataset

# Open India Law publishes legislation as per-jurisdiction Parquet files.
# We use the public OSS mirror instead of the gated Hugging Face dataset.
BASE_URL = "https://oss-data-in.vaquill.ai"
VERSION = "v2026.08.1"
JURISDICTION = "madhya_pradesh"
MAX_DOCUMENTS = 5000

OUTPUT_FILE = Path("data/raw/open_india_law_legislation.jsonl")
SOURCE_URL = f"{BASE_URL}/{VERSION}/in_{JURISDICTION}_legislation.parquet"


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LawLM - Open India Law Downloader")
    print("=" * 60)
    print(f"Source:       {SOURCE_URL}")
    print(f"Target rows:  {MAX_DOCUMENTS:,}")
    print(f"Output:       {OUTPUT_FILE}")
    print()

    dataset = load_dataset(
        "parquet",
        data_files={"train": SOURCE_URL},
        split="train",
        streaming=True,
    )

    count = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for row in dataset:
            text = str(row.get("text") or "").strip()
            if not text:
                continue

            clean_row = {
                "act_id": str(row.get("act_id") or "").strip(),
                "title": str(row.get("title") or "").strip(),
                "section_number": str(row.get("section_number") or "").strip(),
                "section_title": str(row.get("section_title") or "").strip(),
                "state": str(row.get("state") or "").strip(),
                "year": row.get("year"),
                "act_status": str(row.get("act_status") or "").strip(),
                "section_status": str(row.get("section_status") or "").strip(),
                "source_url": str(row.get("source_url") or "").strip(),
                "source_publisher": str(row.get("source_publisher") or "").strip(),
                "text": text,
            }

            file.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            count += 1

            if count % 500 == 0:
                print(f"Downloaded: {count:,}")

            if count >= MAX_DOCUMENTS:
                break

    print()
    print(f"Saved provisions: {count:,}")
    print(f"Output: {OUTPUT_FILE}")

    if count == 0:
        raise RuntimeError("No legal provisions were downloaded.")


if __name__ == "__main__":
    main()
