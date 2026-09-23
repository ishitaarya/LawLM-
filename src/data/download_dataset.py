from datasets import load_dataset
from pathlib import Path
import json

DATASET_NAME = "KanoonGPT/indian-legal-documents"
OUTPUT_FILE = Path("data/raw/legal_documents.jsonl")
MAX_DOCUMENTS = 5000

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("LawLM - Legal Dataset Download")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Target documents: {MAX_DOCUMENTS}")
    print()
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    count = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for item in dataset:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")
            count += 1
            if count % 500 == 0:
                print(f"Downloaded: {count} documents")
            if count >= MAX_DOCUMENTS:
                break
    print()
    print("=" * 60)
    print("Dataset download completed!")
    print(f"Documents saved: {count}")
    print(f"Location: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
