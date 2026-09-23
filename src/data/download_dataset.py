from datasets import load_dataset
from pathlib import Path
import json

DATASET_NAME = "vaquill/open-india-law"
CONFIG_NAME = "legislation"
OUTPUT_FILE = Path("data/raw/open_india_law_legislation.jsonl")
MAX_DOCUMENTS = 5000


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LawLM - Open India Law Dataset Download")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Subset: {CONFIG_NAME}")
    print(f"Target provisions: {MAX_DOCUMENTS}")
    print()

    dataset = load_dataset(
        DATASET_NAME,
        CONFIG_NAME,
        split="train",
        streaming=True,
    )

    count = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for item in dataset:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")
            count += 1

            if count % 500 == 0:
                print(f"Downloaded: {count} provisions")

            if count >= MAX_DOCUMENTS:
                break

    print()
    print("=" * 60)
    print("Dataset download completed!")
    print(f"Provisions saved: {count}")
    print(f"Location: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
