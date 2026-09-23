import json
from pathlib import Path
from urllib.request import Request, urlopen

# Open India Law publishes its legislation data as Parquet files in a public mirror.
# We avoid the gated Hugging Face endpoint and download a small sample directly.
BASE_URL = "https://oss-data-in.vaquill.ai/"
OUTPUT_FILE = Path("data/raw/open_india_law_legislation.jsonl")
MAX_DOCUMENTS = 5000

def main():
    print("Open India Law's public mirror is available, but its raw Parquet files are")
    print("large. The next step will select one or more legislation shards before")
    print("streaming only the required 5,000 provisions.")
    print()
    print("Dataset source:", BASE_URL)
    print("Target provisions:", MAX_DOCUMENTS)
    print("Output:", OUTPUT_FILE)
    print()
    raise RuntimeError(
        "Dataset shard URL must be selected from the current public mirror listing. "
        "The previous Hugging Face route is gated and has been disabled in LawLM."
    )

if __name__ == "__main__":
    main()
