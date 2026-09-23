import json
from pathlib import Path

import sentencepiece as spm

INPUT_FILE = Path("data/splits/train.jsonl")
OUTPUT_FILE = Path("data/splits/train_tokens.jsonl")
MODEL_FILE = Path("tokenizer/lawlm.model")


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {MODEL_FILE}")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Training split not found: {INPUT_FILE}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(MODEL_FILE))
    count = 0

    with INPUT_FILE.open("r", encoding="utf-8") as src, OUTPUT_FILE.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if not text:
                continue

            tokens = tokenizer.encode(text, out_type=int)
            dst.write(json.dumps({"act_id": row.get("act_id"), "tokens": tokens}) + "\n")
            count += 1

    print(f"Encoded training documents: {count:,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
