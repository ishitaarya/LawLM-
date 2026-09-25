"""Phase 7.5.3: encode the validated corpus with the rebuilt tokenizer."""

import json
from pathlib import Path

import sentencepiece as spm

TOKENIZER_DIR = Path("data/tokenizer")
MODEL_FILE = TOKENIZER_DIR / "lawsuit_bpe.model"
SPLITS_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/tokenized")


def encode_split(name: str, tokenizer: spm.SentencePieceProcessor) -> tuple[int, int]:
    input_file = SPLITS_DIR / f"{name}.jsonl"
    output_file = OUTPUT_DIR / f"{name}.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(f"Split not found: {input_file}")

    count = token_count = 0

    with input_file.open("r", encoding="utf-8") as src, output_file.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if not text:
                continue

            tokens = tokenizer.encode(text, out_type=int)
            record = {
                "act_id": row.get("act_id"),
                "section_number": row.get("section_number"),
                "jurisdiction": row.get("jurisdiction") or row.get("state"),
                "tokens": tokens,
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            token_count += len(tokens)

    return count, token_count


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found: {MODEL_FILE}. Run train_tokenizer.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=str(MODEL_FILE))

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5.3 Dataset Encoding")
    print("=" * 72)
    print(f"Tokenizer:  {MODEL_FILE}")
    print(f"Vocabulary: {tokenizer.get_piece_size():,}")

    total_tokens = 0
    total_records = 0

    for split in ("train", "validation", "test"):
        count, tokens = encode_split(split, tokenizer)
        total_records += count
        total_tokens += tokens
        average = tokens / count if count else 0.0
        print(
            f"{split:10s}: {count:6,} records | "
            f"{tokens:10,} tokens | avg {average:.2f}"
        )

    print("-" * 72)
    print(f"TOTAL      : {total_records:6,} records | {total_tokens:10,} tokens")
    print("Tokenizer was trained on the train split only.")
    print("=" * 72)


if __name__ == "__main__":
    main()
