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

    count = 0
    token_count = 0

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
                "tokens": tokens,
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            token_count += len(tokens)

    return count, token_count


def main():
    model_file = TOKENIZER_DIR / "lawsuit_bpe.model"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found: {model_file}. Run train_tokenizer.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_file))

    print("=" * 60)
    print("LawSuit LLM - Phase 3 Dataset Encoding")
    print("=" * 60)
    print(f"Vocabulary: {tokenizer.get_piece_size():,}")

    for split in ("train", "validation", "test"):
        count, token_count = encode_split(split, tokenizer)
        average = token_count / count if count else 0.0
        print(f"{split:10s}: {count:5,} records | {token_count:9,} tokens | avg {average:.2f}")


if __name__ == "__main__":
    main()
