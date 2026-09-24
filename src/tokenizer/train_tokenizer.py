import json
from pathlib import Path

import sentencepiece as spm

INPUT_FILE = Path("data/splits/train.jsonl")
OUTPUT_DIR = Path("data/tokenizer")
MODEL_PREFIX = OUTPUT_DIR / "lawsuit_bpe"
VOCAB_SIZE = 10_000


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Training split not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text_file = OUTPUT_DIR / "training_text.txt"
    with INPUT_FILE.open("r", encoding="utf-8") as src, text_file.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if text:
                dst.write(text + "\n")

    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        normalization_rule_name="identity",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    text_file.unlink(missing_ok=True)

    tokenizer = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")

    print("=" * 60)
    print("LawSuit LLM - Phase 3 BPE Tokenizer")
    print("=" * 60)
    print(f"Corpus:       {INPUT_FILE}")
    print(f"Vocabulary:   {tokenizer.get_piece_size():,}")
    print(f"Model:        {MODEL_PREFIX}.model")
    print(f"Vocabulary:   {MODEL_PREFIX}.vocab")
    print("Training:     train split only")
    print("=" * 60)


if __name__ == "__main__":
    main()
