import json
from pathlib import Path

import sentencepiece as spm

INPUT_FILE = Path("data/processed/legal_corpus.jsonl")
MODEL_PREFIX = Path("tokenizer/lawlm")
VOCAB_SIZE = 8000


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Clean dataset not found: {INPUT_FILE}")

    MODEL_PREFIX.parent.mkdir(parents=True, exist_ok=True)

    text_file = Path("tokenizer/legal_text.txt")
    with INPUT_FILE.open("r", encoding="utf-8") as src, text_file.open("w", encoding="utf-8") as dst:
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
        user_defined_symbols=["§"],
    )

    text_file.unlink(missing_ok=True)

    print("=" * 60)
    print("LawLM - Tokenizer Training")
    print("=" * 60)
    print(f"Corpus:       {INPUT_FILE}")
    print(f"Vocabulary:   {VOCAB_SIZE:,}")
    print(f"Model:        {MODEL_PREFIX}.model")
    print(f"Vocabulary:   {MODEL_PREFIX}.vocab")
    print("=" * 60)


if __name__ == "__main__":
    main()
