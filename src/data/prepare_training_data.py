import json
from pathlib import Path

import torch
import sentencepiece as spm


MODEL_FILE = Path("tokenizer/lawlm.model")

INPUT_FILES = {
    "train": Path("data/splits/train.jsonl"),
    "validation": Path("data/splits/validation.jsonl"),
    "test": Path("data/splits/test.jsonl"),
}

OUTPUT_DIR = Path("data/splits")

BLOCK_SIZE = 128


def load_tokens(input_file, tokenizer):
    all_tokens = []

    with input_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            text = str(row.get("text") or "").strip()

            if not text:
                continue

            tokens = tokenizer.encode(text, out_type=int)
            tokens.append(tokenizer.eos_id())
            all_tokens.extend(tokens)

    return all_tokens


def create_sequences(tokens):
    inputs = []
    targets = []

    for start in range(0, len(tokens) - BLOCK_SIZE, BLOCK_SIZE):
        sequence = tokens[start:start + BLOCK_SIZE + 1]

        if len(sequence) < BLOCK_SIZE + 1:
            continue

        x = sequence[:-1]
        y = sequence[1:]

        inputs.append(x)
        targets.append(y)

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(
        targets, dtype=torch.long
    )


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found: {MODEL_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(MODEL_FILE)
    )

    print("=" * 60)
    print("LawLM - Training Data Preparation")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.get_piece_size():,}")
    print(f"Block size:      {BLOCK_SIZE}")
    print()

    for split_name, input_file in INPUT_FILES.items():
        if not input_file.exists():
            raise FileNotFoundError(
                f"Dataset split not found: {input_file}"
            )

        print(f"Preparing {split_name}...")

        tokens = load_tokens(input_file, tokenizer)
        x, y = create_sequences(tokens)

        output_file = OUTPUT_DIR / f"{split_name}_data.pt"

        torch.save(
            {
                "inputs": x,
                "targets": y,
                "block_size": BLOCK_SIZE,
                "vocab_size": tokenizer.get_piece_size(),
            },
            output_file,
        )

        print(f"  Tokens:    {len(tokens):,}")
        print(f"  Sequences: {len(x):,}")
        print(f"  Shape X:   {tuple(x.shape)}")
        print(f"  Shape Y:   {tuple(y.shape)}")
        print(f"  Saved:     {output_file}")
        print()

    print("=" * 60)
    print("Training data preparation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
