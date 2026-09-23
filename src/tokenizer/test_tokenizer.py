from pathlib import Path

import sentencepiece as spm

MODEL_FILE = Path("tokenizer/lawlm.model")

TEST_TEXTS = [
    "The court may grant relief under the applicable law.",
    "Section 302 of the Indian Penal Code provides punishment.",
    "A contract is enforceable when the legal requirements are satisfied.",
]


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {MODEL_FILE}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(MODEL_FILE))

    print("=" * 60)
    print("LawLM - Tokenizer Test")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.get_piece_size():,}")

    for text in TEST_TEXTS:
        ids = tokenizer.encode(text, out_type=int)
        decoded = tokenizer.decode(ids)

        print()
        print("Original:")
        print(text)
        print("Token IDs:")
        print(ids[:40], "..." if len(ids) > 40 else "")
        print("Decoded:")
        print(decoded)

        if decoded.strip() != text.strip():
            raise AssertionError("Tokenizer decode did not reproduce the input text.")

    print()
    print("Tokenizer test passed!")


if __name__ == "__main__":
    main()
