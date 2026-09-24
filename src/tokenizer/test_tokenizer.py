from pathlib import Path

import sentencepiece as spm

MODEL_FILE = Path("data/tokenizer/lawsuit_bpe.model")

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
    print("LawSuit LLM - Phase 3 Tokenizer Test")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.get_piece_size():,}")

    assert tokenizer.pad_id() == 0
    assert tokenizer.unk_id() == 1
    assert tokenizer.bos_id() == 2
    assert tokenizer.eos_id() == 3

    for text in TEST_TEXTS:
        ids = tokenizer.encode(text, out_type=int)
        decoded = tokenizer.decode(ids)

        if decoded.strip() != text.strip():
            raise AssertionError("Tokenizer decode did not reproduce the input text.")

        print()
        print("Original:", text)
        print("Token IDs:", ids[:40], "..." if len(ids) > 40 else "")
        print("Decoded:", decoded)

    print()
    print("Tokenizer test passed!")


if __name__ == "__main__":
    main()
