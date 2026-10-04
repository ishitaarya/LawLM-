"""Phase 7.5C.1 — Build section-aware QA examples for 256-token training.

Long legal provisions are split into deterministic sentence/clause chunks.
Each chunk receives a question and a self-contained answer. This avoids
training the model on answers that are arbitrarily cut at 256 tokens.
No external or pretrained model is used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import sentencepiece as spm

SOURCE_DIR = Path("data/qa")
OUTPUT_DIR = Path("data/qa_chunked")
TOKENIZER = Path("data/tokenizer/lawsuit_bpe.model")

MAX_LENGTH = 256
EOS_ID = 3
MIN_ANSWER_TOKENS = 8
MAX_ANSWER_TOKENS = 170


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def split_text(text: str) -> list[str]:
    """Split on sentence/clause boundaries while retaining legal wording."""
    text = " ".join(text.split()).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.;:])\s+|(?<=\))\s+", text)
    parts = [part.strip() for part in parts if part.strip()]

    # Merge tiny fragments with the following fragment.
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < 45:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    return merged


def token_chunks(
    tokenizer: spm.SentencePieceProcessor,
    text: str,
) -> list[str]:
    """Split a provision into token-safe chunks."""
    parts = split_text(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = len(tokenizer.encode(part, out_type=int))

        if part_tokens > MAX_ANSWER_TOKENS:
            words = part.split()
            piece: list[str] = []
            piece_tokens = 0
            for word in words:
                word_tokens = len(tokenizer.encode((" " if piece else "") + word, out_type=int))
                if piece and piece_tokens + word_tokens > MAX_ANSWER_TOKENS:
                    chunks.append(" ".join(piece))
                    piece = [word]
                    piece_tokens = len(tokenizer.encode(word, out_type=int))
                else:
                    piece.append(word)
                    piece_tokens += word_tokens
            if piece:
                part_pieces = [" ".join(piece)]
            else:
                part_pieces = []
        else:
            part_pieces = [part]

        for piece in part_pieces:
            piece_tokens = len(tokenizer.encode(piece, out_type=int))
            if current and current_tokens + piece_tokens > MAX_ANSWER_TOKENS:
                chunks.append(" ".join(current))
                current = [piece]
                current_tokens = piece_tokens
            else:
                current.append(piece)
                current_tokens += piece_tokens

    if current:
        chunks.append(" ".join(current))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def build_sequence(
    tokenizer: spm.SentencePieceProcessor,
    question: str,
    answer: str,
) -> tuple[list[int], list[int]]:
    prefix = f"QUESTION:\n{question}\n\nANSWER:\n"
    prefix_ids = tokenizer.encode(prefix, out_type=int)
    answer_ids = tokenizer.encode(answer, out_type=int)

    budget = MAX_LENGTH - len(prefix_ids) - 1
    answer_ids = answer_ids[: max(1, budget)]

    input_ids = prefix_ids + answer_ids + [EOS_ID]
    labels = [-100] * len(prefix_ids) + answer_ids + [EOS_ID]

    return input_ids[:MAX_LENGTH], labels[:MAX_LENGTH]


def main() -> None:
    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "max_sequence_length": MAX_LENGTH,
        "max_answer_tokens": MAX_ANSWER_TOKENS,
        "external_llm_used": False,
        "pretrained_model_used": False,
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        rows = load(SOURCE_DIR / f"{split}.jsonl")
        output_rows: list[dict] = []

        for row in rows:
            source_text = row["answer"].strip()
            chunks = token_chunks(tokenizer, source_text)

            for chunk_index, chunk in enumerate(chunks):
                answer_tokens = len(tokenizer.encode(chunk, out_type=int))
                if answer_tokens < MIN_ANSWER_TOKENS:
                    continue

                question = row["question"]
                if row.get("question_type") == "section_provision":
                    question = f"{question} Give the relevant provision text."

                input_ids, labels = build_sequence(tokenizer, question, chunk)

                output_rows.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": [1] * len(input_ids),
                        "question": question,
                        "answer": chunk,
                        "section": row["section"],
                        "act_id": row["act_id"],
                        "source_id": row["source_id"],
                        "question_type": row.get("question_type", "unknown"),
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "answer_tokens": answer_tokens,
                    }
                )

        with (OUTPUT_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for row in output_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary["splits"][split] = {
            "records": len(output_rows),
            "source_records": len(rows),
            "multi_chunk_sources": sum(
                row["chunk_count"] > 1 for row in output_rows
            ),
            "max_sequence_tokens": max(
                (len(row["input_ids"]) for row in output_rows), default=0
            ),
        }

        print(
            f"{split:>12}: {len(rows):,} source QA -> "
            f"{len(output_rows):,} chunked QA records"
        )

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5C.1 Chunked QA Dataset")
    print("=" * 72)
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(f"Maximum answer tokens: {MAX_ANSWER_TOKENS}")
    print(f"Output: {OUTPUT_DIR}")
    print("External LLM: no")
    print("Pretrained model: no")
    print("=" * 72)


if __name__ == "__main__":
    main()
