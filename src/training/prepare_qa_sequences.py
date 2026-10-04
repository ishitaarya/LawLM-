"""Phase 7.5C — Prepare QA sequences for the 256-token LawSuit LLM.

The base QA dataset contains full legal provisions as both context and answer.
Because the Transformer context length is 256, this preparation step removes
redundant context before shortening the answer. No external or pretrained
model is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm

QA_DIR = Path("data/qa")
OUTPUT_DIR = Path("data/qa_tokenized")
TOKENIZER = Path("data/tokenizer/lawsuit_bpe.model")

MAX_LENGTH = 256
EOS_ID = 3
BOS_ID = 2


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def encode(tokenizer: spm.SentencePieceProcessor, text: str) -> list[int]:
    return tokenizer.encode(text, out_type=int)


def build_prompt(row: dict, context: str) -> str:
    return (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{row['question']}\n\n"
        "ANSWER:\n"
    )


def prepare_row(row: dict, tokenizer: spm.SentencePieceProcessor) -> dict:
    # Keep metadata and the legal answer. Context is progressively shortened
    # because it is redundant with the answer in this deterministic dataset.
    full_context = str(row["context"]).strip()
    answer = str(row["answer"]).strip()

    context_ids = encode(tokenizer, full_context)
    question_ids = encode(tokenizer, str(row["question"]).strip())
    answer_ids = encode(tokenizer, answer)

    # Reserve space for structural labels, question, answer and EOS.
    prefix = (
        "CONTEXT:\n\n"
        "QUESTION:\n"
        f"{row['question'].strip()}\n\n"
        "ANSWER:\n"
    )
    prefix_ids = encode(tokenizer, prefix)

    # Prefer retaining the complete answer. Use as much context as possible
    # without exceeding the model's 256-token sequence length.
    available_for_context = max(
        0, MAX_LENGTH - len(prefix_ids) - len(answer_ids) - 1
    )
    context_kept = context_ids[:available_for_context]

    # If the answer itself is too long, reserve the complete prompt/question
    # and keep the answer prefix that fits.
    if len(prefix_ids) + len(context_kept) + len(answer_ids) + 1 > MAX_LENGTH:
        answer_budget = max(
            1, MAX_LENGTH - len(prefix_ids) - len(context_kept) - 1
        )
        answer_ids = answer_ids[:answer_budget]

    input_ids = (
        encode(tokenizer, "CONTEXT:\n")
        + context_kept
        + encode(tokenizer, "\n\nQUESTION:\n")
        + question_ids
        + encode(tokenizer, "\n\nANSWER:\n")
        + answer_ids
        + [EOS_ID]
    )

    # Final defensive truncation. This should only trigger for unusual
    # tokenizer behavior around the fixed structural strings.
    input_ids = input_ids[:MAX_LENGTH]

    answer_start_text = (
        encode(tokenizer, "CONTEXT:\n")
        + context_kept
        + encode(tokenizer, "\n\nQUESTION:\n")
        + question_ids
        + encode(tokenizer, "\n\nANSWER:\n")
    )
    answer_start = min(len(answer_start_text), len(input_ids))

    labels = [-100] * answer_start + input_ids[answer_start:]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "question": row["question"],
        "answer": row["answer"],
        "section": row["section"],
        "act_id": row["act_id"],
        "source_id": row["source_id"],
        "question_type": row.get("question_type", "unknown"),
        "original_context_tokens": len(context_ids),
        "original_answer_tokens": len(encode(tokenizer, answer)),
        "prepared_context_tokens": len(context_kept),
        "prepared_sequence_tokens": len(input_ids),
        "answer_tokens_used": max(0, len(input_ids) - answer_start),
    }


def main() -> None:
    if not TOKENIZER.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "max_length": MAX_LENGTH,
        "tokenizer_vocab_size": tokenizer.vocab_size(),
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        rows = load(QA_DIR / f"{split}.jsonl")
        output = OUTPUT_DIR / f"{split}.jsonl"

        prepared = [prepare_row(row, tokenizer) for row in rows]

        with output.open("w", encoding="utf-8") as file:
            for row in prepared:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        truncated_answers = sum(
            row["answer_tokens_used"] < row["original_answer_tokens"]
            for row in prepared
        )
        context_trimmed = sum(
            row["prepared_context_tokens"]
            < row["original_context_tokens"]
            for row in prepared
        )

        summary["splits"][split] = {
            "records": len(prepared),
            "context_trimmed": context_trimmed,
            "answer_truncated": truncated_answers,
            "max_sequence_tokens": max(
                (len(row["input_ids"]) for row in prepared), default=0
            ),
        }

        print(
            f"{split:>12}: {len(prepared):,} records | "
            f"context trimmed: {context_trimmed:,} | "
            f"answers truncated: {truncated_answers:,}"
        )

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("=" * 72)
    print("QA sequence preparation complete")
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(f"Output: {OUTPUT_DIR}")
    print("Loss masking: prompt/context tokens = -100; answer tokens are trained")
    print("=" * 72)


if __name__ == "__main__":
    main()
