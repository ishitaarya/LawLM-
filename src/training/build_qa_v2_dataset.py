"""Phase 7.5F — Build a tokenizer-safe, causally aligned QA dataset.

Each source QA record becomes one training example. Prompts identify the Act
and section so generic questions such as "What does Section 5 provide?" are
not ambiguous. Answers are capped to fit the context window.

Labels at position t target the token at position t+1. The prompt is masked,
and the final prompt token predicts the first answer token. No external or
pretrained model is used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import sentencepiece as spm

SOURCE_DIR = Path("data/qa")
OUTPUT_DIR = Path("data/qa_v2")
TOKENIZER = Path("data/tokenizer/lawsuit_bpe.model")
MAX_LENGTH = 256
EOS_ID = 3
MIN_ANSWER_TOKENS = 8
MAX_ANSWER_TOKENS = 160


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_act_title(row: dict) -> str:
    """Extract the human-readable Act title from the source context."""
    context = row.get("context", "")
    match = re.search(r"(?m)^Title:\\s*(.+?)\\s*$", context)
    if match:
        return match.group(1).strip()
    return str(row.get("act_id", "Unknown Act"))


def choose_template(sp, question: str) -> tuple[str, list[int]]:
    candidates = [
        f"{question} Answer:",
        f"{question} Answer",
        f"{question}",
    ]
    for text in candidates:
        ids = sp.encode(text, out_type=int)
        if 1 not in ids:
            return text, ids
    text = candidates[0]
    return text, sp.encode(text, out_type=int)


def build_sequence(sp, question: str, answer: str):
    prefix, prefix_ids = choose_template(sp, question)
    budget = MAX_LENGTH - len(prefix_ids) - 1
    if budget < MIN_ANSWER_TOKENS:
        raise ValueError(
            f"Prompt is too long ({len(prefix_ids)} tokens); "
            f"cannot fit a useful answer in {MAX_LENGTH} tokens."
        )

    answer_ids = sp.encode(answer, out_type=int)[
        : min(budget, MAX_ANSWER_TOKENS)
    ]
    if len(answer_ids) < MIN_ANSWER_TOKENS:
        return prefix, [], [], len(prefix_ids), answer_ids

    ids = prefix_ids + answer_ids + [EOS_ID]
    # Input at position t predicts the next token at position t+1.
    # The last prompt token predicts answer_ids[0]; EOS is also predicted.
    labels = (
        [-100] * (len(prefix_ids) - 1)
        + answer_ids
        + [EOS_ID, -100]
    )
    return prefix, ids, labels, len(prefix_ids), answer_ids


def main():
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "max_sequence_length": MAX_LENGTH,
        "max_answer_tokens": MAX_ANSWER_TOKENS,
        "one_example_per_source_record": True,
        "act_and_section_in_prompt": True,
        "external_llm_used": False,
        "pretrained_model_used": False,
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        rows = load(SOURCE_DIR / f"{split}.jsonl")
        out, unk_prompt, skipped = [], 0, 0

        for row in rows:
            act_title = extract_act_title(row)
            section = str(row.get("section", "Unknown"))
            original_question = " ".join(str(row["question"]).split())
            question = (
                f"Act: {act_title}. Section: {section}. "
                f"Question: {original_question}"
            )
            answer_source = " ".join(str(row["answer"]).split())

            template, ids, labels, prompt_length, answer_ids = build_sequence(
                sp, question, answer_source
            )
            if not ids or len(answer_ids) < MIN_ANSWER_TOKENS:
                skipped += 1
                continue

            # Store exactly the answer represented by the supervised token IDs.
            answer = sp.decode(answer_ids)
            unk_prompt += sum(token == 1 for token in ids[:prompt_length])

            out.append({
                "input_ids": ids,
                "labels": labels,
                "prompt_length": prompt_length,
                "attention_mask": [1] * len(ids),
                "question": question,
                "answer": answer,
                "section": section,
                "act_id": row["act_id"],
                "source_id": row["source_id"],
                "question_type": row.get("question_type", "unknown"),
                "chunk_index": 0,
                "chunk_count": 1,
                "answer_tokens": len(answer_ids),
                "prompt_template": template,
            })

        output_path = OUTPUT_DIR / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\\n")

        summary["splits"][split] = {
            "records": len(out),
            "source_records": len(rows),
            "skipped_short_answers": skipped,
            "prompt_unk_tokens": unk_prompt,
            "prompt_tokens": sum(row["prompt_length"] for row in out),
            "supervised_tokens": sum(
                sum(token != -100 for token in row["labels"]) for row in out
            ),
            "max_sequence_tokens": max(
                (len(row["input_ids"]) for row in out), default=0
            ),
        }
        print(
            f"{split}: {len(out):,} records; "
            f"skipped {skipped}; prompt UNK {unk_prompt:,}"
        )

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Causally aligned QA V2 dataset written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
