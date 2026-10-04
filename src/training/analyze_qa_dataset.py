"""Phase 7.5B — Analyze QA dataset quality and sequence-length risk."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import sentencepiece as spm

QA_DIR = Path("data/qa")
TOKENIZER = Path("data/tokenizer/lawsuit_bpe.model")
OUTPUT = Path("logs/qa_dataset_analysis.json")

MAX_SEQUENCE_LENGTH = 256
PROMPT_TEMPLATE = (
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}"
)


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(p: float) -> int:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
        return ordered[index]

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def main() -> None:
    if not TOKENIZER.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER))

    split_rows = {
        split: load(QA_DIR / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }

    analysis = {
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "tokenizer_vocab_size": tokenizer.vocab_size(),
        "splits": {},
    }

    total_rows = 0

    for split, rows in split_rows.items():
        question_lengths = []
        context_lengths = []
        answer_lengths = []
        prompt_lengths = []
        over_limit = 0
        near_limit = 0
        empty_fields = 0
        question_types = Counter()
        acts = Counter()
        sections = Counter()

        for row in rows:
            question = str(row.get("question", "")).strip()
            context = str(row.get("context", "")).strip()
            answer = str(row.get("answer", "")).strip()

            if not question or not context or not answer:
                empty_fields += 1

            question_lengths.append(len(tokenizer.encode(question, out_type=int)))
            context_lengths.append(len(tokenizer.encode(context, out_type=int)))
            answer_lengths.append(len(tokenizer.encode(answer, out_type=int)))

            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=question,
                answer=answer,
            )
            prompt_length = len(tokenizer.encode(prompt, out_type=int))
            prompt_lengths.append(prompt_length)

            if prompt_length > MAX_SEQUENCE_LENGTH:
                over_limit += 1
            if prompt_length >= int(MAX_SEQUENCE_LENGTH * 0.90):
                near_limit += 1

            question_types[str(row.get("question_type", "unknown"))] += 1
            acts[str(row.get("act_id", "unknown"))] += 1
            sections[str(row.get("section", "unknown"))] += 1

        total_rows += len(rows)

        analysis["splits"][split] = {
            "records": len(rows),
            "empty_required_fields": empty_fields,
            "question_types": dict(question_types),
            "unique_acts": len(acts),
            "unique_sections": len(sections),
            "question_tokens": summarize_lengths(question_lengths),
            "context_tokens": summarize_lengths(context_lengths),
            "answer_tokens": summarize_lengths(answer_lengths),
            "combined_prompt_tokens": summarize_lengths(prompt_lengths),
            "over_256_tokens": over_limit,
            "over_256_percent": round(100 * over_limit / len(rows), 2) if rows else 0.0,
            "at_or_above_90pct_context": near_limit,
            "at_or_above_90pct_context_percent": (
                round(100 * near_limit / len(rows), 2) if rows else 0.0
            ),
            "top_acts": acts.most_common(10),
        }

    analysis["total_records"] = total_rows

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5B QA Dataset Analysis")
    print("=" * 72)
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size():,}")
    print(f"Maximum sequence length: {MAX_SEQUENCE_LENGTH}")
    for split, report in analysis["splits"].items():
        print(f"\n{split.upper()}: {report['records']:,} records")
        print(f"  Question types: {report['question_types']}")
        print(f"  Unique acts: {report['unique_acts']:,}")
        print(f"  Empty required fields: {report['empty_required_fields']}")
        print(
            "  Combined prompt tokens — "
            f"mean {report['combined_prompt_tokens']['mean']}, "
            f"p95 {report['combined_prompt_tokens']['p95']}, "
            f"max {report['combined_prompt_tokens']['max']}"
        )
        print(
            f"  >256 tokens: {report['over_256_tokens']:,} "
            f"({report['over_256_percent']}%)"
        )
        print(
            f"  >=90% of 256: {report['at_or_above_90pct_context']:,} "
            f"({report['at_or_above_90pct_context_percent']}%)"
        )

    print(f"\nTOTAL RECORDS: {total_rows:,}")
    print(f"Report: {OUTPUT}")
    print("QA dataset analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
