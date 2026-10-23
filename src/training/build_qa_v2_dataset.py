"""Build a tokenizer-safe, causally aligned QA dataset from source records.

Before tokenization, remove Act-level metadata from answers when the source
answer contains a repeated full-document header. If a matching "Section N:"
marker exists, keep the target section body and drop a later explicit
"Section M:" header. The builder reports how often this cleanup was applied
or could not identify a matching section; inspect the summary before training.
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
    match = re.search(r"(?m)^Title:\s*(.+?)\s*$", context)
    if match:
        return match.group(1).strip()
    return str(row.get("act_id", "Unknown Act"))


def clean_section_answer(answer: str, section: str) -> tuple[str, str]:
    """Extract a target provision conservatively; reject unidentifiable answers.

    Source corpus rows can contain Act metadata, duplicate section text, and
    adjacent provisions. Never fall back to the whole contaminated answer.
    """
    text = " ".join(str(answer).split()).strip()
    section_value = str(section).strip()
    target = re.escape(section_value)
    marker = re.compile(
        rf"\bSection\s*[:] ?\s*{target}\s*:\s*".replace("[:] ?", "[:]?"),
        flags=re.IGNORECASE,
    )
    match = marker.search(text)
    if not match:
        return "", "target_section_marker_not_found_rejected"

    body = text[match.end():].strip()
    if not body:
        return "", "target_section_body_empty_rejected"

    # Stop at the next numbered provision, including headings whose OCR/markup
    # places a title between Markdown markers and the section number.
    numeric = re.fullmatch(r"(\d+)[A-Za-z]?", section_value)
    if numeric:
        next_number = str(int(numeric.group(1)) + 1)
        next_patterns = [
            rf"\s+(?:>\s*)?(?:\*\*[^*]{{1,100}}\*\*\s*)?{re.escape(next_number)}\.\s+",
            rf"\s+Section\s*[:] ?\s*{re.escape(next_number)}\s*[:.]\s*".replace("[:] ?", "[:]?"),
        ]
        matches = [m for pattern in next_patterns
                   if (m := re.search(pattern, body, flags=re.IGNORECASE))]
        if matches:
            body = body[:min(m.start() for m in matches)].strip()

    # Remove an immediately duplicated copy of the provision opening. This
    # corpus often repeats the first sentence after the explicit Section N:.
    sentence_end = re.search(r"[.!?](?:[”’\"'\]]*)?(?:\s+|$)", body)
    if sentence_end:
        first_sentence = body[:sentence_end.end()].strip()
        remainder = body[sentence_end.end():].lstrip()
        duplicate_prefix = re.compile(
            rf"^{re.escape(section_value)}\.\s*{re.escape(first_sentence)}",
            flags=re.IGNORECASE,
        )
        duplicate = duplicate_prefix.match(remainder)
        if duplicate:
            body = first_sentence + remainder[duplicate.end():]

    body = re.sub(r"\s+", " ", body).strip()
    if len(body) < 30:
        return "", "target_section_body_too_short_rejected"
    # Do not train on obvious Act metadata or a section heading left in the body.
    if re.search(r"\b(?:Act:.*\|\s*(?:India|Repealed)|Section\s+\d+\s*:)", body, re.I):
        return "", "target_section_residual_metadata_rejected"
    return body, "target_section_extracted"


def choose_template(sp, question: str) -> tuple[str, list[int]]:
    candidates = [f"{question} Answer:", f"{question} Answer", f"{question}"]
    for text in candidates:
        ids = sp.encode(text, out_type=int)
        if 1 not in ids:
            return text, ids
    text = candidates[0]
    return text, sp.encode(text, out_type=int)


def select_answer_tokens(sp, answer: str, budget: int) -> tuple[list[int], str]:
    """Prefer a complete sentence; flag hard-cap truncation when needed."""
    all_ids = sp.encode(answer, out_type=int)
    limit = min(budget, MAX_ANSWER_TOKENS)
    if len(all_ids) <= limit:
        return all_ids, "complete"

    capped_ids = all_ids[:limit]
    capped_text = sp.decode(capped_ids)
    matches = list(re.finditer(r'[.!?](?:["”’\')\]]*)(?:\s+|$)', capped_text))
    if matches:
        clean_text = capped_text[:matches[-1].end()].strip()
        clean_ids = sp.encode(clean_text, out_type=int)
        if MIN_ANSWER_TOKENS <= len(clean_ids) <= limit:
            return clean_ids, "sentence_boundary"
    return capped_ids, "hard_token_cap"


def build_sequence(sp, question: str, answer: str):
    prefix, prefix_ids = choose_template(sp, question)
    budget = MAX_LENGTH - len(prefix_ids) - 1
    if budget < MIN_ANSWER_TOKENS:
        raise ValueError(
            f"Prompt is too long ({len(prefix_ids)} tokens); "
            f"cannot fit a useful answer in {MAX_LENGTH} tokens."
        )

    answer_ids, truncation = select_answer_tokens(sp, answer, budget)
    if len(answer_ids) < MIN_ANSWER_TOKENS:
        return prefix, [], [], len(prefix_ids), answer_ids, truncation

    ids = prefix_ids + answer_ids + [EOS_ID]
    labels = [-100] * (len(prefix_ids) - 1) + answer_ids + [EOS_ID, -100]
    return prefix, ids, labels, len(prefix_ids), answer_ids, truncation


def main():
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "max_sequence_length": MAX_LENGTH,
        "max_answer_tokens": MAX_ANSWER_TOKENS,
        "one_example_per_source_record": True,
        "act_and_section_in_prompt": True,
        "section_answer_cleanup": True,\n        "unidentifiable_answers_rejected": True,
        "external_llm_used": False,
        "pretrained_model_used": False,
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        rows = load(SOURCE_DIR / f"{split}.jsonl")
        out, unk_prompt, skipped = [], 0, 0
        truncation_counts = {"complete": 0, "sentence_boundary": 0, "hard_token_cap": 0}
        cleanup_counts = {
            "target_section_extracted": 0,
            "target_section_marker_not_found": 0,
            "target_section_body_empty": 0,
        }

        for row in rows:
            act_title = extract_act_title(row)
            section = str(row.get("section", "Unknown")).strip()
            original_question = " ".join(str(row["question"]).split())
            question = (
                f"Act: {act_title}. Section: {section}. "
                f"Question: {original_question}"
            )
            raw_answer = " ".join(str(row["answer"]).split())
            answer_source, cleanup_status = clean_section_answer(raw_answer, section)
            cleanup_counts[cleanup_status] += 1

            template, ids, labels, prompt_length, answer_ids, truncation = build_sequence(
                sp, question, answer_source
            )
            if not ids or len(answer_ids) < MIN_ANSWER_TOKENS:
                skipped += 1
                continue

            answer = sp.decode(answer_ids)
            unk_prompt += sum(token == 1 for token in ids[:prompt_length])
            truncation_counts[truncation] += 1
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
                "answer_truncation": truncation,
                "answer_cleanup": cleanup_status,
                "prompt_template": template,
            })

        output_path = OUTPUT_DIR / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary["splits"][split] = {
            "records": len(out),
            "source_records": len(rows),
            "skipped_short_answers": skipped,
            "answer_cleanup_counts": cleanup_counts,
            "answer_truncation_counts": truncation_counts,
            "prompt_unk_tokens": unk_prompt,
            "prompt_tokens": sum(row["prompt_length"] for row in out),
            "supervised_tokens": sum(
                sum(token != -100 for token in row["labels"]) for row in out
            ),
            "max_sequence_tokens": max((len(row["input_ids"]) for row in out), default=0),
        }
        print(
            f"{split}: {len(out):,} records; skipped {skipped}; "
            f"cleanup {cleanup_counts}; truncation {truncation_counts}"
        )

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Causally aligned QA V2 dataset written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
