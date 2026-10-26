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


def has_obvious_ocr_corruption(text: str) -> bool:
    """Flag clear OCR debris without trying to auto-correct legal wording.

    This catches high-confidence artifacts only. Plausible-looking OCR
    substitutions still require source review.
    """
    if "\ufffd" in text or "\u25a1" in text:
        return True

    hard_artifact_patterns = (
        r"\b[A-Za-z]{2,}~[A-Za-z]*",       # e.g. Collecto~, Rcgulatio~zs
        r"\b[A-Za-z]{2,}[,;:]{2,}",          # e.g. pow,,
        r"\b[A-Za-z]{2,}-\s+(?=[A-Za-z])",   # broken scan/line fragments
    )
    if any(re.search(pattern, text) for pattern in hard_artifact_patterns):
        return True

    noisy_fragments = re.findall(
        r"(?<![A-Za-z])(?:[A-Z]{1,2}\s+){3,}[A-Z0-9$]{1,3}(?![A-Za-z])"
        r"|\b[A-Za-z]{2,}_[A-Za-z]{1,}\b"
        r"|\b[A-Za-z]{2,}\d[A-Za-z]{2,}\b",
        text,
    )
    return len(noisy_fragments) >= 2


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

    # Stop at the next explicit section marker, or at a likely numbered
    # provision heading. OCR sources use both plain "20." and bold "**20.**".
    numeric = re.fullmatch(r"(\d+)[A-Za-z]?", section_value)
    if numeric:
        target_number = int(numeric.group(1))
        explicit = re.search(
            r"\s+Section\s*:?\s*(\d+[A-Za-z]?)\s*[:.]\s*",
            body,
            flags=re.IGNORECASE,
        )
        candidates = []
        if explicit:
            explicit_num = re.match(r"(\d+)", explicit.group(1))
            if explicit_num and int(explicit_num.group(1)) > target_number:
                candidates.append(explicit)
        # Accept both **20.** and **20**. OCR/Markdown exports vary in
        # whether the period is inside or outside the bold markers.
        # Keep this anchored to a numbered heading followed by uppercase prose
        # to avoid cutting ordinary numeric references in legal text.
        for provision in re.finditer(
            r"\s+(?:>\s*)?(?:\*\*)?(\d+)[A-Za-z]?(?:\.)?(?:\*\*)?\.?(?:\*\*)?\s+(?=[A-Z])",
            body,
        ):
            if int(provision.group(1)) > target_number:
                candidates.append(provision)
        if candidates:
            body = body[:min(m.start() for m in candidates)].strip()
            # A provision boundary should normally follow a complete clause.
            # If extraction stops mid-word or without terminal punctuation,
            # reject the fragment rather than training on a misleading answer.
            if not re.search(r"[.!?;:][\"'’”)}\]]*$", body):
                return "", "target_section_incomplete_before_next_provision_rejected"

    # Remove the common OCR pattern where the section opening is repeated
    # after a bare numbered marker, e.g. "law 5. All duties ...".
    duplicate_pattern = re.compile(
        rf"^(.*?)\s+{re.escape(section_value)}\.\s+(.+)$",
        flags=re.IGNORECASE,
    )
    duplicate = duplicate_pattern.match(body)
    if duplicate:
        opening = duplicate.group(1).strip()
        repeated = duplicate.group(2).strip()
        if opening and repeated.lower().startswith(opening.lower()):
            body = opening + repeated[len(opening):]

    # Remove an embedded bold legal heading before the next provision text.
    # Example: "... in accordance with law. **Legalization ... Indemnity.** And all officers ..."
    body = re.sub(
        r"\s+\*\*[^*]{1,120}\*\*\s+(?=(?:And|But|The|No|Any|All|Every|Nothing|Provided)\b)",
        " ",
        body,
        flags=re.IGNORECASE,
    )

    # Conservative OCR repairs observed in this corpus. Literal replacements
    # are intentional here because these defects occur inside merged OCR tokens.
    body = re.sub(r"\ba11\b", "all", body, flags=re.IGNORECASE)
    body = body.replace("ifthis", "if this").replace("Ifthis", "If this")
    body = body.replace("respectof", "respect of").replace("Respectof", "Respect of")
    body = body.replace("anyof", "any of").replace("Anyof", "Any of")
    body = body.replace("ofthe", "of the").replace("Ofthe", "Of the")
    body = body.replace("intoforce", "into force").replace("Intoforce", "Into force")
    body = body.replace("otherproceeding", "other proceeding")
    body = body.replace("of, British Burma", "of British Burma")
    body = re.sub(r" +", " ", body).strip()
    if has_obvious_ocr_corruption(body):
        return "", "target_section_obvious_ocr_corruption_rejected"
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
        "section_answer_cleanup": True,
        "unidentifiable_answers_rejected": True,
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
            "target_section_marker_not_found_rejected": 0,
            "target_section_body_empty_rejected": 0,
            "target_section_body_too_short_rejected": 0,
            "target_section_residual_metadata_rejected": 0,
            "target_section_incomplete_before_next_provision_rejected": 0,
            "target_section_obvious_ocr_corruption_rejected": 0,
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
            if not answer_source:
                skipped += 1
                continue

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
