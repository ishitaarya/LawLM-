"""Phase 7.5F — Build a tokenizer-safe QA dataset for conservative adaptation.

V2 keeps the project's own tokenizer and legal text. It removes the fragile
newline/template formatting used by the first QA run and records UNK usage.
No external or pretrained model is used.
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

def split_text(text: str) -> list[str]:
    text = " ".join(text.split()).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.;:])\s+|(?<=\))\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    merged = []
    for p in parts:
        if merged and len(p) < 45:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged

def token_chunks(sp, text: str) -> list[str]:
    parts = split_text(text)
    chunks, current = [], []
    current_tokens = 0
    for part in parts:
        ids = sp.encode(part, out_type=int)
        if len(ids) > MAX_ANSWER_TOKENS:
            words = part.split()
            piece = []
            piece_tokens = 0
            for word in words:
                wt = len(sp.encode((" " if piece else "") + word, out_type=int))
                if piece and piece_tokens + wt > MAX_ANSWER_TOKENS:
                    chunks.append(" ".join(piece))
                    piece, piece_tokens = [word], len(sp.encode(word, out_type=int))
                else:
                    piece.append(word)
                    piece_tokens += wt
            pieces = [" ".join(piece)] if piece else []
        else:
            pieces = [part]
        for piece in pieces:
            pt = len(sp.encode(piece, out_type=int))
            if current and current_tokens + pt > MAX_ANSWER_TOKENS:
                chunks.append(" ".join(current))
                current, current_tokens = [piece], pt
            else:
                current.append(piece)
                current_tokens += pt
    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if c.strip()]

def choose_template(sp, question: str) -> tuple[str, list[int]]:
    # Prefer templates with no unknown token. The question itself may contain
    # an unknown character, so candidates are tested rather than assumed.
    candidates = [
        f"Question {question} Answer",
        f"Question {question} Answer:",
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
    answer_ids = sp.encode(answer, out_type=int)[:max(1, budget)]
    ids = prefix_ids + answer_ids + [EOS_ID]
    labels = [-100] * len(prefix_ids) + answer_ids + [EOS_ID]
    return prefix, ids[:MAX_LENGTH], labels[:MAX_LENGTH]

def main():
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"max_sequence_length": MAX_LENGTH, "max_answer_tokens": MAX_ANSWER_TOKENS,
               "external_llm_used": False, "pretrained_model_used": False, "splits": {}}
    for split in ("train", "validation", "test"):
        rows = load(SOURCE_DIR / f"{split}.jsonl")
        out, unk_prompt = [], 0
        for row in rows:
            chunks = token_chunks(sp, row["answer"].strip())
            for chunk_index, chunk in enumerate(chunks):
                if len(sp.encode(chunk, out_type=int)) < MIN_ANSWER_TOKENS:
                    continue
                q = row["question"]
                if row.get("question_type") == "section_provision":
                    q = f"{q} Give the relevant provision text."
                template, ids, labels = build_sequence(sp, q, chunk)
                prompt_len = sum(x == -100 for x in labels)
                unk_prompt += sum(x == 1 for x in ids[:prompt_len])
                out.append({
                    "input_ids": ids, "labels": labels,
                    "attention_mask": [1] * len(ids),
                    "question": q, "answer": chunk,
                    "section": row["section"], "act_id": row["act_id"],
                    "source_id": row["source_id"],
                    "question_type": row.get("question_type", "unknown"),
                    "chunk_index": chunk_index, "chunk_count": len(chunks),
                    "answer_tokens": len(sp.encode(chunk, out_type=int)),
                    "prompt_template": template,
                })
        with (OUTPUT_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["splits"][split] = {
            "records": len(out), "source_records": len(rows),
            "prompt_unk_tokens": unk_prompt,
            "prompt_tokens": sum(sum(x == -100 for x in r["labels"]) for r in out),
            "answer_tokens": sum(sum(x != -100 for x in r["labels"]) for r in out),
            "max_sequence_tokens": max((len(r["input_ids"]) for r in out), default=0),
        }
        print(f"{split}: {len(out):,} records; prompt UNK {unk_prompt:,}")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("QA V2 dataset written to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
