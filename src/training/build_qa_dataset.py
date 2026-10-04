"""
Phase 7.5A — Deterministic legal QA dataset generation.

Creates question/context/answer examples directly from the existing legal
corpus. No external LLM, pretrained model, or generated legal knowledge is
used. Answers are copied from the source provision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCE_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/qa")
TARGET_TOTAL = 20_000


def normalize(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def make_examples(row: dict) -> list[dict]:
    act_id = normalize(row.get("act_id"))
    section = normalize(row.get("section_number")) or "N/A"
    title = normalize(row.get("section_title")) or "this provision"
    text = normalize(row.get("text"))

    if not text:
        return []

    source_id = f"{act_id}:{section}:{row.get('text_sha256', '')}"
    context = (
        f"Act: {act_id}\n"
        f"Section: {section}\n"
        f"Title: {title}\n"
        f"Provision: {text}"
    )

    examples = [
        {
            "question": f"What does Section {section} provide?",
            "context": context,
            "answer": text,
            "section": section,
            "act_id": act_id,
            "source_id": source_id,
            "question_type": "section_provision",
        }
    ]

    if title and title.lower() not in {"this provision", "untitled section"}:
        examples.append(
            {
                "question": f"What is the legal provision regarding {title}?",
                "context": context,
                "answer": text,
                "section": section,
                "act_id": act_id,
                "source_id": source_id,
                "question_type": "title_provision",
            }
        )

    return examples


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(row: dict) -> str:
    payload = (
        row["source_id"]
        + "|"
        + row["question_type"]
        + "|"
        + row["question"]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    split_rows = {}
    for split in ("train", "validation", "test"):
        path = SOURCE_DIR / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus split: {path}")
        split_rows[split] = load_rows(path)

    all_candidates: dict[str, list[dict]] = {}
    for split, rows in split_rows.items():
        candidates = []
        for row in rows:
            candidates.extend(make_examples(row))
        for example in candidates:
            all_candidates.setdefault(stable_key(example), []).append(example)

    unique = [items[0] for items in all_candidates.values()]

    # Preserve the original train/validation/test provenance split.
    by_split: dict[str, list[dict]] = {name: [] for name in split_rows}
    source_split = {
        f"{normalize(row.get('act_id'))}:{normalize(row.get('section_number'))}:{row.get('text_sha256', '')}": split
        for split, rows in split_rows.items()
        for row in rows
    }

    for example in unique:
        split = source_split.get(example["source_id"])
        if split:
            by_split[split].append(example)

    # Cap the total dataset deterministically while preserving split order.
    total = sum(len(rows) for rows in by_split.values())
    if total > TARGET_TOTAL:
        selected = []
        remaining = TARGET_TOTAL
        for split in ("train", "validation", "test"):
            rows = by_split[split]
            take = min(len(rows), round(TARGET_TOTAL * len(rows) / total))
            take = min(take, remaining)
            selected.extend((split, row) for row in rows[:take])
            remaining -= take

        if remaining:
            for split in ("train", "validation", "test"):
                for row in by_split[split]:
                    if (split, row) not in selected:
                        selected.append((split, row))
                        remaining -= 1
                        if remaining == 0:
                            break
                if remaining == 0:
                    break

        by_split = {name: [] for name in split_rows}
        for split, row in selected[:TARGET_TOTAL]:
            by_split[split].append(row)

    for split, rows in by_split.items():
        rows.sort(key=lambda item: (item["source_id"], item["question_type"]))
        write_jsonl(OUTPUT_DIR / f"{split}.jsonl", rows)

    summary = {
        "method": "deterministic_template_from_source_provisions",
        "external_llm_used": False,
        "pretrained_model_used": False,
        "target_total": TARGET_TOTAL,
        "counts": {split: len(rows) for split, rows in by_split.items()},
        "total": sum(len(rows) for rows in by_split.values()),
    }

    write_jsonl(OUTPUT_DIR / "summary.jsonl", [summary])

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5A Deterministic QA Dataset")
    print("=" * 72)
    for split, rows in by_split.items():
        print(f"{split:>12}: {len(rows):,}")
    print(f"{'TOTAL':>12}: {summary['total']:,}")
    print(f"Output: {OUTPUT_DIR}")
    print("External LLM: no")
    print("Pretrained model: no")
    print("=" * 72)


if __name__ == "__main__":
    main()
