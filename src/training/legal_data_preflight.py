"""Read-only preflight audit for the real legal source and QA split files.

This command checks expected inputs, file fingerprints, JSONL structure, QA
quality signals, duplicate questions, and section leakage across splits. It
never rewrites, filters, moves, or trains on source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.document.legal_source_audit import audit_pages, load_pages_with_parse_errors
from src.training.qa_dataset_audit import audit_records, load_jsonl

DEFAULT_PAGES = Path("data/documents/Indian_Contract_Act_1872.jsonl")
DEFAULT_PDF = Path("data/raw/Indian_Contract_Act_1872.pdf")
DEFAULT_TRAIN = Path("data/qa_v2/train.jsonl")
DEFAULT_VALIDATION = Path("data/qa_v2/validation.jsonl")
DEFAULT_TEST = Path("data/qa_v2/test.jsonl")


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"path": str(path), "size_bytes": size, "sha256": digest.hexdigest()}


def _normalise(value: Any) -> str:
    if isinstance(value, (str, int, float)):
        return " ".join(str(value).casefold().split())
    return ""


def _section_values(record: Any) -> set[str]:
    if not isinstance(record, dict):
        return set()
    for key in ("section", "section_number", "section_id", "legal_section"):
        value = record.get(key)
        if isinstance(value, list):
            values = {_normalise(item) for item in value}
            return {item for item in values if item}
        normalized = _normalise(value)
        if normalized:
            return {normalized}
    return set()


def _load_qa_split(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path), "error": "file_missing"}
    records, parse_errors = load_jsonl(path)
    audit = audit_records(records)
    audit["json_parse_error_count"] = len(parse_errors)
    audit["json_parse_errors"] = parse_errors
    if parse_errors:
        audit["structural_integrity_passed"] = False
    else:
        audit["structural_integrity_passed"] = (
            len(records) > 0
            and audit["records_requiring_review_count"] == 0
            and audit["structurally_valid_record_count"] == len(records)
        )
    return {
        "available": True,
        "fingerprint": _fingerprint(path),
        "record_count": len(records),
        "audit": audit,
        "_records": records,
    }


def run_preflight(
    pages_path: Path = DEFAULT_PAGES,
    pdf_path: Path = DEFAULT_PDF,
    train_path: Path = DEFAULT_TRAIN,
    validation_path: Path = DEFAULT_VALIDATION,
    test_path: Path = DEFAULT_TEST,
) -> dict[str, Any]:
    """Inspect the expected source and three QA splits without modifying them."""
    report: dict[str, Any] = {
        "report_type": "lawsuit_llm_real_data_preflight",
        "inputs": {},
        "cross_split_checks": {},
        "all_required_files_available": False,
        "structural_preflight_passed": False,
        "legal_accuracy_verified": False,
        "ready_for_training": False,
        "training_started": False,
        "source_data_modified": False,
        "note": (
            "This is a read-only structural preflight. A clean result does not "
            "verify legal accuracy, authoritative-source currency, OCR fidelity, "
            "human review, or permission to train."
        ),
    }

    required_paths = {
        "source_pdf": pdf_path,
        "source_pages_jsonl": pages_path,
        "qa_train": train_path,
        "qa_validation": validation_path,
        "qa_test": test_path,
    }
    report["inputs"]["source_pdf"] = (
        {"available": True, "fingerprint": _fingerprint(pdf_path)}
        if pdf_path.is_file()
        else {"available": False, "path": str(pdf_path), "error": "file_missing"}
    )

    if pages_path.is_file():
        try:
            pages, parse_errors = load_pages_with_parse_errors(pages_path)
            page_audit = audit_pages(pages)
            page_audit["json_parse_error_count"] = len(parse_errors)
            page_audit["json_parse_errors"] = parse_errors
            if parse_errors:
                page_audit["structural_page_integrity_passed"] = False
            report["inputs"]["source_pages_jsonl"] = {
                "available": True,
                "fingerprint": _fingerprint(pages_path),
                "audit": page_audit,
            }
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            report["inputs"]["source_pages_jsonl"] = {
                "available": True,
                "path": str(pages_path),
                "error": f"{type(exc).__name__}: {exc}",
                "structural_page_integrity_passed": False,
            }
    else:
        report["inputs"]["source_pages_jsonl"] = {
            "available": False, "path": str(pages_path), "error": "file_missing"
        }

    split_paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    split_reports: dict[str, dict[str, Any]] = {}
    for split, path in split_paths.items():
        result = _load_qa_split(path)
        split_reports[split] = result
        # Keep raw records internal to this calculation; never serialize them.
        report["inputs"][f"qa_{split}"] = {
            key: value for key, value in result.items() if key != "_records"
        }

    question_locations: dict[str, list[str]] = {}
    section_locations: dict[str, list[str]] = {}
    exact_record_locations: dict[str, list[str]] = {}
    for split, result in split_reports.items():
        for record in result.get("_records", []):
            if not isinstance(record, dict):
                continue
            question = _normalise(record.get("question"))
            if question:
                question_locations.setdefault(question, []).append(split)
            canonical = json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            exact_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            exact_record_locations.setdefault(exact_hash, []).append(split)
            for section in _section_values(record):
                section_locations.setdefault(section, []).append(split)

    def cross_split(items: dict[str, list[str]]) -> list[dict[str, Any]]:
        findings = []
        for value, locations in sorted(items.items()):
            unique = sorted(set(locations))
            if len(unique) > 1:
                findings.append({"value": value, "splits": unique})
        return findings

    duplicate_questions = cross_split(question_locations)
    duplicate_records = cross_split(exact_record_locations)
    section_overlap = cross_split(section_locations)
    report["cross_split_checks"] = {
        "duplicate_questions_across_splits": duplicate_questions,
        "duplicate_record_hashes_across_splits": duplicate_records,
        "sections_shared_across_splits": section_overlap,
        "duplicate_question_count": len(duplicate_questions),
        "duplicate_record_count": len(duplicate_records),
        "shared_section_count": len(section_overlap),
        "question_split_isolation_passed": not duplicate_questions,
        "exact_record_split_isolation_passed": not duplicate_records,
    }

    all_available = all(
        report["inputs"][key]["available"]
        for key in (
            "source_pdf", "source_pages_jsonl", "qa_train", "qa_validation", "qa_test"
        )
    )
    pages_ok = report["inputs"]["source_pages_jsonl"].get(
        "audit", {}
    ).get("structural_page_integrity_passed", False)
    splits_ok = all(
        split_reports[name].get("available")
        and split_reports[name].get("audit", {}).get("structural_integrity_passed", False)
        for name in split_paths
    )
    report["all_required_files_available"] = all_available
    report["structural_preflight_passed"] = bool(
        all_available
        and pages_ok
        and splits_ok
        and not duplicate_questions
        and not duplicate_records
    )
    # Section overlap is reported for review, not automatically treated as a
    # failure: a legitimate legal section may support examples in multiple splits.
    report["shared_section_overlap_requires_review"] = bool(section_overlap)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_preflight(
        pages_path=args.pages,
        pdf_path=args.pdf,
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is None:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(f"Read-only data preflight report created: {args.output}")


if __name__ == "__main__":
    main()
