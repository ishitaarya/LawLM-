"""Phase 6B: extract text and page metadata from legal PDF documents.

The loader preserves page boundaries so later retrieval can cite source pages.
It does not interpret the document or provide legal advice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader


def extract_pdf(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must have a .pdf extension.")

    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "character_count": len(text),
            }
        )

    return pages


def save_pages(pages: list[dict], output_path: Path, source_pdf: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for page in pages:
            record = {"source_file": source_pdf.name, **page}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a legal PDF while preserving page numbers."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/documents/<pdf-name>.jsonl)",
    )
    args = parser.parse_args()

    output = args.output or Path("data/documents") / f"{args.pdf.stem}.jsonl"
    pages = extract_pdf(args.pdf)
    save_pages(pages, output, args.pdf)

    non_empty = sum(bool(page["text"]) for page in pages)
    characters = sum(page["character_count"] for page in pages)

    print("=" * 64)
    print("LawSuit LLM — Phase 6B PDF Text Extraction")
    print("=" * 64)
    print(f"PDF:              {args.pdf}")
    print(f"Pages:            {len(pages):,}")
    print(f"Pages with text:  {non_empty:,}")
    print(f"Characters:       {characters:,}")
    print(f"Output:           {output}")
    print("=" * 64)


if __name__ == "__main__":
    main()
