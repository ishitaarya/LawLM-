"""
LawSuit LLM — Phase 6B Legal Text Chunker

Converts page-level PDF JSONL into retrieval-friendly legal chunks.
"""

import json
import re
from pathlib import Path

DEFAULT_INPUT = Path("data/documents/Indian_Contract_Act_1872.jsonl")
DEFAULT_OUTPUT = Path("data/documents/Indian_Contract_Act_1872_chunks.jsonl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

SECTION_PATTERN = re.compile(
    r"(?m)(?:^|\n)\s*(\d+[A-Z]?)\.\s+([^\n]+)"
)


def clean_text(text: str) -> str:
    """Normalize extracted PDF text."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str) -> tuple[str | None, str | None]:
    """Detect the last legal section heading present in a text fragment."""
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        return None, None

    match = matches[-1]
    return match.group(1), match.group(2).strip()


def split_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def load_pages(path: Path) -> list[dict]:
    """Load page records from JSONL."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def create_chunks(pages: list[dict]) -> list[dict]:
    """Create retrieval-friendly chunks with chunk-level section metadata."""
    chunks = []
    chunk_counter = 1

    for page in pages:
        source_file = page.get("source_file", "")
        page_number = page.get("page_number")
        raw_text = page.get("text", "")
        text = clean_text(raw_text)

        if not text:
            continue

        page_section_number, page_section_title = detect_section(text)

        for chunk_text in split_text(text):
            chunk_section_number, chunk_section_title = detect_section(chunk_text)

            # Prefer a section heading actually present in this chunk.
            section_number = chunk_section_number or page_section_number
            section_title = chunk_section_title or page_section_title

            chunks.append(
                {
                    "chunk_id": f"contract_{chunk_counter:05d}",
                    "source_file": source_file,
                    "page_number": page_number,
                    "section_number": section_number,
                    "section_title": section_title,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                }
            )
            chunk_counter += 1

    return chunks


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Save chunks as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    input_path = DEFAULT_INPUT
    output_path = DEFAULT_OUTPUT

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("=" * 64)
    print("LawSuit LLM — Phase 6B Legal Text Chunking")
    print("=" * 64)

    pages = load_pages(input_path)
    print(f"Input pages:       {len(pages):,}")

    chunks = create_chunks(pages)
    save_chunks(chunks, output_path)

    total_chars = sum(chunk["character_count"] for chunk in chunks)
    average_chars = total_chars / len(chunks) if chunks else 0

    print(f"Chunks created:    {len(chunks):,}")
    print(f"Characters:        {total_chars:,}")
    print(f"Average chunk:     {average_chars:.1f}")
    print(f"Chunk size target: {CHUNK_SIZE}")
    print(f"Chunk overlap:     {CHUNK_OVERLAP}")
    print(f"Output:            {output_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
