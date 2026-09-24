"""
LawSuit LLM — Phase 6B Legal Text Chunker

Converts page-level PDF JSONL into retrieval-friendly legal chunks while
preserving legal section boundaries.
"""

import json
import re
from pathlib import Path

DEFAULT_INPUT = Path("data/documents/Indian_Contract_Act_1872.jsonl")
DEFAULT_OUTPUT = Path("data/documents/Indian_Contract_Act_1872_chunks.jsonl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

SECTION_PATTERN = re.compile(
    r"(?m)(?:^|\n)\s*(\d+[A-Z]?)\.\s+(.+?)(?:\s*[—–-]\s*)"
)


def clean_text(text: str) -> str:
    """Normalize extracted PDF text."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_section_headings(text: str) -> list[re.Match]:
    """Find reliable legal section headings in body text."""
    return list(SECTION_PATTERN.finditer(text))


def split_into_sections(
    text: str,
    current_section_number: str | None,
    current_section_title: str | None,
) -> list[tuple[str | None, str | None, str]]:
    """
    Split page text at legal section headings.

    This prevents a 1200-character chunk from crossing section boundaries
    without preserving the correct section metadata.
    """
    matches = find_section_headings(text)

    if not matches:
        return [(current_section_number, current_section_title, text)]

    segments = []
    cursor = 0
    active_number = current_section_number
    active_title = current_section_title

    # Text before the first heading belongs to the section carried from
    # the previous page, if one exists.
    prefix = text[:matches[0].start()].strip()
    if prefix:
        segments.append((active_number, active_title, prefix))

    for index, match in enumerate(matches):
        active_number = match.group(1)
        active_title = match.group(2).strip()

        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment_text = text[start:end].strip()

        if segment_text:
            segments.append((active_number, active_title, segment_text))

        cursor = end

    return segments


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
    """Create retrieval-friendly chunks while preserving section boundaries."""
    chunks = []
    chunk_counter = 1
    current_section_number = None
    current_section_title = None

    for page in pages:
        source_file = page.get("source_file", "")
        page_number = page.get("page_number")
        text = clean_text(page.get("text", ""))

        if not text:
            continue

        sections = split_into_sections(
            text,
            current_section_number,
            current_section_title,
        )

        for section_number, section_title, section_text in sections:
            if section_number:
                current_section_number = section_number
                current_section_title = section_title

            for chunk_text in split_text(section_text):
                chunks.append(
                    {
                        "chunk_id": f"contract_{chunk_counter:05d}",
                        "source_file": source_file,
                        "page_number": page_number,
                        "section_number": section_number or current_section_number,
                        "section_title": section_title or current_section_title,
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
