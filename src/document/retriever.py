"""
LawSuit LLM — Phase 6B Legal Chunk Retriever

Retrieves the most relevant legal chunks for a user query using TF-IDF
and cosine similarity. This first retrieval baseline uses no pretrained
embedding model.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path("data/documents/Indian_Contract_Act_1872_chunks.jsonl")
DEFAULT_TOP_K = 5

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "was", "were", "what", "when", "where", "which", "who",
    "with", "under", "this", "these", "those", "does", "do", "can",
}


def tokenize(text: str) -> list[str]:
    """Tokenize legal text into simple normalized terms."""
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", text.lower())
        if token not in STOPWORDS
    ]


def load_chunks(path: Path) -> list[dict]:
    """Load legal chunks from JSONL."""
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return chunks


def build_index(chunks: list[dict]):
    """Build an in-memory TF-IDF index."""
    document_frequency = Counter()
    term_frequencies = []

    for chunk in chunks:
        counts = Counter(tokenize(chunk["text"]))
        term_frequencies.append(counts)

        for term in counts:
            document_frequency[term] += 1

    total_documents = len(chunks)
    idf = {
        term: math.log((1 + total_documents) / (1 + frequency)) + 1.0
        for term, frequency in document_frequency.items()
    }

    vectors = []

    for counts in term_frequencies:
        vector = {}

        total_terms = sum(counts.values())

        if total_terms:
            for term, count in counts.items():
                vector[term] = (count / total_terms) * idf[term]

        vectors.append(vector)

    return idf, vectors


def cosine_similarity(query_vector: dict[str, float],
                      document_vector: dict[str, float]) -> float:
    """Calculate cosine similarity between sparse vectors."""
    if not query_vector or not document_vector:
        return 0.0

    dot_product = sum(
        value * document_vector.get(term, 0.0)
        for term, value in query_vector.items()
    )

    query_norm = math.sqrt(
        sum(value * value for value in query_vector.values())
    )

    document_norm = math.sqrt(
        sum(value * value for value in document_vector.values())
    )

    if query_norm == 0.0 or document_norm == 0.0:
        return 0.0

    return dot_product / (query_norm * document_norm)


def search(query: str, chunks: list[dict], idf: dict, vectors: list[dict],
           top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return top-k legal chunks for a query."""
    query_counts = Counter(tokenize(query))
    total_terms = sum(query_counts.values())

    query_vector = {}

    if total_terms:
        for term, count in query_counts.items():
            if term in idf:
                query_vector[term] = (
                    (count / total_terms) * idf[term]
                )

    results = []

    for chunk, vector in zip(chunks, vectors):
        score = cosine_similarity(query_vector, vector)

        if score > 0:
            result = dict(chunk)
            result["score"] = score
            results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


def print_results(query: str, results: list[dict]) -> None:
    """Print retrieval results in a readable format."""
    print("=" * 64)
    print("LawSuit LLM — Phase 6B Legal Retrieval")
    print("=" * 64)
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print()

    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] Score: {result['score']:.4f}")
        print(f"    Chunk:   {result['chunk_id']}")
        print(f"    Page:    {result['page_number']}")
        print(f"    Section: {result.get('section_number') or 'N/A'}")
        if result.get("section_title"):
            print(f"    Title:   {result['section_title']}")
        preview = re.sub(r"\s+", " ", result["text"])
        print(f"    Text:    {preview[:350]}")
        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrieve relevant legal chunks using TF-IDF."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the essentials of a valid contract?",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    if not DEFAULT_INPUT.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {DEFAULT_INPUT}"
        )

    chunks = load_chunks(DEFAULT_INPUT)

    if not chunks:
        raise RuntimeError("No legal chunks found.")

    idf, vectors = build_index(chunks)
    results = search(args.query, chunks, idf, vectors, args.top_k)

    print_results(args.query, results)


if __name__ == "__main__":
    main()
