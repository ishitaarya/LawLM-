"""
LawSuit LLM — Phase 6B Legal Chunk Retriever

Retrieves the most relevant legal chunks for a user query using sparse
TF-IDF-style retrieval. No pretrained embedding model is used.
"""

import argparse
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

LEGAL_EXPANSIONS = {
    "valid contract": [
        "valid", "contract", "agreement", "enforceable",
        "lawful", "consideration", "competent", "consent",
        "free consent", "competent to contract", "lawful object",
    ],
    "essentials of a valid contract": [
        "valid", "contract", "agreement", "enforceable",
        "consideration", "competent", "consent",
        "free consent", "lawful consideration", "lawful object",
    ],
    "requirements of a contract": [
        "contract", "agreement", "enforceable", "consideration",
        "competent", "consent", "lawful object",
    ],
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", text.lower())
        if token not in STOPWORDS
    ]


def make_terms(text: str) -> list[str]:
    """Create unigram and adjacent bigram terms."""
    tokens = tokenize(text)
    terms = list(tokens)
    terms.extend(
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:])
    )
    return terms


def expand_query(query: str) -> list[str]:
    """Add transparent legal terms for common natural-language queries."""
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    terms = make_terms(query)

    for phrase, expansion in LEGAL_EXPANSIONS.items():
        if phrase in normalized:
            terms.extend(expansion)

    return terms


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return chunks


def build_index(chunks: list[dict]):
    """Build sparse TF-IDF vectors from chunk text and metadata."""
    document_frequency = Counter()
    term_frequencies = []

    for chunk in chunks:
        searchable = " ".join(
            [
                chunk.get("text", ""),
                chunk.get("section_title") or "",
                str(chunk.get("section_number") or ""),
            ]
        )
        counts = Counter(make_terms(searchable))
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
        total_terms = sum(counts.values())
        vector = {}

        if total_terms:
            for term, count in counts.items():
                vector[term] = (count / total_terms) * idf[term]

        vectors.append(vector)

    return idf, vectors


def cosine_similarity(query_vector: dict[str, float],
                      document_vector: dict[str, float]) -> float:
    if not query_vector or not document_vector:
        return 0.0

    dot_product = sum(
        value * document_vector.get(term, 0.0)
        for term, value in query_vector.items()
    )

    query_norm = math.sqrt(sum(v * v for v in query_vector.values()))
    document_norm = math.sqrt(sum(v * v for v in document_vector.values()))

    if query_norm == 0.0 or document_norm == 0.0:
        return 0.0

    return dot_product / (query_norm * document_norm)


def search(query: str, chunks: list[dict], idf: dict[str, float],
           vectors: list[dict[str, float]],
           top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return ranked legal chunks."""
    query_terms = expand_query(query)
    query_counts = Counter(query_terms)
    total_terms = sum(query_counts.values())
    query_vector = {}

    if total_terms:
        for term, count in query_counts.items():
            if term in idf:
                query_vector[term] = (count / total_terms) * idf[term]

    normalized_query = re.sub(r"\s+", " ", query.lower()).strip()
    results = []

    for chunk, vector in zip(chunks, vectors):
        score = cosine_similarity(query_vector, vector)

        text = chunk.get("text", "").lower()
        title = (chunk.get("section_title") or "").lower()

        if normalized_query and normalized_query in text:
            score += 0.20

        if normalized_query and normalized_query in title:
            score += 0.25

        if "valid contract" in normalized_query:
            if any(
                phrase in text
                for phrase in (
                    "what agreements are contracts",
                    "lawful consideration and lawful object",
                    "competent to contract",
                    "free consent",
                    "lawful object",
                )
            ):
                score += 0.30

        if score > 0:
            result = dict(chunk)
            result["score"] = score
            results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def print_results(query: str, results: list[dict]) -> None:
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
    parser = argparse.ArgumentParser(
        description="Retrieve relevant legal chunks using sparse lexical retrieval."
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
        raise FileNotFoundError(f"Chunk file not found: {DEFAULT_INPUT}")

    chunks = load_chunks(DEFAULT_INPUT)
    if not chunks:
        raise RuntimeError("No legal chunks found.")

    idf, vectors = build_index(chunks)
    results = search(args.query, chunks, idf, vectors, args.top_k)
    print_results(args.query, results)


if __name__ == "__main__":
    main()
