"""
LawSuit LLM — Phase 6C End-to-End Legal QA Pipeline

Pipeline:
question -> retrieval -> context building -> from-scratch LawSuit LLM generation.

This is an educational/research pipeline, not legal advice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from src.document.context_builder import build_context
from src.document.retriever import build_index, load_chunks, search
from src.generation.generate import load_model

DEFAULT_CHUNKS = Path("data/documents/Indian_Contract_Act_1872_chunks.jsonl")
DEFAULT_CHECKPOINT = Path("checkpoints/lawsuit_llm_epoch_03.pt")
DEFAULT_TOKENIZER = Path("data/tokenizer/lawsuit_bpe.model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_answer_prompt(query: str, context: str, tokenizer: spm.SentencePieceProcessor) -> str:
    """Build a compact prompt while keeping the question and instruction visible."""
    instruction = (
        "\n\nQUESTION\n"
        f"{query.strip()}"
        "\n\nANSWER USING ONLY THE RETRIEVED CONTEXT:\n"
    )

    context_ids = tokenizer.encode(context, out_type=int)
    instruction_ids = tokenizer.encode(instruction, out_type=int)

    available = 240 - len(instruction_ids)
    if available <= 0:
        raise ValueError("Question/instruction is too long for the model context.")

    compact_context = tokenizer.decode(context_ids[-available:])
    return compact_context + instruction


def answer_question(
    query: str,
    top_k: int = 3,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> tuple[str, list[dict], str]:
    """Run retrieval, context construction, and LLM generation."""
    if not DEFAULT_CHUNKS.exists():
        raise FileNotFoundError(f"Chunk file not found: {DEFAULT_CHUNKS}")
    if not DEFAULT_TOKENIZER.exists():
        raise FileNotFoundError(f"Tokenizer not found: {DEFAULT_TOKENIZER}")

    chunks = load_chunks(DEFAULT_CHUNKS)
    if not chunks:
        raise RuntimeError("No legal chunks found.")

    idf, vectors = build_index(chunks)
    results = search(query, chunks, idf, vectors, top_k=top_k)

    context = build_context(query, results, max_chunks=top_k)

    tokenizer = spm.SentencePieceProcessor(model_file=str(DEFAULT_TOKENIZER))
    prompt = build_answer_prompt(query, context, tokenizer)

    model = load_model(DEFAULT_CHECKPOINT)
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=40,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_id(),
    )

    generated_ids = generated[0].tolist()
    new_ids = generated_ids[len(ids):]
    answer = tokenizer.decode(new_ids).strip()

    return answer, results, context


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LawSuit LLM Phase 6C legal QA pipeline."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="When is an agreement without consideration valid?",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    answer, results, context = answer_question(
        args.query,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("=" * 72)
    print("LawSuit LLM — Phase 6C End-to-End Legal QA")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Question: {args.query}")
    print()
    print("Retrieved sections:")
    for rank, result in enumerate(results, start=1):
        print(
            f"  [{rank}] Section {result.get('section_number', 'N/A')} "
            f"| page {result.get('page_number', 'N/A')} "
            f"| score {result.get('score', 0.0):.4f}"
        )
    print()
    print("Generated answer:")
    print(answer or "[No generated text]")
    print()
    print("Research/educational output only; not legal advice.")
    print("=" * 72)


if __name__ == "__main__":
    main()
