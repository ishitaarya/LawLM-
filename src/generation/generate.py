"""Phase 6 generation: load the trained LawSuit LLM and generate legal-style text."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from src.model.lawlm_model import LawSuitLLM

CHECKPOINT = Path("checkpoints/lawsuit_llm_epoch_03.pt")
TOKENIZER = Path("data/tokenizer/lawsuit_bpe.model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint: Path) -> LawSuitLLM:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = LawSuitLLM(
        vocab_size=10_000,
        block_size=256,
        embed_dim=384,
        num_heads=6,
        num_layers=4,
        ff_hidden_dim=1536,
        dropout=0.1,
    )
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def generate_text(
    model: LawSuitLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 40,
    repetition_penalty: float = 1.15,
) -> str:
    ids = tokenizer.encode(prompt, out_type=int)
    if not ids:
        raise ValueError("Prompt produced no tokens.")

    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_id(),
    )
    return tokenizer.decode(generated[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with LawSuit LLM")
    parser.add_argument(
        "--prompt",
        default="The court may grant relief under the applicable law because",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    if not TOKENIZER.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    model = load_model(args.checkpoint)
    output = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.repetition_penalty,
    )

    print("=" * 60)
    print("LawSuit LLM - Phase 6 Text Generation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print("=" * 60)
    print("Prompt:")
    print(args.prompt)
    print("
Generated text:")
    print(output)
    print("=" * 60)
    print("Research/educational output only; not legal advice.")


if __name__ == "__main__":
    main()
