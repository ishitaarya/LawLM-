"""Phase 7.2 generation benchmark for the expanded LawSuit LLM."""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm
import torch

from src.model.lawlm_model import LawSuitLLM

CHECKPOINT = Path("checkpoints/lawsuit_llm_expanded_best.pt")
TOKENIZER = Path("data/tokenizer/lawsuit_bpe.model")
OUTPUT = Path("logs/expanded_generation_benchmark.json")

PROMPTS = [
    "A valid contract requires",
    "An agreement without consideration is",
    "Free consent means that",
    "The court may grant relief when",
    "A person is liable under the law if",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> LawSuitLLM:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    config = state.get("config", {})
    model = LawSuitLLM(
        vocab_size=int(config.get("vocab_size", 10_000)),
        block_size=int(config.get("block_size", 256)),
        embed_dim=int(config.get("embed_dim", 384)),
        num_heads=int(config.get("num_heads", 6)),
        num_layers=int(config.get("num_layers", 4)),
        ff_hidden_dim=int(config.get("ff_hidden_dim", 1536)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def generate(
    model: LawSuitLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
) -> str:
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=80,
            temperature=0.7,
            top_k=40,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_id(),
        )
    return tokenizer.decode(generated[0].tolist())


def main() -> None:
    if not TOKENIZER.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    model = load_model()

    results = []
    print("=" * 68)
    print("LawSuit LLM - Phase 7.2 Expanded Generation Benchmark")
    print("=" * 68)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Prompts: {len(PROMPTS)}")
    print("=" * 68)

    for index, prompt in enumerate(PROMPTS, start=1):
        output = generate(model, tokenizer, prompt)
        results.append({"id": index, "prompt": prompt, "output": output})
        print(f"\n[{index}] Prompt: {prompt}")
        print(f"Generated: {output}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "phase": "7.2",
                "checkpoint": str(CHECKPOINT),
                "parameters": model.parameter_count(),
                "device": str(DEVICE),
                "generation": results,
                "note": "Qualitative generation benchmark; outputs are not legal advice.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 68)
    print(f"Saved: {OUTPUT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
