"""Phase 7.5E — Evaluate the QA-fine-tuned LawSuit LLM.

Evaluates the held-out chunked QA test split and runs deterministic generation
checks on representative test questions. No external or pretrained LLM is used.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from src.model.lawlm_model import LawLM
from src.training.train_qa import QADataset, collate


DEFAULT_CHECKPOINT = Path("checkpoints/lawsuit_llm_qa_best.pt")
DEFAULT_TEST = Path("data/qa_chunked/test.jsonl")
DEFAULT_TOKENIZER = Path("data/tokenizer/lawsuit_bpe.model")
DEFAULT_OUTPUT = Path("logs/qa_evaluation_test.json")

BLOCK_SIZE = 256
EOS_ID = 3


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def load_checkpoint(path: Path, device: torch.device) -> tuple[LawLM, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = LawLM(
        vocab_size=int(config.get("vocab_size", 10_000)),
        block_size=int(config.get("block_size", BLOCK_SIZE)),
        embed_dim=int(config.get("embed_dim", config.get("n_embd", 384))),
        num_heads=int(config.get("num_heads", config.get("n_head", 6))),
        num_layers=int(config.get("num_layers", config.get("n_layer", 4))),
        ff_hidden_dim=int(config.get("ff_hidden_dim", config.get("ffn_dim", 1536))),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate_loss(model: LawLM, loader: DataLoader, device: torch.device) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    batches = 0
    supervised_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(input_ids, targets=labels)
            if loss is None:
                raise RuntimeError("Model returned no loss during QA evaluation.")
            total_loss += float(loss.item())
            supervised_tokens += int((labels != -100).sum().item())
            batches += 1

    return total_loss / max(batches, 1), supervised_tokens


def generation_prompt(question: str) -> str:
    return f"QUESTION:\n{question}\n\nANSWER:\n"


def decode_generation(model, tokenizer, question, device, max_new_tokens, temperature, top_k) -> str:
    prompt_ids = tokenizer.encode(generation_prompt(question), out_type=int)
    if len(prompt_ids) >= BLOCK_SIZE:
        prompt_ids = prompt_ids[: BLOCK_SIZE - 1]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=1.15,
        eos_token_id=EOS_ID,
    )
    answer_ids = generated[0].tolist()[len(prompt_ids):]
    return tokenizer.decode(answer_ids).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LawSuit LLM QA fine-tuning.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--generation-count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    test_path = Path(args.test)
    tokenizer_path = Path(args.tokenizer)

    for path in (checkpoint_path, test_path, tokenizer_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    dataset = QADataset(test_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    test_loss, supervised_tokens = evaluate_loss(model, loader, device)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    generation_rows = []
    model.eval()
    for index in range(min(args.generation_count, len(dataset))):
        row = dataset.rows[index]
        generated = decode_generation(model, tokenizer, row["question"], device, args.max_new_tokens, args.temperature, args.top_k)
        generation_rows.append({
            "test_index": index,
            "question": row["question"],
            "expected_answer": row["answer"],
            "section": row["section"],
            "act_id": row["act_id"],
            "generated_answer": generated,
        })

    report = {
        "phase": "7.5E",
        "task": "QA fine-tuned model evaluation",
        "split": "test",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "parameters": model.parameter_count(),
        "test_examples": len(dataset),
        "supervised_answer_tokens": supervised_tokens,
        "test_loss": test_loss,
        "test_perplexity": perplexity(test_loss),
        "training_epoch": checkpoint.get("epoch"),
        "training_validation_loss": checkpoint.get("validation_loss"),
        "training_validation_perplexity": perplexity(float(checkpoint["validation_loss"])) if checkpoint.get("validation_loss") is not None else None,
        "generation_settings": {
            "count": len(generation_rows),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": 1.15,
        },
        "generation_samples": generation_rows,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_pretrained_llm": False,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5E QA Evaluation")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Test examples: {len(dataset):,}")
    print(f"Supervised answer tokens: {supervised_tokens:,}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {perplexity(test_loss):.2f}")
    print(f"Training validation loss: {checkpoint.get('validation_loss')}")
    print(f"Generation samples: {len(generation_rows)}")
    print(f"Report: {output_path}")
    print("=" * 72)

    for sample in generation_rows:
        print("\n" + "-" * 72)
        print(f"TEST #{sample['test_index']} | Section {sample['section']}")
        print(f"Question: {sample['question']}")
        print(f"Expected: {sample['expected_answer']}")
        print(f"Generated: {sample['generated_answer']}")


if __name__ == "__main__":
    main()
