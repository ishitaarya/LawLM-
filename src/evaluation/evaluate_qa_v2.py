"""Evaluate QA V2 loss and inspect deterministic or sampled free-generation outputs.

Greedy decoding is the default so repeated runs are comparable. Sampling remains
available with --strategy sample for exploratory inspection. This evaluation is
diagnostic: token loss alone does not establish legal answer correctness.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sentencepiece as spm
import torch

from src.model.lawlm_model import LawLM
from src.training.train_qa_v2 import QADataset

CKPT = Path("checkpoints/lawsuit_llm_qa_v2_best.pt")
TEST = Path("data/qa_v2/test.jsonl")
TOK = Path("data/tokenizer/lawsuit_bpe.model")
EOS_ID = 3


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))


@torch.no_grad()
def generate_greedy(model, prompt: torch.Tensor, max_new_tokens: int, eos_token_id: int):
    """Greedy decoding using the model's existing forward pass."""
    output = prompt
    for _ in range(max_new_tokens):
        context = output[:, -model.block_size :]
        logits, _ = model(context)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        output = torch.cat((output, next_token), dim=1)
        if torch.all(next_token == eos_token_id):
            break
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate LawLM QA V2.")
    parser.add_argument("--checkpoint", default=str(CKPT))
    parser.add_argument("--test", default=str(TEST))
    parser.add_argument("--tokenizer", default=str(TOK))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--strategy",
        choices=("greedy", "sample"),
        default="greedy",
        help="greedy is deterministic; sample uses temperature and top-k sampling",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output", default="logs/qa_v2_evaluation.json")
    args = parser.parse_args()

    if args.count < 0 or args.max_new_tokens < 1:
        parser.error("--count must be >= 0 and --max-new-tokens must be >= 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = LawLM(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_hidden_dim=config["ff_hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = QADataset(Path(args.test))
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)

    total_loss = 0.0
    with torch.no_grad():
        for row in dataset:
            input_ids = row["input_ids"].unsqueeze(0).to(device)
            labels = row["labels"].unsqueeze(0).to(device)
            _, loss = model(input_ids, targets=labels)
            total_loss += loss.detach().item()

    mean_loss = total_loss / max(1, len(dataset))
    samples = []

    for index in range(min(args.count, len(dataset))):
        row = dataset.rows[index]
        prompt_ids = row["input_ids"][: row["prompt_length"]]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        if args.strategy == "greedy":
            generated_ids = generate_greedy(
                model, prompt, args.max_new_tokens, EOS_ID
            )[0].tolist()
        else:
            generated_ids = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=1.15,
                eos_token_id=EOS_ID,
            )[0].tolist()

        new_tokens = generated_ids[len(prompt_ids) :]
        generated_text = tokenizer.decode(new_tokens).strip()
        samples.append(
            {
                "index": index,
                "question": row["question"],
                "expected": row["answer"],
                "generated": generated_text,
                "generated_ids": new_tokens,
                "first_generated_id": new_tokens[0] if new_tokens else None,
                "strategy": args.strategy,
                "prompt_length": len(prompt_ids),
                "answer_truncation": row.get("answer_truncation"),
            }
        )

    report = {
        "phase": "7.5G",
        "checkpoint": str(args.checkpoint),
        "test_examples": len(dataset),
        "test_loss": mean_loss,
        "test_perplexity": perplexity(mean_loss),
        "generation_strategy": args.strategy,
        "samples": samples,
        "pretrained_llm": False,
        "note": "Generation samples require qualitative legal review; loss/perplexity do not prove legal correctness.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("QA V2 EVALUATION")
    print("=" * 72)
    print(
        f"Test: {len(dataset)} | Loss: {mean_loss:.4f} | "
        f"PPL: {perplexity(mean_loss):.2f} | Strategy: {args.strategy}"
    )
    for sample in samples:
        print(f"\n#{sample['index']} Question: {sample['question']}")
        print("Expected:", sample["expected"])
        print("Generated:", sample["generated"])
    print("Report:", output_path)


if __name__ == "__main__":
    main()
