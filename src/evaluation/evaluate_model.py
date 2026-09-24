"""Phase 7.1 model evaluation for LawSuit LLM.

Evaluates a saved checkpoint on the held-out test split and writes a compact
JSON report containing loss, perplexity, parameter count, and run metadata.
"""
from __future__ import annotations
import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.model.lawlm_model import LawSuitLLM
from src.training.causal_lm_dataset import build_split_dataset

def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))

def evaluate(model: LawSuitLLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for input_ids, targets in loader:
            _, loss = model(input_ids.to(device), targets.to(device))
            if loss is None:
                raise RuntimeError("Evaluation loss was not produced.")
            total_loss += loss.item()
            batches += 1
    return total_loss / max(1, batches)

def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = LawSuitLLM(
        vocab_size=int(config.get("vocab_size", 10_000)),
        block_size=int(config.get("block_size", 256)),
        embed_dim=int(config.get("embed_dim", 384)),
        num_heads=int(config.get("num_heads", 6)),
        num_layers=int(config.get("num_layers", 4)),
        ff_hidden_dim=int(config.get("ff_hidden_dim", 1536)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LawSuit LLM checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/lawsuit_llm_epoch_03.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--output", default="logs/evaluation_test.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    dataset = build_split_dataset("test", block_size=args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    test_loss = evaluate(model, loader, device)
    report = {
        "phase": "7.1", "split": "test", "checkpoint": str(checkpoint_path),
        "device": str(device), "parameters": model.parameter_count(),
        "test_samples": len(dataset), "block_size": args.block_size,
        "test_loss": test_loss, "test_perplexity": perplexity(test_loss),
        "training_epoch": checkpoint.get("epoch"),
        "training_validation_loss": checkpoint.get("validation_loss"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("=" * 60)
    print("LawSuit LLM - Phase 7.1 Model Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Test samples: {len(dataset):,}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {perplexity(test_loss):.2f}")
    print(f"Report: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
