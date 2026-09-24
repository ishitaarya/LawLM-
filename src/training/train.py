"""Phase 5 training loop for LawSuit LLM."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.lawlm_model import LawSuitLLM
from src.training.causal_lm_dataset import build_split_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLOCK_SIZE = 256
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 4
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")


def build_model() -> LawSuitLLM:
    return LawSuitLLM(
        vocab_size=10_000, block_size=BLOCK_SIZE, embed_dim=384,
        num_heads=6, num_layers=4, ff_hidden_dim=1536, dropout=0.1,
    ).to(DEVICE)


def evaluate(model: LawSuitLLM, loader: DataLoader) -> float:
    model.eval()
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for input_ids, targets in loader:
            _, loss = model(input_ids.to(DEVICE), targets.to(DEVICE))
            if loss is None:
                raise RuntimeError("Validation loss was not produced.")
            total_loss += loss.item()
            batches += 1
    return total_loss / max(1, batches)


def save_checkpoint(model, optimizer, epoch, train_loss, validation_loss) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"lawsuit_llm_epoch_{epoch:02d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "config": {
            "vocab_size": 10_000, "block_size": BLOCK_SIZE, "embed_dim": 384,
            "num_heads": 6, "num_layers": 4, "ff_hidden_dim": 1536,
        },
    }, path)
    return path


def main() -> None:
    torch.manual_seed(42)

    train_dataset = build_split_dataset("train", block_size=BLOCK_SIZE)
    validation_dataset = build_split_dataset("validation", block_size=BLOCK_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = max(1, math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOG_DIR / "training_history.jsonl"

    print("=" * 60)
    print("LawSuit LLM - Phase 5 Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, batches = 0.0, 0

        for batch_index, (input_ids, targets) in enumerate(train_loader, start=1):
            _, loss = model(input_ids.to(DEVICE), targets.to(DEVICE))
            if loss is None:
                raise RuntimeError("Training loss was not produced.")

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            running_loss += loss.item()
            batches += 1

            if batch_index % GRADIENT_ACCUMULATION_STEPS == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        train_loss = running_loss / max(1, batches)
        validation_loss = evaluate(model, validation_loader)
        checkpoint = save_checkpoint(model, optimizer, epoch, train_loss, validation_loss)

        record = {
            "epoch": epoch, "step": global_step, "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "checkpoint": str(checkpoint),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\\n")

        print(f"Epoch {epoch}:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Validation loss: {validation_loss:.4f}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.8f}")
        print(f"  Checkpoint: {checkpoint}")

    print("=" * 60)
    print("Phase 5 training run completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
