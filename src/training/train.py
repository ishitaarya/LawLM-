"""Phase 7.5.5 training loop for the expanded LawSuit LLM corpus."""

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
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 4
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
CHECKPOINT_PREFIX = "lawsuit_llm_expanded"
HISTORY_PATH = LOG_DIR / "expanded_training_history.jsonl"


def build_model() -> LawSuitLLM:
    return LawSuitLLM(
        vocab_size=10_000,
        block_size=BLOCK_SIZE,
        embed_dim=384,
        num_heads=6,
        num_layers=4,
        ff_hidden_dim=1536,
        dropout=0.1,
    ).to(DEVICE)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def evaluate(model: LawSuitLLM, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for input_ids, targets in loader:
            _, loss = model(input_ids.to(DEVICE), targets.to(DEVICE))
            if loss is None:
                raise RuntimeError("Validation loss was not produced.")
            total_loss += loss.item()
            batches += 1
    return total_loss / max(1, batches)


def save_checkpoint(
    model: LawSuitLLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    train_loss: float,
    validation_loss: float,
    best_validation_loss: float,
    path: Path,
) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "config": {
                "vocab_size": 10_000,
                "block_size": BLOCK_SIZE,
                "embed_dim": 384,
                "num_heads": 6,
                "num_layers": 4,
                "ff_hidden_dim": 1536,
                "dropout": 0.1,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "dataset": "expanded_25k_source_14299_cleaned",
            },
        },
        path,
    )
    return path


def main() -> None:
    torch.manual_seed(42)

    train_dataset = build_split_dataset("train", block_size=BLOCK_SIZE)
    validation_dataset = build_split_dataset("validation", block_size=BLOCK_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = max(
        1, math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    )
    total_steps = steps_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("LawSuit LLM - Phase 7.5.5 Expanded-Corpus Training")
    print("=" * 68)
    print(f"Device: {DEVICE}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Train blocks: {len(train_dataset):,}")
    print(f"Validation blocks: {len(validation_dataset):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Optimizer steps/epoch: {steps_per_epoch:,}")
    print(f"Total optimizer steps: {total_steps:,}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("Initialization: random")
    print("Pretrained weights: no")
    print("=" * 68)

    best_validation_loss = float("inf")
    global_step = 0

    with HISTORY_PATH.open("w", encoding="utf-8") as history:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            batches = 0

            print(f"Epoch {epoch}/{EPOCHS} - {len(train_loader):,} batches")
            for batch_index, (input_ids, targets) in enumerate(train_loader, start=1):
                _, loss = model(input_ids.to(DEVICE), targets.to(DEVICE))
                if loss is None:
                    raise RuntimeError("Training loss was not produced.")

                (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                running_loss += loss.item()
                batches += 1

                if (
                    batch_index == 1
                    or batch_index % 100 == 0
                    or batch_index == len(train_loader)
                ):
                    average_loss = running_loss / batches
                    percent = 100.0 * batch_index / len(train_loader)
                    print(
                        f"  Batch {batch_index:,}/{len(train_loader):,} "
                        f"({percent:5.1f}%) | loss={loss.item():.4f} | avg={average_loss:.4f}",
                        flush=True,
                    )

                if (
                    batch_index % GRADIENT_ACCUMULATION_STEPS == 0
                    or batch_index == len(train_loader)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

            train_loss = running_loss / max(1, batches)
            validation_loss = evaluate(model, validation_loader)

            epoch_checkpoint = CHECKPOINT_DIR / (
                f"{CHECKPOINT_PREFIX}_epoch_{epoch:02d}.pt"
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                train_loss,
                validation_loss,
                best_validation_loss,
                epoch_checkpoint,
            )

            best_checkpoint = None
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_checkpoint = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_best.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    train_loss,
                    validation_loss,
                    best_validation_loss,
                    best_checkpoint,
                )

            record = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "train_perplexity": perplexity(train_loss),
                "validation_perplexity": perplexity(validation_loss),
                "epoch_checkpoint": str(epoch_checkpoint),
                "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()

            print(f"Epoch {epoch}:")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Validation loss: {validation_loss:.4f}")
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.8f}")
            print(f"  Train perplexity: {perplexity(train_loss):.2f}")
            print(f"  Validation perplexity: {perplexity(validation_loss):.2f}")
            print(f"  Checkpoint: {epoch_checkpoint}")
            if best_checkpoint:
                print(f"  New best checkpoint: {best_checkpoint}")

    print("=" * 68)
    print("Phase 7.5.5 expanded-corpus training completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
