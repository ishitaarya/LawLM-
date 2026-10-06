"""Phase 7.5D — CPU-friendly supervised QA fine-tuning.

Starts from the project's own from-scratch legal-language checkpoint. No
external pretrained LLM is used. Loss is computed only on answer tokens
(label values != -100).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.model.lawlm_model import LawLM

DATA_DIR = Path("data/qa_chunked")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")

BASE_CHECKPOINT = CHECKPOINT_DIR / "lawsuit_llm_expanded_best.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "lawsuit_llm_qa_best.pt"
HISTORY_FILE = LOG_DIR / "qa_training_history.jsonl"

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
BLOCK_SIZE = 256
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class QADataset(Dataset):
    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as file:
            self.rows = [json.loads(line) for line in file if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)

    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[i, :length] = item["input_ids"]
        labels[i, :length] = item["labels"]
        attention_mask[i, :length] = item["attention_mask"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def load_base_checkpoint(device: torch.device) -> tuple[LawLM, dict]:
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {BASE_CHECKPOINT}"
        )

    checkpoint = torch.load(BASE_CHECKPOINT, map_location=device)

    model = LawLM(
        vocab_size=10000,
        block_size=BLOCK_SIZE,
        embed_dim=384,
        num_layers=4,
        num_heads=6,
        ff_hidden_dim=1536,
        dropout=0.1,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state_dict/model")

    model.load_state_dict(state_dict)
    return model, checkpoint


def evaluate(
    model: LawLM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(input_ids, targets=labels)
            if loss is None:
                raise RuntimeError("Model returned no loss during QA evaluation.")

            total_loss += float(loss.item())
            batches += 1

    return total_loss / max(batches, 1)


def save_checkpoint(
    path: Path,
    model: LawLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "model_config": {
                "vocab_size": 10000,
                "block_size": BLOCK_SIZE,
                "n_embd": 384,
                "n_layer": 4,
                "n_head": 6,
                "ffn_dim": 1536,
                "dropout": 0.1,
            },
            "training": {
                "task": "legal QA supervised fine-tuning",
                "initialization": "from project's own from-scratch legal LM",
                "external_pretrained_llm": False,
                "answer_only_loss": True,
            },
        },
        path,
    )


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = QADataset(DATA_DIR / "train.jsonl")
    validation_dataset = QADataset(DATA_DIR / "validation.jsonl")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    model, base_checkpoint = load_base_checkpoint(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5D QA Fine-Tuning")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Base checkpoint: {BASE_CHECKPOINT}")
    print(f"Base training epoch: {base_checkpoint.get('epoch', 'unknown')}")
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("Loss: answer tokens only")
    print("External pretrained LLM: no")
    print("=" * 72)

    best_validation = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        optimizer_steps = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(input_ids, targets=labels)
            if loss is None:
                raise RuntimeError("Model returned no loss for QA training.")

            loss_for_backward = loss / GRADIENT_ACCUMULATION
            loss_for_backward.backward()
            running_loss += float(loss.item())

            if (
                batch_index % GRADIENT_ACCUMULATION == 0
                or batch_index == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            if batch_index % 1000 == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_index:,}/{len(train_loader):,} | "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = running_loss / max(len(train_loader), 1)
        validation_loss = evaluate(model, validation_loader, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_perplexity": perplexity(train_loss),
            "validation_perplexity": perplexity(validation_loss),
            "optimizer_steps": optimizer_steps,
            "learning_rate": LEARNING_RATE,
            "device": str(device),
            "checkpoint": str(BEST_CHECKPOINT),
        }

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        print(
            f"Epoch {epoch} complete | "
            f"Train loss {train_loss:.4f} | "
            f"Validation loss {validation_loss:.4f} | "
            f"Validation PPL {perplexity(validation_loss):.2f}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                epoch,
                validation_loss,
            )
            print(f"New best QA checkpoint: {BEST_CHECKPOINT}")

    print("=" * 72)
    print("QA fine-tuning complete")
    print(f"Best validation loss: {best_validation:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT}")
    print(f"History: {HISTORY_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
