import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Allow this file to run directly from the project root:
# python src/training/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lawlm_model import LawLM


TRAIN_FILE = Path("data/splits/train_data.pt")
VAL_FILE = Path("data/splits/validation_data.pt")
CHECKPOINT_DIR = Path("checkpoints")

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

DEVICE = torch.device("cpu")


def load_split(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    dataset = TensorDataset(data["inputs"], data["targets"])
    return dataset, data


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            _, loss = model(inputs, targets)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")

    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation data not found: {VAL_FILE}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LawLM - Training")
    print("=" * 60)
    print(f"Device:        {DEVICE}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Epochs:        {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay:  {WEIGHT_DECAY}")
    print()

    train_dataset, train_data = load_split(TRAIN_FILE)
    val_dataset, _ = load_split(VAL_FILE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LawLM(
        vocab_size=train_data["vocab_size"],
        block_size=train_data["block_size"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    history = []

    print(f"Training sequences:   {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        total_loss = 0.0
        total_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for inputs, targets in progress:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(total_batches, 1)
        val_loss = evaluate(model, val_loader)
        elapsed = time.time() - start_time

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_seconds": elapsed,
            }
        )

        print()
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"time={elapsed:.1f}s"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            CHECKPOINT_DIR / "latest.pt",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                CHECKPOINT_DIR / "best.pt",
            )
            print("New best model saved: checkpoints/best.pt")

    history_path = Path("experiments/training_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print()
    print("=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Latest checkpoint:   {CHECKPOINT_DIR / 'latest.pt'}")
    print(f"Best checkpoint:     {CHECKPOINT_DIR / 'best.pt'}")
    print(f"History:             {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
