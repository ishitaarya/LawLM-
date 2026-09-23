import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lawlm_model import LawLM


TRAIN_FILE = Path("data/splits/train_data.pt")
VAL_FILE = Path("data/splits/validation_data.pt")
EXPERIMENT_DIR = Path("experiments")

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DEVICE = torch.device("cpu")

# Experiment B: moderate increase in model capacity.
EMBED_DIM = 320
NUM_HEADS = 5
NUM_LAYERS = 6
FF_HIDDEN_DIM = 1280
DROPOUT = 0.1


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
            _, loss = model(inputs.to(DEVICE), targets.to(DEVICE))
            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    train_dataset, train_data = load_split(TRAIN_FILE)
    val_dataset, _ = load_split(VAL_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = LawLM(
        vocab_size=train_data["vocab_size"],
        block_size=train_data["block_size"],
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_hidden_dim=FF_HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")

    print("=" * 60)
    print("LawLM - Experiment B: Larger Model")
    print("=" * 60)
    print(f"Layers:       {NUM_LAYERS}")
    print(f"Embedding:    {EMBED_DIM}")
    print(f"Attention:    {NUM_HEADS} heads")
    print(f"FFN hidden:   {FF_HIDDEN_DIM}")
    print(f"Parameters:   {parameter_count:,}")
    print(f"Epochs:       {EPOCHS}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for inputs, targets in progress:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            _, loss = model(inputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            total_loss += loss.item()
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(batches, 1)
        val_loss = evaluate(model, val_loader)
        elapsed = time.time() - start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_seconds": elapsed,
            }
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "vocab_size": train_data["vocab_size"],
                    "block_size": train_data["block_size"],
                    "embed_dim": EMBED_DIM,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "ff_hidden_dim": FF_HIDDEN_DIM,
                    "dropout": DROPOUT,
                },
                EXPERIMENT_DIR / "experiment_larger_best.pt",
            )

    output = {
        "experiment": "larger_model",
        "parameters": parameter_count,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ff_hidden_dim": FF_HIDDEN_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_validation_loss": best_val_loss,
        "history": history,
    }

    with (EXPERIMENT_DIR / "experiment_larger_model.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(output, file, indent=2)

    print()
    print("=" * 60)
    print("Experiment B completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Parameters: {parameter_count:,}")
    print("Saved: experiments/experiment_larger_best.pt")
    print("Saved: experiments/experiment_larger_model.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
