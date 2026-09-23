import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lawlm_model import LawLM


TEST_FILE = Path("data/splits/test_data.pt")
CHECKPOINT_FILE = Path("checkpoints/best.pt")

BATCH_SIZE = 8
DEVICE = torch.device("cpu")


def load_test_data():
    data = torch.load(
        TEST_FILE,
        map_location="cpu",
        weights_only=True,
    )

    dataset = TensorDataset(
        data["inputs"],
        data["targets"],
    )

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


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_FILE}")

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_FILE}"
        )

    test_dataset, test_data = load_test_data()

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    model = LawLM(
        vocab_size=test_data["vocab_size"],
        block_size=test_data["block_size"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(model, loader)
    perplexity = math.exp(test_loss)

    print("=" * 60)
    print("LawLM - Model Evaluation")
    print("=" * 60)
    print(f"Test sequences:       {len(test_dataset):,}")
    print(f"Best training epoch:  {checkpoint['epoch']}")
    print(f"Training loss:        {checkpoint['train_loss']:.4f}")
    print(f"Validation loss:      {checkpoint['val_loss']:.4f}")
    print(f"Test loss:            {test_loss:.4f}")
    print(f"Test perplexity:      {perplexity:.2f}")
    print()
    print("Evaluation completed.")
    print("Disclaimer: This evaluation measures language-model")
    print("performance on the project dataset. It does not")
    print("measure legal correctness or provide legal advice.")
    print("=" * 60)


if __name__ == "__main__":
    main()
