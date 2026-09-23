import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS_DIR = Path("experiments")
GRAPHS_DIR = Path("graphs")


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_training_history():
    history = load_json(EXPERIMENTS_DIR / "experiment_10_epoch.json")["history"]

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Training Loss")
    plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LawLM Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "training_validation_loss.png", dpi=150)
    plt.close()


def plot_model_comparison():
    baseline_test_loss = 4.2221
    experiment_a_test_loss = 4.0694
    experiment_b_test_loss = 4.1749

    names = [
        "Baseline\n5 Epochs",
        "Experiment A\n10 Epochs",
        "Experiment B\nLarger Model",
    ]
    values = [
        baseline_test_loss,
        experiment_a_test_loss,
        experiment_b_test_loss,
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("Test Loss")
    plt.title("LawLM Test Loss Comparison")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "test_loss_comparison.png", dpi=150)
    plt.close()


def plot_perplexity_comparison():
    values = [68.17, 58.52, 65.03]

    names = [
        "Baseline\n5 Epochs",
        "Experiment A\n10 Epochs",
        "Experiment B\nLarger Model",
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("Perplexity")
    plt.title("LawLM Perplexity Comparison")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "perplexity_comparison.png", dpi=150)
    plt.close()


def plot_parameter_comparison():
    values = [5.24, 5.24, 10.00]

    names = [
        "Baseline\n5 Epochs",
        "Experiment A\n10 Epochs",
        "Experiment B\nLarger Model",
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("Parameters (Millions)")
    plt.title("LawLM Model Size Comparison")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "parameter_comparison.png", dpi=150)
    plt.close()


def main():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        EXPERIMENTS_DIR / "experiment_10_epoch.json",
        EXPERIMENTS_DIR / "experiment_c_comparison.json",
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required experiment file not found: {path}")

    plot_training_history()
    plot_model_comparison()
    plot_perplexity_comparison()
    plot_parameter_comparison()

    print("=" * 60)
    print("LawLM - Experiment Graphs")
    print("=" * 60)
    print("Generated:")
    print("  graphs/training_validation_loss.png")
    print("  graphs/test_loss_comparison.png")
    print("  graphs/perplexity_comparison.png")
    print("  graphs/parameter_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
