import math
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lawlm_model import LawLM


TEST_FILE = Path("data/splits/test_data.pt")
TOKENIZER_FILE = Path("tokenizer/lawlm.model")

BASELINE_CHECKPOINT = Path("checkpoints/best.pt")
EXPERIMENT_A_CHECKPOINT = Path("experiments/experiment_10_epoch_best.pt")
EXPERIMENT_B_CHECKPOINT = Path("experiments/experiment_larger_best.pt")

BATCH_SIZE = 8
MAX_NEW_TOKENS = 60
TEMPERATURE = 0.8
TOP_K = 40
DEVICE = torch.device("cpu")

PROMPTS = [
    "The court may",
    "Under the applicable law,",
    "A person who commits an offence",
]


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
            _, loss = model(inputs.to(DEVICE), targets.to(DEVICE))
            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def load_checkpoint(path, model_kwargs):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model = LawLM(**model_kwargs).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def generate(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt, out_type=int)

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

    return tokenizer.decode(output_ids[0].tolist())


def main():
    for path in [
        TEST_FILE,
        TOKENIZER_FILE,
        BASELINE_CHECKPOINT,
        EXPERIMENT_A_CHECKPOINT,
        EXPERIMENT_B_CHECKPOINT,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    test_dataset, test_data = load_test_data()

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_FILE)
    )

    models = [
        (
            "Baseline 5 Epochs",
            BASELINE_CHECKPOINT,
            {
                "vocab_size": test_data["vocab_size"],
                "block_size": test_data["block_size"],
            },
        ),
        (
            "Experiment A - 10 Epochs",
            EXPERIMENT_A_CHECKPOINT,
            {
                "vocab_size": test_data["vocab_size"],
                "block_size": test_data["block_size"],
            },
        ),
        (
            "Experiment B - Larger Model",
            EXPERIMENT_B_CHECKPOINT,
            {
                "vocab_size": test_data["vocab_size"],
                "block_size": test_data["block_size"],
                "embed_dim": 320,
                "num_heads": 5,
                "num_layers": 6,
                "ff_hidden_dim": 1280,
                "dropout": 0.1,
            },
        ),
    ]

    print("=" * 60)
    print("LawLM - Experiment C: Model Comparison")
    print("=" * 60)
    print(f"Test sequences: {len(test_dataset):,}")
    print(f"Temperature:    {TEMPERATURE}")
    print(f"Top-k:          {TOP_K}")
    print(f"New tokens:     {MAX_NEW_TOKENS}")
    print()

    results = []

    for name, checkpoint_path, model_kwargs in models:
        model, checkpoint = load_checkpoint(
            checkpoint_path,
            model_kwargs,
        )

        test_loss = evaluate(model, loader)
        perplexity = math.exp(test_loss)

        result = {
            "model": name,
            "epoch": checkpoint["epoch"],
            "train_loss": checkpoint["train_loss"],
            "validation_loss": checkpoint["val_loss"],
            "test_loss": test_loss,
            "perplexity": perplexity,
            "samples": {},
        }

        print("-" * 60)
        print(name)
        print(f"Epoch:             {checkpoint['epoch']}")
        print(f"Train loss:        {checkpoint['train_loss']:.4f}")
        print(f"Validation loss:   {checkpoint['val_loss']:.4f}")
        print(f"Test loss:         {test_loss:.4f}")
        print(f"Test perplexity:   {perplexity:.2f}")
        print()

        for prompt in PROMPTS:
            sample = generate(model, tokenizer, prompt)
            result["samples"][prompt] = sample

            print(f"Prompt: {prompt}")
            print(f"Output: {sample}")
            print()

        results.append(result)

    output_file = Path("experiments/experiment_c_comparison.json")

    with output_file.open("w", encoding="utf-8") as file:
        import json
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Experiment C completed!")
    print(f"Saved: {output_file}")
    print("Disclaimer: Generated text is for research and")
    print("educational purposes only and is NOT legal advice.")
    print("=" * 60)


if __name__ == "__main__":
    main()
