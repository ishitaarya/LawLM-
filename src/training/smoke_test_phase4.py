"""Phase 4 smoke test: random model, forward pass, loss and backward pass."""

import torch

from src.model.lawlm_model import LawSuitLLM


def main() -> None:
    torch.manual_seed(42)

    model = LawSuitLLM(
        vocab_size=10_000,
        block_size=256,
        embed_dim=384,
        num_heads=6,
        num_layers=4,
        ff_hidden_dim=1536,
        dropout=0.1,
    )

    parameter_count = model.parameter_count()
    input_ids = torch.randint(0, 10_000, (2, 256), dtype=torch.long)
    targets = torch.randint(0, 10_000, (2, 256), dtype=torch.long)

    logits, loss = model(input_ids, targets)

    if loss is None:
        raise RuntimeError("Expected a training loss.")
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]

    if not gradients:
        raise RuntimeError("No gradients were produced.")

    print("=" * 60)
    print("LawSuit LLM - Phase 4 Transformer Smoke Test")
    print("=" * 60)
    print(f"Parameters: {parameter_count:,}")
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.4f}")
    print("Backward pass: OK")
    print("Weights: random initialization")
    print("Pretrained LLM: false")
    print("=" * 60)


if __name__ == "__main__":
    main()
