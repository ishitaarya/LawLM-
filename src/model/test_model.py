import torch

from lawlm_model import LawLM


def main():
    torch.manual_seed(42)

    model = LawLM()

    input_ids = torch.randint(
        low=0,
        high=model.vocab_size,
        size=(2, model.block_size),
    )

    targets = torch.randint(
        low=0,
        high=model.vocab_size,
        size=(2, model.block_size),
    )

    logits, loss = model(input_ids, targets)

    expected_shape = (
        2,
        model.block_size,
        model.vocab_size,
    )

    assert logits.shape == expected_shape
    assert loss is not None
    assert torch.isfinite(loss)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("=" * 60)
    print("LawLM - Transformer Model Test")
    print("=" * 60)
    print(f"Parameters: {parameter_count:,}")
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.4f}")
    print()
    print("Transformer forward pass test passed!")


if __name__ == "__main__":
    main()
