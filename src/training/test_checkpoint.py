import torch

from src.model.lawlm_model import LawLM


def main():
    checkpoint_path = "checkpoints/best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = LawLM()
    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 60)
    print("LawLM - Checkpoint Test")
    print("=" * 60)
    print(f"Epoch:       {checkpoint['epoch']}")
    print(f"Train loss:  {checkpoint['train_loss']:.4f}")
    print(f"Val loss:    {checkpoint['val_loss']:.4f}")
    print("Checkpoint loaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
