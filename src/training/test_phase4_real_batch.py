"""Run one real legal-data batch through the LawSuit LLM.

This is a bridge test between the Phase 3 tokenized corpus and the Phase 4
Transformer. It performs one optimizer update but does not run full training.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.model.lawlm_model import LawSuitLLM
from src.training.causal_lm_dataset import build_split_dataset


def main() -> None:
    torch.manual_seed(42)

    dataset = build_split_dataset("train", block_size=256)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    input_ids, targets = next(iter(loader))

    model = LawSuitLLM(
        vocab_size=10_000,
        block_size=256,
        embed_dim=384,
        num_heads=6,
        num_layers=4,
        ff_hidden_dim=1536,
        dropout=0.1,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    logits, loss = model(input_ids, targets)

    if loss is None:
        raise RuntimeError("Expected a training loss from real legal data.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print("=" * 60)
    print("LawSuit LLM - Phase 4 Real Legal Batch Test")
    print("=" * 60)
    print(f"Dataset samples: {len(dataset):,}")
    print(f"Batch input shape: {tuple(input_ids.shape)}")
    print(f"Batch target shape: {tuple(targets.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss before update: {loss.item():.4f}")
    print(f"Gradient norm: {float(gradient_norm):.4f}")
    print("Backward pass: OK")
    print("AdamW optimizer step: OK")
    print("Real legal data → model → loss → update: OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
