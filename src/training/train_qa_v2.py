"""QA V2 training with automatic resume and a rolling set of five recovery checkpoints.

The best-validation checkpoint is stored separately from the five rolling
recovery checkpoints. Recovery checkpoints include optimizer/RNG state and
the exact epoch order, allowing training to continue from the next batch.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from src.model.lawlm_model import LawLM

DATA = Path("data/qa_v2")
BASE = Path("checkpoints/lawsuit_llm_expanded_best.pt")
BEST = Path("checkpoints/lawsuit_llm_qa_v2_best.pt")
RESUME_DIR = Path("checkpoints/qa_v2_resume")
HISTORY = Path("logs/qa_v2_training_history.jsonl")
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 1
LR = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42
BLOCK = 256
SAVE_EVERY_UPDATES = 50
KEEP_RESUME_CHECKPOINTS = 5


def seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class QADataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "input_ids": torch.tensor(r["input_ids"], dtype=torch.long),
            "labels": torch.tensor(r["labels"], dtype=torch.long),
        }


def collate(batch):
    n = max(x["input_ids"].numel() for x in batch)
    ids = torch.zeros((len(batch), n), dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        length = x["input_ids"].numel()
        ids[i, :length] = x["input_ids"]
        labels[i, :length] = x["labels"]
    return ids, labels


def ppl(x):
    return math.exp(min(x, 20))


def model_from_checkpoint(device):
    ck = torch.load(BASE, map_location=device, weights_only=False)
    model = LawLM(
        vocab_size=10000,
        block_size=BLOCK,
        embed_dim=384,
        num_heads=6,
        num_layers=4,
        ff_hidden_dim=1536,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    return model


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for ids, labels in loader:
            _, loss = model(ids.to(device), targets=labels.to(device))
            total += float(loss)
            batches += 1
    return total / max(1, batches)


def restore_rng(checkpoint):
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])


def checkpoint_payload(model, optimizer, *, epoch, next_batch_in_epoch,
                       epoch_order, global_step, running_loss, batches_seen,
                       best_validation_loss, device):
    return {
        "checkpoint_version": 1,
        "epoch": epoch,
        "next_batch_in_epoch": next_batch_in_epoch,
        "epoch_order": epoch_order,
        "global_step": global_step,
        "running_loss": running_loss,
        "batches_seen": batches_seen,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "vocab_size": 10000,
            "block_size": BLOCK,
            "embed_dim": 384,
            "num_layers": 4,
            "num_heads": 6,
            "ff_hidden_dim": 1536,
            "dropout": 0.1,
        },
        "training": {
            "task": "conservative legal QA adaptation",
            "external_pretrained_llm": False,
            "base_checkpoint": str(BASE),
            "learning_rate": LR,
            "gradient_accumulation": GRAD_ACCUM,
        },
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_resume_checkpoint(payload, global_step):
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    path = RESUME_DIR / f"lawsuit_llm_qa_v2_resume_step_{global_step:08d}.pt"
    torch.save(payload, path)

    checkpoints = sorted(
        RESUME_DIR.glob("lawsuit_llm_qa_v2_resume_step_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_path in checkpoints[KEEP_RESUME_CHECKPOINTS:]:
        old_path.unlink()

    print(f"Recovery checkpoint saved: {path} (keeping newest {KEEP_RESUME_CHECKPOINTS})")


def latest_resume_checkpoint():
    if not RESUME_DIR.exists():
        return None
    checkpoints = list(RESUME_DIR.glob("lawsuit_llm_qa_v2_resume_step_*.pt"))
    return max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description="Train LawLM QA V2 with rolling recovery checkpoints.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore recovery checkpoints and start from the Phase-5 base checkpoint.",
    )
    args = parser.parse_args()

    seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = QADataset(DATA / "train.jsonl")
    val = QADataset(DATA / "validation.jsonl")
    val_loader = DataLoader(
        val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0
    )

    model = model_from_checkpoint(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    resume_path = None if args.fresh else latest_resume_checkpoint()
    start_epoch = 1
    start_batch = 0
    resume_order = None
    global_step = 0
    running_loss = 0.0
    batches_seen = 0
    best = float("inf")
    resume_checkpoint = None

    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"])
        start_batch = int(resume_checkpoint["next_batch_in_epoch"])
        resume_order = resume_checkpoint.get("epoch_order")
        global_step = int(resume_checkpoint.get("global_step", 0))
        running_loss = float(resume_checkpoint.get("running_loss", 0.0))
        batches_seen = int(resume_checkpoint.get("batches_seen", 0))
        best = float(resume_checkpoint.get("best_validation_loss", float("inf")))
        print(f"Resuming from: {resume_path}")
        print(f"Resume position: epoch {start_epoch}, next batch {start_batch + 1:,}, optimizer step {global_step:,}")
    else:
        print("Starting from Phase-5 base checkpoint:", BASE)
        if args.fresh:
            print("--fresh supplied; recovery checkpoints were ignored.")

    print("=" * 72)
    print("LawSuit LLM — Phase 7.5F QA V2")
    print("Device:", device)
    print("Train:", len(train), "| Validation:", len(val))
    print("LR:", LR, "| Grad accumulation:", GRAD_ACCUM, "| Epochs:", EPOCHS)
    print("Rolling recovery checkpoints:", RESUME_DIR, "| Retained:", KEEP_RESUME_CHECKPOINTS)
    print("Best-validation checkpoint:", BEST)
    print("=" * 72)

    if start_epoch > EPOCHS:
        print(f"Configured epochs ({EPOCHS}) are already complete. Nothing to train.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch == start_epoch and resume_order is not None:
            order = resume_order
            batch_offset = start_batch
            epoch_running_loss = running_loss
            epoch_batches_seen = batches_seen
        else:
            order = torch.randperm(len(train)).tolist()
            batch_offset = 0
            epoch_running_loss = 0.0
            epoch_batches_seen = 0

        train_loader = DataLoader(
            train, batch_size=BATCH_SIZE, sampler=order,
            collate_fn=collate, num_workers=0,
        )
        iterator = iter(train_loader)
        for _ in range(batch_offset):
            next(iterator)

        # Iterator creation can consume a torch RNG value; restore the saved
        # state after iterator setup so dropout resumes consistently.
        if resume_checkpoint is not None and epoch == start_epoch:
            restore_rng(resume_checkpoint)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        for bi, (ids, labels) in enumerate(iterator, start=batch_offset + 1):
            _, loss = model(ids.to(device), targets=labels.to(device))
            (loss / GRAD_ACCUM).backward()
            epoch_running_loss += float(loss)
            epoch_batches_seen += 1

            if bi % GRAD_ACCUM == 0 or bi == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % SAVE_EVERY_UPDATES == 0 or bi == len(train):
                    payload = checkpoint_payload(
                        model, optimizer, epoch=epoch, next_batch_in_epoch=bi,
                        epoch_order=order, global_step=global_step,
                        running_loss=epoch_running_loss, batches_seen=epoch_batches_seen,
                        best_validation_loss=best, device=device,
                    )
                    save_resume_checkpoint(payload, global_step)

            if bi % 2000 == 0 or bi == len(train):
                print(f"Epoch {epoch} Batch {bi:,}/{len(train):,} Loss {loss.item():.4f}")

        train_loss = epoch_running_loss / max(1, epoch_batches_seen)
        val_loss = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_perplexity": ppl(train_loss),
            "validation_perplexity": ppl(val_loss),
            "learning_rate": LR,
            "gradient_accumulation": GRAD_ACCUM,
            "base_checkpoint": str(BASE),
            "pretrained_llm": False,
            "global_step": global_step,
        }
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Epoch {epoch}: train {train_loss:.4f} | "
            f"val {val_loss:.4f} | PPL {ppl(val_loss):.2f}"
        )

        if val_loss < best:
            best = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": val_loss,
                "model_config": {
                    "vocab_size": 10000, "block_size": BLOCK, "embed_dim": 384,
                    "num_layers": 4, "num_heads": 6, "ff_hidden_dim": 1536, "dropout": 0.1,
                },
                "training": {
                    "task": "conservative legal QA adaptation",
                    "external_pretrained_llm": False,
                    "base_checkpoint": str(BASE),
                    "learning_rate": LR,
                },
            }, BEST)
            print("Best checkpoint saved:", BEST)

        # Mark this epoch complete, so a later run starts at the next epoch.
        completed_payload = checkpoint_payload(
            model, optimizer, epoch=epoch + 1, next_batch_in_epoch=0,
            epoch_order=None, global_step=global_step, running_loss=0.0,
            batches_seen=0, best_validation_loss=best, device=device,
        )
        save_resume_checkpoint(completed_payload, global_step)

    print("=" * 72)
    print("QA V2 training finished. Best:", BEST, "| Validation loss:", best)


if __name__ == "__main__":
    main()
