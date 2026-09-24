"""Phase 4 dataset utilities for LawSuit LLM.

Converts tokenized legal provisions into fixed-length causal language-model
samples without loading the complete corpus into memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset


class CausalLMDataset(Dataset):
    """Map-style dataset over fixed-length token blocks.

    Each split is streamed once during indexing preparation. Only token-block
    offsets are retained in memory; the full corpus is never loaded at once.
    """

    def __init__(
        self,
        input_file: str | Path,
        block_size: int = 256,
        eos_id: int = 3,
        drop_last: bool = True,
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")

        self.input_file = Path(input_file)
        self.block_size = block_size
        self.eos_id = eos_id
        self.drop_last = drop_last

        if not self.input_file.exists():
            raise FileNotFoundError(f"Tokenized split not found: {self.input_file}")

        self._blocks: list[list[int]] = []
        self._build_blocks()

    def _token_stream(self) -> Iterator[int]:
        with self.input_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                tokens = row.get("tokens", [])
                if not isinstance(tokens, list):
                    continue
                for token in tokens:
                    yield int(token)
                yield self.eos_id

    def _build_blocks(self) -> None:
        buffer: list[int] = []

        for token in self._token_stream():
            buffer.append(token)
            while len(buffer) >= self.block_size + 1:
                self._blocks.append(buffer[: self.block_size + 1])
                del buffer[: self.block_size]

        if not self.drop_last and len(buffer) >= 2:
            self._blocks.append(buffer)

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self._blocks[index]
        if len(block) < 2:
            raise RuntimeError("A causal LM block must contain at least 2 tokens")

        input_ids = torch.tensor(block[:-1], dtype=torch.long)
        targets = torch.tensor(block[1:], dtype=torch.long)
        return input_ids, targets


def build_split_dataset(
    split: str,
    block_size: int = 256,
    tokenized_dir: str | Path = "data/tokenized",
) -> CausalLMDataset:
    """Build a causal LM dataset for train/validation/test."""
    path = Path(tokenized_dir) / f"{split}.jsonl"
    return CausalLMDataset(path, block_size=block_size)
