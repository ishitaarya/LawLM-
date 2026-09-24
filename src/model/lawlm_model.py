"""LawSuit LLM — decoder-only Transformer with causal generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class LawSuitLLM(nn.Module):
    def __init__(self, vocab_size: int = 10000, block_size: int = 256, embed_dim: int = 384,
                 num_heads: int = 6, num_layers: int = 4, ff_hidden_dim: int = 1536,
                 dropout: float = 0.1, tie_embeddings: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_hidden_dim = ff_hidden_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.apply(self._init_weights)
        if tie_embeddings:
            self.output_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, seq_len = input_ids.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.block_size}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.output_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int | None = None,
                 repetition_penalty: float = 1.0, eos_token_id: int | None = None) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.block_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature

            if repetition_penalty > 1.0:
                for batch_index in range(input_ids.size(0)):
                    seen = torch.unique(input_ids[batch_index])
                    next_logits[batch_index, seen] /= repetition_penalty

            if top_k is not None:
                k = min(max(1, top_k), self.vocab_size)
                values, _ = torch.topk(next_logits, k)
                threshold = values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return input_ids


LawLM = LawSuitLLM
