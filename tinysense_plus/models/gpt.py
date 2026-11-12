"""Minimal causal GPT used as the autoregressive entropy model.

Adapted from Andrej Karpathy's minGPT (https://github.com/karpathy/minGPT,
MIT License), as vendored in taming-transformers. See NOTICE.md.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self, n_embd: int, n_head: int, block_size: int, attn_pdrop: float, resid_pdrop: float
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer(
            "mask", mask.view(1, 1, block_size, block_size), persistent=False
        )
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        head_dim = C // self.n_head
        k = self.key(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(
        self, n_embd: int, n_head: int, block_size: int, attn_pdrop: float, resid_pdrop: float
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Decoder-only Transformer returning next-token logits for every position."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 12,
        n_head: int = 8,
        n_embd: int = 256,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        x = self.drop(self.tok_emb(idx) + self.pos_emb[:, :t, :])
        x = self.blocks(x)
        return self.head(self.ln_f(x))
