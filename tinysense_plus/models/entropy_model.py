"""Causal autoregressive entropy model q_psi over VQ indices."""

import torch
import torch.nn.functional as F
from torch import nn

from .gpt import GPT

SOS_TOKEN = 0


class EntropyModel(nn.Module):
    """Predicts p(i_t | i_<t) for every position of a VQ-index sequence.

    Codebook index ``k`` in ``[0, K)`` is mapped to token ``k + 1``; token 0 is a
    start-of-sequence symbol.
    """

    def __init__(
        self,
        codebook_size: int,
        block_size: int = 512,
        n_layer: int = 2,
        n_head: int = 16,
        n_embd: int = 1024,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.transformer = GPT(
            vocab_size=codebook_size + 1,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, N, K) for the distribution of each index.

        Position t only sees the indices before it, so the logits at position 0
        are conditioned on the start symbol alone.
        """
        shifted = indices + 1
        sos = torch.zeros(shifted.shape[0], 1, dtype=torch.long, device=indices.device)
        tokens = torch.cat((sos, shifted), dim=1)
        logits = self.transformer(tokens[:, :-1])
        return logits[:, :, 1:]

    @torch.no_grad()
    def complete_prefix(self, prefix_indices: torch.Tensor, total_len: int) -> torch.Tensor:
        """Greedily (argmax) generate the missing suffix of an index sequence."""
        sequence = prefix_indices.clone()
        for _ in range(prefix_indices.size(1), total_len):
            next_logits = self.forward(sequence)[:, -1, :]
            next_token = torch.argmax(F.softmax(next_logits, dim=-1), dim=-1)
            sequence = torch.cat([sequence, next_token.unsqueeze(1)], dim=1)
        return sequence
