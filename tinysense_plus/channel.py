"""Synthetic index-corruption channel."""

import torch


def substitute_indices(
    indices: torch.Tensor, error_rate: float, codebook_size: int
) -> torch.Tensor:
    """Replace each VQ index, independently with probability ``error_rate``, by a
    uniformly random codebook index.

    This corrupts the decoded index map. It does not act on the arithmetic-coded
    bitstream and does not model packetisation or loss of entropy-coded data.
    """
    keep_rate = 1 - error_rate
    keep = torch.bernoulli(keep_rate * torch.ones(indices.shape, device=indices.device))
    keep = keep.round().to(dtype=torch.int64)
    random_indices = torch.randint_like(indices, codebook_size)
    return keep * indices + (1 - keep) * random_indices
