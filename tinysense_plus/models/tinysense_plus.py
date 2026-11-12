"""TinySense+ model: VQ compression of CSI with an entropy-coded index stream."""

import math
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

from ..channel import substitute_indices
from .entropy_model import EntropyModel
from .networks import Decoder, Encoder, PoseRegressor

try:
    import torchac
except Exception:  # compiled lazily; needs a C++ toolchain
    torchac = None

EMBEDDING_DIM = 256
TOKEN_DIM = 56
LATENT_HEIGHT = 7
NUM_TOKENS = EMBEDDING_DIM * LATENT_HEIGHT // TOKEN_DIM  # 32 VQ indices per CSI frame


class ForwardOutput(NamedTuple):
    rate_loss: torch.Tensor
    entropy_loss: torch.Tensor
    vq_loss: torch.Tensor
    latent: torch.Tensor
    recon: torch.Tensor
    pose: torch.Tensor
    coded_bytes: Optional[float] = None  # arithmetic-coded bytes / sample (test only)
    fixed_bytes: Optional[float] = None  # fixed-length-coded bytes / sample (test only)


class TinySensePlus(nn.Module):
    """Edge encoder + VQ + entropy model, and server decoder + pose head.

    Args:
        codebook_size: number of VQ codewords K.
        num_joints: number of 2-D keypoints predicted by the pose head.
        train_index_error_rate: probability with which VQ indices are replaced by
            random ones during training (0 disables the corruption).
        commitment_weight: VQ commitment-loss weight.
    """

    def __init__(
        self,
        codebook_size: int,
        num_joints: int = 17,
        train_index_error_rate: float = 0.1,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_joints = num_joints
        self.train_index_error_rate = train_index_error_rate

        self.encoder = Encoder()
        self.pre_vq_conv = nn.Sequential(
            nn.Conv2d(32, EMBEDDING_DIM, kernel_size=1, stride=1),
            nn.BatchNorm2d(EMBEDDING_DIM),
            nn.ReLU(),
        )
        self.latent_pool = nn.AdaptiveAvgPool2d((LATENT_HEIGHT, 1))
        self.pose_head = PoseRegressor(
            input_dim=EMBEDDING_DIM * LATENT_HEIGHT,
            output_dim=num_joints * 2,
            hidden_dim=32,
        )
        self.decoder = Decoder(in_channel=EMBEDDING_DIM)
        self.vq = VectorQuantize(
            dim=TOKEN_DIM,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            decay=0.8,
            kmeans_init=True,
            threshold_ema_dead_code=2,
        )
        self.entropy_model = EntropyModel(codebook_size)

    # ------------------------------------------------------------------ coding
    def fixed_length_bytes(self, indices: torch.Tensor) -> int:
        """Bytes needed to send ``indices`` with ceil(log2 K) bits per index."""
        bits_per_index = max(1, math.ceil(math.log2(self.codebook_size)))
        total_bits = indices.size(0) * indices.size(1) * bits_per_index
        return (total_bits + 7) // 8

    @staticmethod
    def _cdf(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        zeros = torch.zeros_like(probs[..., :1])
        cdf = torch.cumsum(torch.cat([zeros, probs], dim=-1), dim=-1)
        return torch.clamp(cdf, 0.0, 1.0).cpu().contiguous()

    @torch.no_grad()
    def encode(self, indices: torch.Tensor):
        """Arithmetic-code every sample with ``entropy_model``.

        Returns the list of byte streams and their total length in bytes. Only the
        stream payload is counted: no header, framing or synchronisation data.
        """
        if torchac is None:
            raise ImportError(
                "torchac is required for arithmetic coding (pip install torchac; "
                "it compiles a small C++ extension on first import)."
            )
        streams: List[bytes] = []
        total_bytes = 0
        for b in range(indices.size(0)):
            sample = indices[b : b + 1]
            cdf = self._cdf(self.entropy_model(sample))
            stream = torchac.encode_float_cdf(cdf, sample.to(torch.int16).cpu())
            streams.append(stream)
            total_bytes += len(stream)
        return streams, total_bytes

    @torch.no_grad()
    def decode_with_known_context(
        self, streams: List[bytes], context_indices: torch.Tensor
    ) -> torch.Tensor:
        """Decode ``streams`` using CDFs computed from ``context_indices``.

        The probability model for position t is evaluated from the *true* indices
        (teacher forcing) in a single pass, so this verifies that a stream
        round-trips, but it is not a sequential receiver that only has access to
        already-decoded symbols.
        """
        decoded = []
        for b, stream in enumerate(streams):
            sample = context_indices[b : b + 1]
            cdf = self._cdf(self.entropy_model(sample))
            decoded.append(torchac.decode_float_cdf(cdf, stream).long().to(sample.device))
        return torch.cat(decoded, dim=0)

    # ----------------------------------------------------------------- forward
    def _quantise(self, x: torch.Tensor):
        batch = x.shape[0]
        z = self.latent_pool(self.pre_vq_conv(self.encoder(x)))
        z = z.view(batch, -1, TOKEN_DIM)
        z_quantised, indices, vq_loss = self.vq(z)
        return z, z_quantised, indices, vq_loss

    def forward(
        self,
        x: torch.Tensor,
        is_test: bool = False,
        error_rate: float = 0.0,
        prefix_ratio: float = 1.0,
        tau: float = 0.1,
        detach_rate_target: bool = False,
    ) -> ForwardOutput:
        """Run the model.

        Training (``is_test=False``): differentiable rate loss (soft assignment of
        latents to codewords, cross-entropy under ``entropy_model``) and the
        entropy-model training loss are returned.

        Testing (``is_test=True``): the first ``prefix_ratio`` of the indices is
        coded, optionally corrupted by index substitution with probability
        ``error_rate``, and the missing suffix is generated by the entropy model.
        """
        batch = x.shape[0]
        z, z_quantised, indices, vq_loss = self._quantise(x)

        rate_loss = torch.tensor(0.0, device=x.device)
        entropy_loss = torch.tensor(0.0, device=x.device)
        coded_bytes = fixed_bytes = None

        if not is_test:
            rate_target = z.detach() if detach_rate_target else z
            distance = torch.cdist(rate_target, self.vq.codebook.detach())
            p_soft = F.softmax(-distance / tau, dim=-1)
            logits = self.entropy_model(indices)
            log_q = F.log_softmax(logits, dim=-1)
            rate_loss = -torch.sum(p_soft * log_q, dim=-1).mean()
            entropy_loss = F.cross_entropy(
                logits.reshape(-1, self.codebook_size), indices.reshape(-1)
            )

            if self.train_index_error_rate > 0:
                received = substitute_indices(
                    indices.cpu(), self.train_index_error_rate, self.codebook_size
                ).to(indices.device)
            else:
                received = indices
        else:
            seq_len = indices.size(1)
            prefix_len = max(1, int(seq_len * prefix_ratio))
            prefix = indices[:, :prefix_len]

            fixed_bytes = self.fixed_length_bytes(prefix) / max(1, batch)
            streams, total_bytes = self.encode(prefix)
            coded_bytes = float(total_bytes) / max(1, batch)
            decoded = self.decode_with_known_context(streams, prefix)

            if error_rate > 0.0:
                decoded = substitute_indices(
                    decoded.cpu(), error_rate, self.codebook_size
                ).to(indices.device)
            received = self.entropy_model.complete_prefix(decoded, total_len=seq_len)

        if not is_test and self.train_index_error_rate == 0:
            z_hat = z_quantised
        else:
            z_hat = self.vq.get_codes_from_indices(received)
            if not is_test:
                z_hat = z + (z_hat - z).detach()  # straight-through estimator

        z_hat = z_hat.view(batch, EMBEDDING_DIM, LATENT_HEIGHT, 1)
        pose = self.pose_head(z_hat).reshape(batch, -1, 2)
        recon = self.decoder(z_hat)
        return ForwardOutput(
            rate_loss, entropy_loss, vq_loss, z, recon, pose, coded_bytes, fixed_bytes
        )
