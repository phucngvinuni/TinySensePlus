"""Selective-kernel (SK) convolution blocks used by the encoder and discriminator."""

import torch
from torch import nn

NUM_BRANCHES = 4
ATTENTION_REDUCTION = 4


class SKConv(nn.Module):
    """Selective-kernel convolution with attention pooled over the frequency axis.

    Four 3x3 branches with dilation 1..4 are summed, pooled over the last
    (time) axis, and a per-branch softmax attention over channels is applied.
    """

    def __init__(self, input_dim: int, output_dim: int, freq_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.output_dim = output_dim
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_dim,
                        output_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1 + i,
                        dilation=1 + i,
                        groups=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True),
                )
                for i in range(NUM_BRANCHES)
            ]
        )
        hidden = int(output_dim / ATTENTION_REDUCTION)
        self.fc = nn.Sequential(
            nn.Conv1d(output_dim, hidden, kernel_size=1, stride=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList(
            [
                nn.Conv1d(hidden, output_dim, kernel_size=1, stride=1)
                for _ in range(NUM_BRANCHES)
            ]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        feats = torch.cat([conv(x) for conv in self.convs], dim=1)
        feats = feats.view(
            batch, NUM_BRANCHES, feats.shape[2], self.output_dim, feats.shape[3]
        )
        fused = torch.sum(feats, dim=1)

        pooled = torch.mean(fused, dim=3)
        pooled = pooled.view(batch, pooled.shape[2], pooled.shape[1])
        squeezed = self.fc(pooled)
        attention = torch.cat([fc(squeezed) for fc in self.fcs], dim=1)
        attention = attention.view(
            batch, NUM_BRANCHES, self.output_dim, self.freq_dim, 1
        )
        attention = self.softmax(attention)
        attention = attention.view(
            batch, NUM_BRANCHES, self.freq_dim, self.output_dim, 1
        )

        out = torch.sum(feats * attention, dim=1)
        return torch.transpose(out, 1, 2)


class SKUnit(nn.Module):
    """Bottleneck unit: 1x1 conv, SK convolution, and a residual shortcut."""

    def __init__(
        self, in_features: int, mid_features: int, out_features: int, freq_dim: int
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
        )
        self.conv2_sk = nn.Sequential(
            SKConv(mid_features, out_features, freq_dim),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_features),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2_sk(self.conv1(x))
        return self.relu(out + self.shortcut(x))
