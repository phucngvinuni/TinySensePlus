"""Encoder, decoder, discriminator and pose head for MM-Fi WiFi-CSI amplitude input.

Input CSI tensors have shape (B, 3, 114, 10): 3 antennas, 114 subcarriers,
10 packets.
"""

import torch
from torch import nn

from .sk_blocks import SKUnit


class Encoder(nn.Module):
    """Three SK units with 2x average pooling in between.

    (B, 3, 114, 10) -> (B, 32, 28, 2)
    """

    def __init__(self):
        super().__init__()
        self.SKunit1 = SKUnit(in_features=3, mid_features=16, out_features=32, freq_dim=114)
        self.SKunit2 = SKUnit(in_features=32, mid_features=64, out_features=128, freq_dim=57)
        self.SKunit3 = SKUnit(in_features=128, mid_features=64, out_features=32, freq_dim=28)
        self.pool_1 = nn.AvgPool2d(2)
        self.pool_2 = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool_1(self.SKunit1(x))
        x = self.pool_2(self.SKunit2(x))
        return self.SKunit3(x)


class Discriminator(nn.Module):
    """Real/fake classifier on CSI; returns the probability that the input is real."""

    def __init__(self):
        super().__init__()
        self.features = Encoder()
        self.final_pool = nn.AdaptiveAvgPool2d((7, 1))
        self.fc1 = nn.Linear(32 * 7 * 1, 512)
        self.bn = nn.BatchNorm1d(512, momentum=0.9)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.final_pool(self.features(x)).view(batch, -1)
        x = self.relu(self.bn(self.fc1(x)))
        return self.sigmoid(self.fc2(x))


class PoseRegressor(nn.Module):
    """MLP head mapping the quantised latent to 2-D joint coordinates."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.bn(self.fc2(x))))
        return self.fc3(x)


class Decoder(nn.Module):
    """Transposed-conv decoder reconstructing CSI from the quantised latent.

    (B, in_channel, 7, 1) -> (B, 3, 114, 10)
    """

    def __init__(self, in_channel: int):
        super().__init__()
        self.transposed_conv1 = nn.ConvTranspose2d(
            in_channel, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.transposed_conv2 = nn.ConvTranspose2d(
            128, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.transposed_conv3 = nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(3)
        self.pool = nn.AdaptiveAvgPool2d((114, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.transposed_conv1(x)))
        x = self.relu(self.bn2(self.transposed_conv2(x)))
        x = self.relu(self.bn3(self.transposed_conv3(x)))
        return self.pool(x)
