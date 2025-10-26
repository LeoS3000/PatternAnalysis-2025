encoder.py
# --------------------------------------------------------------
# src/models/encoder.py
# --------------------------------------------------------------
import torch
import torch.nn as nn
from .unet import UNet
from typing import Callable, Optional


class Encoder(nn.Module):
    """
    Encoder that uses a UNet backbone and returns (mu, logvar).

    The UNet is asked to output ``latent_dim * 2`` channels.
    The first half are interpreted as the mean, the second half as the
    log‑variance. A global average‑pool collapses the spatial dimensions.
    """
    def __init__(self,
                 img_channels: int = 1,
                 latent_dim: int = 128,
                 base_channels: int = 64,
                 depth: int = 4,
                 bilinear: bool = True,
                 norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d,
                 dropout: float = 0.0):
        super().__init__()
        self.unet = UNet(in_channels=img_channels,
                         out_channels=latent_dim * 2,
                         base_channels=base_channels,
                         depth=depth,
                         bilinear=bilinear,
                         norm_layer=norm_layer,
                         dropout=dropout)

        # Collapse H×W → 1×1 so we get a flat vector per image
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # (B, 2*latent_dim, H, W)
        out = self.unet(x)
        # (B, 2*latent_dim, 1, 1) → (B, 2*latent_dim)
        out = self.pool(out).squeeze(-1).squeeze(-1)

        # split into mean and log‑variance
        mu, logvar = torch.chunk(out, 2, dim=1)   # each → (B, latent_dim)
        return mu, logvar