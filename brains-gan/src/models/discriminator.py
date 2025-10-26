discriminator.py
# In src/models/discriminator.py
import torch
from torch import nn

class PatchDiscriminator(nn.Module):
    """
    A lightweight PatchGAN discriminator.
    It classifies each N×N patch as real/fake and averages the result.
    Works well for 256×256 images and is cheap to train.
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            # Conv => LeakyReLU (no BN on first layer, per PatchGAN paper)
            # Input: (B, 1, 256, 256) -> Output: (B, 64, 128, 128)
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (base) → (base*2)
            # Input: (B, 64, 128, 128) -> Output: (B, 128, 64, 64)
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # This block was correctly commented out to weaken the discriminator.
            # # (base*2) → (base*4)
            # nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(base_channels*4, affine=True),
            # nn.LeakyReLU(0.2, inplace=True),

            # (base*2) → (base*8) -- THIS IS THE CORRECTED LAYER
            # It now correctly takes base_channels*2 (128) as input.
            # Input: (B, 128, 64, 64) -> Output: (B, 512, 63, 63)
            nn.Conv2d(base_channels * 2, base_channels * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Final 1‑channel output (real/fake score per patch)
            # This layer was commented out in your file but is required.
            # Input: (B, 512, 63, 63) -> Output: (B, 1, 62, 62)
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)   # raw logits – we will apply BCEWithLogitsLoss later