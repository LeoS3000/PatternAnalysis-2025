unet.py
# --------------------------------------------------------------
# src/models/unet.py
# --------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


# ----------------------------------------------------------------------
# DoubleConv – unchanged
# ----------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(conv → norm → ReLU) * 2"""
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 mid_ch: Optional[int] = None,
                 norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d,
                 dropout: float = 0.0):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch

        layers = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(mid_ch))
        layers.append(nn.ReLU(inplace=True))

        layers += [
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


# ----------------------------------------------------------------------
# Down – unchanged
# ----------------------------------------------------------------------
class Down(nn.Module):
    """max‑pool then DoubleConv"""
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d,
                 dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch,
                       norm_layer=norm_layer,
                       dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# ----------------------------------------------------------------------
# Up – now supports two execution paths:
#   * with a skip connection (standard UNet)
#   * without a skip connection (decoder‑only mode)
# ----------------------------------------------------------------------
class Up(nn.Module):
    """
    Upscaling then DoubleConv.

    *bilinear=True*  → nn.Upsample + 1×1 conv to shrink channels.
    *bilinear=False* → ConvTranspose2d already shrinks channels.

    When ``skip`` is provided we concatenate and use ``self.conv_concat``
    (expects 2 × out_ch channels).  When ``skip`` is ``None`` we run
    ``self.conv_noskip`` which expects only ``out_ch`` channels.
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 bilinear: bool = True,
                 norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d,
                 dropout: float = 0.0):
        super().__init__()

        if bilinear:
            # 1️⃣ spatial up‑sampling (no learnable params)
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            # 2️⃣ reduce channel dimension from in_ch → out_ch
            self.reduce = nn.Conv2d(in_ch, out_ch,
                                    kernel_size=1, bias=False)
        else:
            # ConvTranspose2d already halves the channel count
            self.up = nn.ConvTranspose2d(in_ch, out_ch,
                                         kernel_size=2,
                                         stride=2)

        # ------------------------------------------------------------------
        # Two different DoubleConv heads
        # ------------------------------------------------------------------
        # 1️⃣ when we have a skip connection → concat → 2*out_ch channels
        self.conv_concat = DoubleConv(out_ch * 2,
                                      out_ch,
                                      norm_layer=norm_layer,
                                      dropout=dropout)

        # 2️⃣ when there is **no** skip → we only have out_ch channels
        self.conv_noskip = DoubleConv(out_ch,
                                      out_ch,
                                      norm_layer=norm_layer,
                                      dropout=dropout)

    def forward(self, x1: torch.Tensor, skip: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x1   : tensor from the previous decoder stage
        skip : encoder feature map (or ``None`` for decoder‑only mode)
        """
        # 1️⃣ up‑sample
        x1 = self.up(x1)

        # 2️⃣ if we used bilinear up‑sampling we still have `in_ch` channels,
        #    so we need the 1×1 conv to bring it down to `out_ch`.
        if hasattr(self, "reduce"):
            x1 = self.reduce(x1)

        # ------------------------------------------------------------------
        # Decoder‑only mode (no skip)
        # ------------------------------------------------------------------
        if skip is None:
            # just run the DoubleConv that expects `out_ch` channels
            return self.conv_noskip(x1)

        # ------------------------------------------------------------------
        # Normal UNet mode (skip connection present)
        # ------------------------------------------------------------------
        # Pad if needed (odd sized inputs)
        diffY = skip.size(2) - x1.size(2)
        diffX = skip.size(3) - x1.size(3)
        x1 = F.pad(x1,
                   [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([skip, x1], dim=1)   # (B, out_ch*2, H, W)
        return self.conv_concat(x)


# ----------------------------------------------------------------------
# OutConv – unchanged
# ----------------------------------------------------------------------
class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.Tanh()  # <-- ADD THIS ACTIVATION LAYER
        )

    def forward(self, x):
        return self.conv(x)


# ----------------------------------------------------------------------
# UNet – now supports a *decoder‑only* mode
# ----------------------------------------------------------------------
class UNet(nn.Module):
    """
    Classic UNet with optional bilinear up‑sampling,
    configurable normalisation layer and dropout.

    Parameters
    ----------
    decoder_only : bool, default=False
        If True the forward pass **skips the down‑sampling blocks** and
        only runs the up‑sampling path.  This makes the class usable as a
        pure decoder (exactly what the VAE‑GAN needs) while still being a
        UNet implementation for the assignment.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 64,
                 depth: int = 4,
                 bilinear: bool = True,
                 norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d,
                 dropout: float = 0.0,
                 decoder_only: bool = False):
        super().__init__()
        self.depth = depth
        self.decoder_only = decoder_only

        # ------------------- Encoder (only used when decoder_only=False) -------------------
        self.inc = DoubleConv(in_channels,
                              base_channels,
                              norm_layer=norm_layer,
                              dropout=dropout)

        self.down_blocks = nn.ModuleList()
        chs = base_channels
        for _ in range(depth):
            self.down_blocks.append(
                Down(chs, chs * 2,
                     norm_layer=norm_layer,
                     dropout=dropout)
            )
            chs *= 2

        # ------------------- Decoder (always built) -------------------
        self.up_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up_blocks.append(
                Up(chs, chs // 2,
                   bilinear=bilinear,
                   norm_layer=norm_layer,
                   dropout=dropout)
            )
            chs //= 2

        # ------------------- Output head -------------------
        self.outc = OutConv(base_channels, out_channels)

    # ------------------------------------------------------------------
    # Forward – two behaviours
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        If ``decoder_only`` is False → full UNet (encoder + decoder).
        If ``decoder_only`` is True  → only the decoder part; the input
        must already have the channel dimension that the deepest encoder
        would have produced (i.e. base_channels * 2**depth).
        """
        if not self.decoder_only:
            # ---------- full encoder ----------
            x = self.inc(x)
            skips = []
            for down in self.down_blocks:
                skips.append(x)          # store BEFORE down‑sampling
                x = down(x)              # go deeper

            # ---------- decoder ----------
            for up in self.up_blocks:
                skip = skips.pop()
                x = up(x, skip)

            # ---------- final 1×1 conv ----------
            return self.outc(x)

        # --------------------------------------------------------------
        # decoder‑only branch
        # --------------------------------------------------------------
        # ``x`` is expected to be (B, C_deep, H, W) where
        # C_deep = base_channels * 2**depth.
        # No skip connections are available, so we simply call each Up
        # block with ``skip=None``.
        for up in self.up_blocks:
            x = up(x, None)   # <-- decoder‑only call (no skip)

        return self.outc(x)