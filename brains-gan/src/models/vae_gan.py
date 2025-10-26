vae_gan.py
# --------------------------------------------------------------
# src/models/vae_gan.py
# --------------------------------------------------------------
import torch
from torch import nn
from typing import Callable, Optional

# ------------------------------------------------------------------
# Local imports – keep the same names you already have
# ------------------------------------------------------------------
from .encoder import Encoder               # your encoder implementation
from .unet import UNet                     # UNet with decoder_only flag
from .discriminator import PatchDiscriminator


class VAEGAN(nn.Module):
    """
    VAE‑GAN that uses:
    * Encoder → (mu, logvar) of size `latent_dim`
    * Decoder → UNet in *decoder‑only* mode
    * Discriminator → PatchGAN (unchanged)

    The only change compared to the previous version is the **latent‑to‑feature
    projection** (`self.latent_proj`).  It expands the 1 × 1 latent map to the
    spatial size that the decoder expects (16 × 16 for depth=4, 2**depth in general).
    """
    def __init__(self,
                 latent_dim: int = 128,
                 img_channels: int = 1,
                 base_channels: int = 64,
                 depth: int = 4,
                 bilinear: bool = True,
                 norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d,
                 dropout: float = 0.0):
        super().__init__()

        # ------------------------------------------------------------------
        # 1️⃣  Store hyper‑parameters that we will need later
        # ------------------------------------------------------------------
        self.latent_dim   = latent_dim
        self.base_channels = base_channels
        self.depth        = depth
        # deepest encoder channel count = base_channels * 2**depth
        self.deep_channels = base_channels * (2 ** depth)   # e.g. 64 * 2⁴ = 1024

        # ------------------------------------------------------------------
        # 2️⃣  Encoder – unchanged
        # ------------------------------------------------------------------
        self.encoder = Encoder(img_channels=img_channels,
                               latent_dim=latent_dim,
                               base_channels=base_channels,
                               depth=depth,
                               bilinear=bilinear,
                               norm_layer=norm_layer,
                               dropout=dropout)

        # ------------------------------------------------------------------
        # 3️⃣  **Projection** from (B, latent_dim) → (B, deep_channels, 16, 16)
        # ------------------------------------------------------------------
        # We first reshape the latent vector to (B, latent_dim, 1, 1) and then
        # apply a ConvTranspose2d that upsamples it by a factor of 2**depth.
        # kernel_size = stride = 2**depth gives exactly the required output size.
        self.latent_proj = nn.ConvTranspose2d(
            in_channels=latent_dim,
            out_channels=self.deep_channels,
            kernel_size=2 ** depth,          # 16 for depth=4
            stride=2 ** depth,               # same as kernel_size → 1×1 → 16×16
            bias=False
        )

        # ------------------------------------------------------------------
        # 4️⃣  Decoder – UNet in *decoder‑only* mode
        # ------------------------------------------------------------------
        self.decoder = UNet(in_channels=self.deep_channels,   # must match proj output
                            out_channels=img_channels,
                            base_channels=base_channels,
                            depth=depth,
                            bilinear=bilinear,
                            norm_layer=norm_layer,
                            dropout=dropout,
                            decoder_only=True)          # <‑‑ key flag

        # ------------------------------------------------------------------
        # 5️⃣  Discriminator – unchanged
        # ------------------------------------------------------------------
        self.discriminator = PatchDiscriminator(in_channels=img_channels,
                                                base_channels=base_channels)

    # ------------------------------------------------------------------
    # Re‑parameterisation (same as before)
    # ------------------------------------------------------------------
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma²) using the re‑parameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Forward (generator = encoder + decoder)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        recon      : reconstructed image (B, C, H, W)
        mu, logvar: latent distribution parameters (B, latent_dim)
        z_spatial : projected latent tensor fed to the decoder
                     shape = (B, deep_channels, 2**depth, 2**depth)
        """
        # 1️⃣ Encode → mu, logvar
        mu, logvar = self.encoder(x)                # (B, latent_dim) each

        # 2️⃣ Sample latent vector
        z = self.reparameterize(mu, logvar)         # (B, latent_dim)

        # 3️⃣ Reshape to (B, latent_dim, 1, 1) and **project** to the
        #    spatial size expected by the decoder (deep_channels, 16, 16)
        z = z.unsqueeze(-1).unsqueeze(-1)           # (B, latent_dim, 1, 1)
        z_spatial = self.latent_proj(z)             # (B, deep_channels, 16, 16)

        # 4️⃣ Decode – UNet runs only the up‑sampling path
        recon = self.decoder(z_spatial)              # (B, img_channels, H, W)

        return recon, mu, logvar, z_spatial