from models.vae_gan import VAEGAN
from models.unet import UNet
if __name__ == "__main__":
    import torch
    u = UNet(in_channels=3, out_channels=1, base_channels=64, depth=4, bilinear=True)
    x = torch.randn(2, 3, 256, 256)          # (B, C, H, W)

    # forward pass – should run without error
    out = u(x)
    print(out.shape)                         # → torch.Size([2, 1, 256, 256])