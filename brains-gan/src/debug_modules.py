# debug_modules.py  (place it in brains-gan/src/)
import torch
from models.unet import UNet, DoubleConv, Down, Up, OutConv
from models.encoder import Encoder          # if you have one
from models.vae_gan import VAEGAN

def show(tensor, name):
    """Utility to print shape with a label."""
    print(f"{name:20s} → {tuple(tensor.shape)}")
    return tensor

def test_unet():
    print("\n=== UNet ===")
    net = UNet(in_channels=3, out_channels=256,
               base_channels=64, depth=4, bilinear=True)
    x = torch.randn(2, 3, 256, 256)

    # encoder (manual)
    x1 = net.inc(x);               show(x1, "inc")
    skips = [x1]
    for i, d in enumerate(net.down_blocks, 1):
        x1 = d(x1);                show(x1, f"down{i}")
        skips.append(x1)

    # **remove the deepest feature from the skip list**
    skips.pop()                     # <-- this line fixes the channel mismatch

    # decoder
    for i, u in enumerate(net.up_blocks, 1):
        skip = skips.pop()
        x1 = u(x1, skip);          show(x1, f"up{i}")

    out = net.outc(x1);            show(out, "outc")
    return out

def test_encoder():
    print("\n=== Encoder (if you have one) ===")
    # replace with your actual encoder class if different
    from models.encoder import Encoder
    enc = Encoder(img_channels=3, latent_dim=128, base_channels=64)
    x = torch.randn(2, 3, 256, 256)
    mu, logvar = enc(x)
    show(mu, "mu"); show(logvar, "logvar")
    return mu, logvar

def test_vae_gan():
    print("\n=== VAEGAN (UNet decoder‑only) ===")
    model = VAEGAN(latent_dim=128,
                   img_channels=3,
                   base_channels=64,
                   depth=4,
                   bilinear=True).to('cpu')
    x = torch.randn(2, 3, 256, 256)   # dummy batch
    recon, mu, logvar, z_spatial = model(x)

    print("input shape      :", x.shape)                # (2,3,256,256)
    print("mu shape         :", mu.shape)               # (2,128)
    print("logvar shape     :", logvar.shape)           # (2,128)
    print("z_spatial shape  :", z_spatial.shape)        # (2,1024,1,1)
    print("recon shape      :", recon.shape)            # (2,3,256,256)
    assert recon.shape == x.shape, "Reconstruction must match input size"
    print("✅ VAEGAN forward pass works!")

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    test_unet()          # sanity‑check the full UNet (encoder+decoder)
    test_vae_gan()       # sanity‑check the VAE‑GAN with decoder‑only UNet