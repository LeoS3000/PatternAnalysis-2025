# train_vae.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T

from dataset import PNGDataset          
from model_vae import VAE              


print("=== VAE training start ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyper‑parameters
latent_dim   = 128
batch_size   = 16
epochs       = 30         
lr           = 1e-3
img_dir      = '../data/keras_png_slices_train'
out_dir      = 'out/vae'
os.makedirs(out_dir, exist_ok=True)

# Dataset & DataLoader
dataset = PNGDataset(img_dir)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Model, optimiser, loss
model = VAE(latent_dim).to(device)          # <-- VAEGAN wrapper works as VAE
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse_loss = torch.nn.MSELoss()

def vae_loss(recon, x, mu, logvar):
    """Standard VAE loss = reconstruction (MSE) + KL divergence."""
    recon_loss = mse_loss(recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ------------------------------------------------------------------
# 7️⃣  Training loop
# ------------------------------------------------------------------
for epoch in range(epochs):
    model.train()                     # make sure we are in training mode
    for i, x in enumerate(loader):
        x = x.to(device)              # (B, C, H, W) – already normalised

        optimizer.zero_grad()
        recon, mu, logvar = model(x)  # forward
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        # --------------------------------------------------------------
        # Logging & checkpointing
        # --------------------------------------------------------------
        if i % 100 == 0:
            print(f"[Epoch {epoch:02d} | Step {i:04d}] loss = {loss.item():.4f}")

            # Save a *detached* grid of reconstructions (no grads attached)
            model.eval()
            with torch.no_grad():
                save_image(
                    recon[:8].cpu().detach(),
                    os.path.join(out_dir, f"recon_e{epoch}_s{i}.png")
                )
            model.train()

    ckpt_path = os.path.join(out_dir, f"ckpt_epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)

print("Training finished")