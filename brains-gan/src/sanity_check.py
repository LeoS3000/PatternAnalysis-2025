import torch
from models.vae_gan import VAEGAN
from torchvision.utils import save_image
torch.autograd.set_detect_anomaly(True)
device = torch.device('cpu')
model = VAEGAN(latent_dim=128).to(device)

# Dummy batch of random images (batch=2, 3×256×256)
x = torch.randn(2, 3, 256, 256, device=device)

recon, mu, logvar, _ = model(x)
print('recon shape:', recon.shape)          # (2,3,256,256)
print('mu shape:', mu.shape)                # (2,128)
print('logvar shape:', logvar.shape)        # (2,128)

grid = torch.cat([x[:1], recon[:1]], dim=0)   # real vs. fake
save_image(grid, 'sanity.png')
print('Saved sanity.png – open it to see the reconstruction.')