# losses.py
import torch
import torch.nn as nn

# ---- VAE losses -------------------------------------------------------
mse_loss = nn.MSELoss(reduction='mean')   # you can also use BCEWithLogitsLoss if you prefer

def vae_reconstruction_loss(recon, target):
    """Pixel‑wise reconstruction loss (MSE)."""
    return mse_loss(recon, target)

def vae_kl_loss(mu, logvar):
    """KL divergence between N(μ,σ²) and N(0,1)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# ---- GAN losses -------------------------------------------------------
bce_logits = nn.BCEWithLogitsLoss()

def discriminator_loss(real_logits, fake_logits):
    """Standard GAN loss for the discriminator."""
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)
    loss_real = bce_logits(real_logits, real_labels)
    loss_fake = bce_logits(fake_logits, fake_labels)
    return (loss_real + loss_fake) * 0.5

def generator_adversarial_loss(fake_logits):
    """Generator wants the discriminator to think its fakes are real."""
    real_labels = torch.ones_like(fake_logits)   # “pretend they are real”
    return bce_logits(fake_logits, real_labels)