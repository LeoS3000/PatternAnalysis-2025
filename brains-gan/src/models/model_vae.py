import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 256x256 -> 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),# 16x16 -> 8x8
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(8*8*512, latent_dim)
        self.fc_logvar = nn.Linear(8*8*512, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 8*8*512)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 128x128 -> 256x256
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 8, 8)
        x = self.deconv(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
