from torch import nn
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