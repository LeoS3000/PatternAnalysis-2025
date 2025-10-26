import torch
import torch.nn as nn
import torch.nn.functional as F

# LocalizationModule3D: 3x3x3 conv, then 1x1x1 conv halving channels
class LocalizationModule3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels // 2, kernel_size=1)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# ContextModule3D: Pre-activation residual block with InstanceNorm, LeakyReLU, Dropout, skip connection
class ContextModule3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(p=dropout_p)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.act1(self.norm1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.act2(self.norm2(out))
        out = self.conv2(out)
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
        out += identity
        return out

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- ENCODER PATH ---
        self.inc = ContextModule3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        self.down4 = Down3D(512, 1024)

        # --- DECODER PATH ---
        # Up 1
        self.up1 = Up3D(1024, 512)
        # Concatenated size is 512 (from up) + 512 (skip) = 1024
        self.loc1 = LocalizationModule3D(1024, 512) # Output has 256 channels

        # Up 2 (CORRECTED)
        self.up2 = Up3D(256, 256) # <-- FIX: in_channels should be 256 (from loc1)
        # Concatenated size is 128 (from up) + 256 (skip) = 384
        self.loc2 = LocalizationModule3D(384, 256) # Output has 128 channels

        # Up 3 (CORRECTED)
        self.up3 = Up3D(128, 128) # <-- FIX: in_channels should be 128 (from loc2)
        # Concatenated size is 64 (from up) + 128 (skip) = 192
        self.loc3 = LocalizationModule3D(192, 128) # Output has 64 channels

        # Up 4 (CORRECTED)
        self.up4 = Up3D(64, 64) # <-- FIX: in_channels should be 64 (from loc3)
        # Concatenated size is 32 (from up) + 64 (skip) = 96
        self.loc4 = LocalizationModule3D(96, 64) # Output has 32 channels

        # --- OUTPUT AND DEEP SUPERVISION ---
        self.outc = OutConv3D(32, n_classes)
        self.seg2 = nn.Conv3d(64, n_classes, kernel_size=1)
        self.seg3 = nn.Conv3d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.loc1(x)
        x = self.up2(x, x3)
        x = self.loc2(x)
        seg3_out = self.seg3(x)  # Deep supervision output 1 (feature map 128->64)
        x = self.up3(x, x2)
        x = self.loc3(x)
        seg2_out = self.seg2(x)  # Deep supervision output 2 (feature map 64->32)
        x = self.up4(x, x1)
        x = self.loc4(x)
        logits = self.outc(x)

        # Upsample deep supervision outputs to match final output size
        seg3_up = F.interpolate(seg3_out, size=logits.shape[2:], mode='trilinear', align_corners=True)
        seg2_up = F.interpolate(seg2_out, size=logits.shape[2:], mode='trilinear', align_corners=True)
        # Sum all outputs
        logits = logits + seg2_up + seg3_up
        return logits

# 3D versions of the building blocks

class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2 for 3D"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then context module for 3D"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            ContextModule3D(out_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv for 3D"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Instead of transposed conv, use interpolate + conv
        self.up_conv = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv = None  # Will be replaced by localization module in next step
        self.out_channels = out_channels

    def forward(self, x1, x2):
        # Upsample by trilinear interpolation
        x1 = F.interpolate(x1, scale_factor=2, mode='trilinear', align_corners=True)
        x1 = self.up_conv(x1)
        # Pad if needed
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x1], dim=1)
        # The localization module will be applied after concatenation in the next step
        return x

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)