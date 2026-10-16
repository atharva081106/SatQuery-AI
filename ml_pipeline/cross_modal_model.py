#!/usr/bin/env python3
"""
ml_pipeline/cross_modal_model.py

SatQueryOpticalSARNet: Modality-Specific Feature Fusion Network for Joint Optical-SAR Analysis.
Architecture:
  - Modality-Specific Encoders:
      * Optical Branch: 3-channel RGB multi-scale spectral extractor
      * SAR Branch: Log-scaled, speckle-filtered radar backscatter extractor (VV/VH or 1-channel intensity)
  - Cross-Modal Feature Fusion:
      * Squeeze-and-Excitation cross-attention channel gating
      * Multi-scale skip connections
  - Multi-Target Land-Cover Head:
      * Channel 0: Built-up / Urban infrastructure
      * Channel 1: Water bodies / Hydrology
      * Channel 2: Vegetation / Forest canopy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, leak: bool = False):
        super().__init__()
        act = nn.LeakyReLU(0.1, inplace=True) if leak else nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SpeckleFilter(nn.Module):
    """
    Differentiable 2D local mean/variance adaptive speckle suppression filter
    approximating a spatial Lee filter for synthetic aperture radar backscatter.
    """
    def __init__(self, channels: int = 1, kernel_size: int = 5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        kernel = torch.ones((channels, 1, kernel_size, kernel_size), dtype=torch.float32) / (kernel_size * kernel_size)
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local mean
        mean = F.conv2d(x, self.kernel, padding=self.pad, groups=self.channels)
        # Local variance
        mean_sq = F.conv2d(x ** 2, self.kernel, padding=self.pad, groups=self.channels)
        var = F.relu(mean_sq - mean ** 2)
        # Adaptive weight
        noise_var = torch.mean(var, dim=[2, 3], keepdim=True) * 0.5
        weight = var / (var + noise_var + 1e-6)
        filtered = mean + weight * (x - mean)
        return filtered

class ChannelGate(nn.Module):
    """Squeeze-and-Excitation cross-modal channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class SatQueryOpticalSARNet(nn.Module):
    """
    Deep Optical-SAR Dual-Encoder Feature Fusion Network.
    Trained on BigEarthNet.txt (arXiv:2603.29630) Sentinel-1 + Sentinel-2 co-registered pairs.
    """
    def __init__(self, sar_channels: int = 1, num_classes: int = 3):
        super().__init__()
        self.sar_channels = sar_channels
        self.num_classes = num_classes

        # ── Optical Branch (RGB) ─────────────────────────────────────────────
        self.opt_conv0 = DoubleConv(3, 32, leak=False)
        self.opt_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64, leak=False))
        self.opt_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, leak=False))

        # ── SAR Branch (Microwave Backscatter) ──────────────────────────────
        self.sar_filter = SpeckleFilter(channels=sar_channels, kernel_size=5)
        self.sar_conv0 = DoubleConv(sar_channels, 32, leak=True)
        self.sar_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64, leak=True))
        self.sar_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, leak=True))

        # ── Cross-Modal Fusion Bottleneck ───────────────────────────────────
        self.fuse_gate = ChannelGate(256)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ── Decoder with Multi-Modal Skip Connections ───────────────────────
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(64 + 64 + 64, 64)  # 64(up) + 64(opt) + 64(sar)

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up0 = DoubleConv(32 + 32 + 32, 32)  # 32(up) + 32(opt) + 32(sar)

        # Final multi-target classifier (0: Built-up, 1: Water, 2: Vegetation)
        self.out_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        # Preprocess SAR: log-scale backscatter
        sar_log = torch.sign(sar) * torch.log1p(torch.abs(sar))
        sar_filtered = self.sar_filter(sar_log)

        # Optical path
        opt_f0 = self.opt_conv0(optical)         # [B, 32, H, W]
        opt_f1 = self.opt_down1(opt_f0)          # [B, 64, H/2, W/2]
        opt_f2 = self.opt_down2(opt_f1)          # [B, 128, H/4, W/4]

        # SAR path
        sar_f0 = self.sar_conv0(sar_filtered)    # [B, 32, H, W]
        sar_f1 = self.sar_down1(sar_f0)          # [B, 64, H/2, W/2]
        sar_f2 = self.sar_down2(sar_f1)          # [B, 128, H/4, W/4]

        # Fusion at bottleneck
        bottleneck = torch.cat([opt_f2, sar_f2], dim=1) # [B, 256, H/4, W/4]
        bottleneck = self.fuse_gate(bottleneck)
        fused = self.fuse_conv(bottleneck)               # [B, 128, H/4, W/4]

        # Decoder Stage 1
        x = self.up1(fused)                              # [B, 64, H/2, W/2]
        x = torch.cat([x, opt_f1, sar_f1], dim=1)        # [B, 192, H/2, W/2]
        x = self.conv_up1(x)                             # [B, 64, H/2, W/2]

        # Decoder Stage 0
        x = self.up0(x)                                  # [B, 32, H, W]
        x = torch.cat([x, opt_f0, sar_f0], dim=1)        # [B, 96, H, W]
        x = self.conv_up0(x)                             # [B, 32, H, W]

        # Output logits
        logits = self.out_head(x)                        # [B, num_classes, H, W]
        return logits

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = SatQueryOpticalSARNet(sar_channels=1, num_classes=3)
    p_count = count_parameters(net)
    print(f"SatQueryOpticalSARNet initialized. Total trainable parameters: {p_count:,}")

    x_opt = torch.randn(2, 3, 256, 256)
    x_sar = torch.randn(2, 1, 256, 256)
    out = net(x_opt, x_sar)
    print(f"Input Optical: {x_opt.shape}, SAR: {x_sar.shape} -> Output Logits: {out.shape}")
