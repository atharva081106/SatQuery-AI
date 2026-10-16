"""
ml_pipeline/change_model.py

Learned Bi-Temporal Remote Sensing Change Detection Architecture.
Implements SatQueryChangeNet (Siamese Feature Difference U-Net).
Reference: arXiv:2112.06343 / IEEE TGRS / IGARSS 2022.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class SatQueryChangeNet(nn.Module):
    """
    Siamese Feature Difference Network for Remote Sensing Change Detection.
    Extracts deep representations from T1 and T2, computes multi-scale feature
    discrepancy, and decodes a high-fidelity change probability mask.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Shared Siamese Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Difference Fusion & Decoder
        self.diff_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(64 + 128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up1 = DoubleConv(32 + 64, 32)

        self.up0 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.outc = nn.Sequential(
            DoubleConv(32 + 32, 16),
            nn.Conv2d(16, 1, 1)
        )

    def extract_features(self, x):
        f0 = self.inc(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        return f0, f1, f2, f3

    def forward(self, x1, x2):
        f0_1, f1_1, f2_1, f3_1 = self.extract_features(x1)
        f0_2, f1_2, f2_2, f3_2 = self.extract_features(x2)

        # Multi-scale feature difference
        d3 = self.diff_conv3(torch.abs(f3_2 - f3_1))
        d2 = torch.abs(f2_2 - f2_1)
        d1 = torch.abs(f1_2 - f1_1)
        d0 = torch.abs(f0_2 - f0_1)

        u2 = self.up2(d3)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(x)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up1(torch.cat([u1, d1], dim=1))

        u0 = self.up0(x)
        if u0.shape[2:] != d0.shape[2:]:
            u0 = F.interpolate(u0, size=d0.shape[2:], mode="bilinear", align_corners=False)
        logits = self.outc(torch.cat([u0, d0], dim=1))
        return logits
