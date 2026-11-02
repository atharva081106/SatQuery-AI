"""
ml_pipeline/change_model_sota.py

State-of-the-Art (SOTA) Remote Sensing Bi-Temporal Change Detection Architecture:
SatQueryChangeNet-SOTA (Spatial-Temporal Cross-Attention Difference Network).

Key Innovations vs Baseline:
1. Directional + Correlational Difference Fusion (ST-CADM):
   Replaces naive |f2 - f1| with [f1, f2, |f2 - f1|, f1 * f2] to preserve semantic trajectory.
2. Bidirectional Spatial-Temporal Cross-Attention:
   Enables T1 features to attend to T2 features and vice-versa, suppressing seasonal/lighting
   radiometric discrepancies while highlighting true structural alterations.
3. Multi-Scale Feature Pyramid Decoder (H/2, H/4, H/8, H/16):
   Captures both localized changes (individual buildings, vehicles) and extensive changes (flood zones, deforested regions).
4. SOTA Loss Formulations:
   Combined Focal Loss (gamma=2.0) + Soft Dice Loss for extreme foreground imbalance (10-15% change rate).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> LeakyReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class CrossTemporalAttentionBlock(nn.Module):
    """
    Bidirectional Spatial-Temporal Cross-Attention between Epoch 1 (T1) and Epoch 2 (T2).
    Computes query-key-value attention across temporal feature maps to filter out illumination
    and seasonal variations while amplifying genuine semantic land-cover transitions.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.k1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.v1 = nn.Conv2d(channels, channels, 1, bias=False)

        self.q2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.k2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.v2 = nn.Conv2d(channels, channels, 1, bias=False)

        self.proj1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        B, C, H, W = f1.shape
        N = H * W

        # Spatial-temporal cross attention: f1 queries f2, f2 queries f1
        # Project queries, keys, values
        q1 = self.q1(f1).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2) # [B, heads, N, head_dim]
        k2 = self.k2(f2).view(B, self.num_heads, self.head_dim, N)                     # [B, heads, head_dim, N]
        v2 = self.v2(f2).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2) # [B, heads, N, head_dim]

        q2 = self.q2(f2).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k1 = self.k1(f1).view(B, self.num_heads, self.head_dim, N)
        v1 = self.v1(f1).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn12 = torch.softmax(torch.matmul(q1, k2) * scale, dim=-1) # [B, heads, N, N]
        out1 = torch.matmul(attn12, v2).permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        attn21 = torch.softmax(torch.matmul(q2, k1) * scale, dim=-1)
        out2 = torch.matmul(attn21, v1).permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        f1_att = self.norm1(f1 + self.proj1(out1))
        f2_att = self.norm2(f2 + self.proj2(out2))
        return f1_att, f2_att

class SpatialTemporalCADM(nn.Module):
    """
    Spatial-Temporal Cross-Attention Difference Module (ST-CADM).
    Extracts deep interactions:
      Diff = Conv([f1, f2, |f2 - f1|, f1 * f2])
    """
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention and in_channels >= 64:
            self.attn = CrossTemporalAttentionBlock(in_channels)
        else:
            self.attn = None

        # 4x channels because [f1, f2, |f2 - f1|, f1 * f2]
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        if self.attn is not None and f1.shape[2] <= 64:
            f1, f2 = self.attn(f1, f2)

        abs_diff = torch.abs(f2 - f1)
        hadamard = f1 * f2
        combined = torch.cat([f1, f2, abs_diff, hadamard], dim=1)
        return self.fuse_conv(combined)

class SatQueryChangeNetSOTA(nn.Module):
    """
    Next-Generation SOTA Siamese Remote Sensing Change Detection Network.
    4-Stage Multi-Scale Encoder + ST-CADM Difference Fusion + Feature Pyramid Decoder.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Shared Siamese Encoder
        self.inc = DoubleConv(in_channels, 32)                                # Stage 0: H, W
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))     # Stage 1: H/2, W/2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))    # Stage 2: H/4, W/4
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))   # Stage 3: H/8, W/8
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))   # Stage 4: H/16, W/16

        # Difference Modules at each scale
        self.cadm4 = SpatialTemporalCADM(512, 256, use_attention=True)
        self.cadm3 = SpatialTemporalCADM(256, 128, use_attention=True)
        self.cadm2 = SpatialTemporalCADM(128, 64, use_attention=False)
        self.cadm1 = SpatialTemporalCADM(64, 32, use_attention=False)
        self.cadm0 = SpatialTemporalCADM(32, 16, use_attention=False)

        # Multi-scale Decoder with Skip Connections
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(32 + 32, 32)

        self.up0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up0 = DoubleConv(16 + 16, 16)

        # Final Change Mask Classifier
        self.out_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Auxiliary Change Representation (for CDVQA Feature Head)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.change_vector_proj = nn.Linear(256, 512)

    def extract_encoder_features(self, x: torch.Tensor):
        c0 = self.inc(x)
        c1 = self.down1(c0)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        return c0, c1, c2, c3, c4

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, return_change_vector: bool = False):
        # Extract Siamese multi-scale representations
        c0_1, c1_1, c2_1, c3_1, c4_1 = self.extract_encoder_features(x1)
        c0_2, c1_2, c2_2, c3_2, c4_2 = self.extract_encoder_features(x2)

        # Multi-scale ST-CADM Difference Features
        d4 = self.cadm4(c4_1, c4_2) # [B, 256, H/16, W/16]
        d3 = self.cadm3(c3_1, c3_2) # [B, 128, H/8, W/8]
        d2 = self.cadm2(c2_1, c2_2) # [B, 64, H/4, W/4]
        d1 = self.cadm1(c1_1, c1_2) # [B, 32, H/2, W/2]
        d0 = self.cadm0(c0_1, c0_2) # [B, 16, H, W]

        # Decoder Stage 3
        u3 = self.up3(d4)
        if u3.shape[2:] != d3.shape[2:]:
            u3 = F.interpolate(u3, size=d3.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up3(torch.cat([u3, d3], dim=1))

        # Decoder Stage 2
        u2 = self.up2(x)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up2(torch.cat([u2, d2], dim=1))

        # Decoder Stage 1
        u1 = self.up1(x)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up1(torch.cat([u1, d1], dim=1))

        # Decoder Stage 0
        u0 = self.up0(x)
        if u0.shape[2:] != d0.shape[2:]:
            u0 = F.interpolate(u0, size=d0.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up0(torch.cat([u0, d0], dim=1))

        logits = self.out_head(x)

        if return_change_vector:
            # Pooled change vector for CDVQA
            pooled = self.global_pool(d4).flatten(1)
            c_vec = self.change_vector_proj(pooled)
            return logits, c_vec

        return logits

# ─────────────────────────────────────────────────────────────────────────────
# SOTA Losses for Extreme Class Imbalance
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss with alpha balancing and gamma focusing parameter."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        return (focal_weight * bce).mean()

class SoftDiceLoss(nn.Module):
    """Differentiable Soft Dice Loss with Laplace smoothing."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class SOTAChangeLoss(nn.Module):
    """Combined Focal (0.5) + Soft Dice (0.5) Loss."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.focal(logits, targets) + 0.5 * self.dice(logits, targets)

class SatQueryChangeNetResNetSOTA(nn.Module):
    """
    Next-Generation SOTA Siamese Change Detection Network with Pretrained
    ResNet-18 Backbone + Multi-Scale ST-CADM Difference Fusion + Feature Pyramid Decoder.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64 ch, /4
        self.layer2 = backbone.layer2  # 128 ch, /8
        self.layer3 = backbone.layer3  # 256 ch, /16
        self.layer4 = backbone.layer4  # 512 ch, /32

        # ST-CADM Difference Fusion at each scale
        self.cadm4 = SpatialTemporalCADM(512, 256, use_attention=True)
        self.cadm3 = SpatialTemporalCADM(256, 128, use_attention=True)
        self.cadm2 = SpatialTemporalCADM(128, 64, use_attention=False)
        self.cadm1 = SpatialTemporalCADM(64, 32, use_attention=False)

        # Feature Pyramid Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(32 + 32, 32)

        # Final upsample to full resolution (256x256)
        self.up0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.out_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.change_vector_proj = nn.Linear(256, 512)

    def extract_features(self, x: torch.Tensor):
        c1 = self.relu(self.bn1(self.conv1(x)))     # [B, 64, H/2, W/2]
        c2 = self.layer1(self.maxpool(c1))         # [B, 64, H/4, W/4]
        c3 = self.layer2(c2)                       # [B, 128, H/8, W/8]
        c4 = self.layer3(c3)                       # [B, 256, H/16, W/16]
        c5 = self.layer4(c4)                       # [B, 512, H/32, W/32]
        return c1, c2, c3, c4, c5

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, return_change_vector: bool = False):
        c1_1, c2_1, c3_1, c4_1, c5_1 = self.extract_features(x1)
        c1_2, c2_2, c3_2, c4_2, c5_2 = self.extract_features(x2)

        d4 = self.cadm4(c5_1, c5_2)  # [B, 256, H/32, W/32]
        d3 = self.cadm3(c4_1, c4_2)  # [B, 128, H/16, W/16]
        d2 = self.cadm2(c3_1, c3_2)  # [B, 64, H/8, W/8]
        d1 = self.cadm1(c2_1, c2_2)  # [B, 32, H/4, W/4]

        u3 = self.up3(d4)
        if u3.shape[2:] != d3.shape[2:]:
            u3 = F.interpolate(u3, size=d3.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(x)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(x)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up1(torch.cat([u1, d1], dim=1))

        u0 = self.up0(x)  # [B, 16, H/2, W/2]
        u_full = F.interpolate(u0, scale_factor=2.0, mode="bilinear", align_corners=False) # [B, 16, H, W]
        logits = self.out_head(u_full)

        if return_change_vector:
            pooled = self.global_pool(d4).flatten(1)
            c_vec = self.change_vector_proj(pooled)
            return logits, c_vec

        return logits

if __name__ == "__main__":
    net = SatQueryChangeNetResNetSOTA(pretrained=True)
    p_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"SatQueryChangeNetResNetSOTA initialized successfully. Total trainable parameters: {p_count:,}")

    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    out, c_vec = net(x1, x2, return_change_vector=True)
    print(f"Input: {x1.shape} -> Output Change Logits: {out.shape}, Change Vector: {c_vec.shape}")

