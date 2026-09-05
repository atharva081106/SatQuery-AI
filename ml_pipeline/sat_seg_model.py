import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DoubleConv(nn.Module):
    """(Convolution => [BN] => LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SatSegNet(nn.Module):
    """
    SatSegNet: High-Performance Multi-Scale U-Net tailored for Satellite & Aerial Semantic Segmentation.
    Outputs dense pixel probabilities across 6 remote sensing land-cover classes.
    """
    def __init__(self, n_channels=3, n_classes=6):
        super(SatSegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))

        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(32, 16)

        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder
        d3 = self.conv_up1(torch.cat([x3, self.up1(x4)], dim=1))
        d2 = self.conv_up2(torch.cat([x2, self.up2(d3)], dim=1))
        d1 = self.conv_up3(torch.cat([x1, self.up3(d2)], dim=1))

        logits = self.outc(d1)
        return logits

class CombinedDiceLoss(nn.Module):
    """
    Combined Cross-Entropy + Multiclass Soft Dice Loss.
    Optimal for handling satellite class imbalance.
    """
    def __init__(self, weight=None, num_classes=6, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        
        # Soft Dice
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)
        
        return 0.5 * ce_loss + 0.5 * dice_loss

class SatelliteSegDataset(Dataset):
    """PyTorch Dataset for Satellite Multi-Class Segmentation"""
    def __init__(self, data_dir, split="train", target_size=(128, 128)):
        self.split = split
        self.target_size = target_size
        self.img_dir = os.path.join(data_dir, split, "images")
        self.mask_dir = os.path.join(data_dir, split, "masks")
        
        self.images = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        base = os.path.basename(img_path).replace(f"{self.split}_tile_", f"{self.split}_mask_")
        mask_path = os.path.join(self.mask_dir, base)
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        if self.target_size:
            image = image.resize(self.target_size, Image.BILINEAR)
        img_np = np.array(image, dtype=np.float32) / 255.0
        # HWC -> CHW
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        
        # Load Mask
        mask = Image.open(mask_path)
        if self.target_size:
            mask = mask.resize(self.target_size, Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np)
        
        return img_tensor, mask_tensor
