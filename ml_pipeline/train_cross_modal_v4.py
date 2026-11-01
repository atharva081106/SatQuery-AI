#!/usr/bin/env python3
"""
ml_pipeline/train_cross_modal_v4.py

SatQuery AI V4 — Improved Optical-SAR Training Pipeline
Fixes over v1 (train_cross_modal.py):
  1. HWC→CHW mask transposition bug fixed (targets now correctly [3, H, W])
  2. Class-weighted BCE loss (invfreq weighting for built-up/water/vegetation)
  3. Data augmentation: random flips, 90-degree rotations, SAR speckle noise
  4. Extended training with CosineAnnealingLR warmup schedule
  5. Per-class optimal threshold tuned on validation split
  6. Best-checkpoint saving by val_miou (not just last epoch)

Output checkpoint: ml_pipeline/checkpoints/v4/optical_sar/optical_sar_net_v4.pth
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from cross_modal_model import SatQueryOpticalSARNet
from data.bigearthnet_loader import BigEarthNetDataset
from train_cross_modal import generate_spatial_targets

# ──────────────────────────────────────────────────────────────────────────────
# FIX 1: Corrected target shape (HWC→CHW) + ESA WorldCover GT when available
# ──────────────────────────────────────────────────────────────────────────────
def build_target_hwc_to_chw(item: dict, s2_arr: np.ndarray, s1_arr: np.ndarray,
                              img_size: int = 256) -> np.ndarray:
    """
    Returns target as [3, H, W] float32 with values in {0.0, 1.0}.
    Priority: authentic ESA WorldCover ground truth > CLC radiometric derivation.
    """
    if "target_mask" in item:
        raw = item["target_mask"].numpy() if hasattr(item["target_mask"], "numpy") else item["target_mask"]
        # raw is [H, W, 3] (HWC) — transpose to [3, H, W]
        if raw.ndim == 3 and raw.shape[2] == 3:
            tgt_hwc = raw  # [120, 120, 3]
        elif raw.ndim == 3 and raw.shape[0] == 3:
            return raw.astype(np.float32)  # already CHW
        else:
            tgt_hwc = raw
        # Resize each channel independently then stack
        channels = []
        for c in range(3):
            ch = Image.fromarray(tgt_hwc[..., c]).resize((img_size, img_size), Image.NEAREST)
            channels.append(np.array(ch, dtype=np.float32))
        return np.stack(channels, axis=0)  # [3, H, W]  ← FIX
    else:
        # Fall back to CLC radiometric derivation — already [3, H, W]
        tgt = generate_spatial_targets(s2_arr, s1_arr, item["clc_labels"])
        return tgt  # [3, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# FIX 2: Class-weighted BCE loss (inverse-frequency weighting)
# ──────────────────────────────────────────────────────────────────────────────
class ClassWeightedBCEDiceLoss(nn.Module):
    """
    Combined weighted BCE + Dice loss for heavily imbalanced multi-label segmentation.
    Class weights are computed from actual pixel coverage statistics:
      Built-up:  4.6% → weight ≈ 16x
      Water:    18.8% → weight ≈  4x
      Vegetation: 75.2% → weight ≈  1.3x
    pos_weights must be shape [3] — internally reshaped to [1, 3, 1, 1] for broadcast
    with [B, 3, H, W] tensors.
    """
    def __init__(self, pos_weights: torch.Tensor, smooth: float = 1.0, dice_weight: float = 0.6):
        super().__init__()
        # Store as buffer so it moves with .to(device) and serializes properly
        # Shape: [1, 3, 1, 1] for broadcast over [B, 3, H, W]
        self.register_buffer("pos_weights", pos_weights.view(1, -1, 1, 1))
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Manual weighted BCE (avoids the shape mismatch from nn.BCEWithLogitsLoss pos_weight)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights.expand_as(targets)
        )

        probs = torch.sigmoid(logits)
        # Dice per class, then mean
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = torch.mean(1.0 - dice)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ──────────────────────────────────────────────────────────────────────────────
# FIX 3: Augmentation-aware collate_fn
# ──────────────────────────────────────────────────────────────────────────────
def apply_augmentation(opt_t: torch.Tensor, sar_t: torch.Tensor,
                        tgt_t: torch.Tensor, augment: bool) -> tuple:
    """
    Applies identical spatial augmentations to optical, SAR, and target tensors.
    opt_t: [3, H, W], sar_t: [1, H, W], tgt_t: [3, H, W]
    Returns: augmented tensors
    """
    if not augment:
        return opt_t, sar_t, tgt_t

    # Random horizontal flip
    if random.random() > 0.5:
        opt_t = torch.flip(opt_t, dims=[2])
        sar_t = torch.flip(sar_t, dims=[2])
        tgt_t = torch.flip(tgt_t, dims=[2])

    # Random vertical flip
    if random.random() > 0.5:
        opt_t = torch.flip(opt_t, dims=[1])
        sar_t = torch.flip(sar_t, dims=[1])
        tgt_t = torch.flip(tgt_t, dims=[1])

    # Random 90-degree rotation (0/90/180/270)
    k = random.randint(0, 3)
    if k > 0:
        opt_t = torch.rot90(opt_t, k, dims=[1, 2])
        sar_t = torch.rot90(sar_t, k, dims=[1, 2])
        tgt_t = torch.rot90(tgt_t, k, dims=[1, 2])

    # SAR-specific: add mild multiplicative speckle noise
    if random.random() > 0.5:
        speckle = 1.0 + 0.05 * torch.randn_like(sar_t)
        sar_t = sar_t * speckle

    return opt_t, sar_t, tgt_t


def collate_fn_v4(batch: list, img_size: int = 256, augment: bool = False) -> dict:
    """Fixed collate: correctly transposes [H,W,3] targets to [3,H,W]."""
    opt_list, sar_list, tgt_list = [], [], []

    for item in batch:
        # Optical: resize to img_size × img_size
        s2_img = item["s2_image"].resize((img_size, img_size), Image.Resampling.BILINEAR)
        s2_arr = np.array(s2_img, dtype=np.float32)

        # SAR: extract channel 0 (VV), resize
        s1_raw = item["s1_sar"].numpy() if hasattr(item["s1_sar"], "numpy") else item["s1_sar"]
        if s1_raw.ndim == 3:
            s1_img = Image.fromarray(s1_raw[..., 0]).resize((img_size, img_size), Image.Resampling.BILINEAR)
        else:
            s1_img = Image.fromarray(s1_raw).resize((img_size, img_size), Image.Resampling.BILINEAR)
        s1_arr = np.array(s1_img, dtype=np.float32)

        opt_t = torch.from_numpy(s2_arr / 255.0).permute(2, 0, 1)   # [3, H, W]
        sar_t = torch.from_numpy(s1_arr).unsqueeze(0)                # [1, H, W]

        # ── FIX: correct CHW target ──
        tgt_arr = build_target_hwc_to_chw(item, s2_arr, s1_arr, img_size)
        tgt_t = torch.from_numpy(tgt_arr)                            # [3, H, W]

        # ── FIX: augmentation ──
        opt_t, sar_t, tgt_t = apply_augmentation(opt_t, sar_t, tgt_t, augment=augment)

        opt_list.append(opt_t)
        sar_list.append(sar_t)
        tgt_list.append(tgt_t)

    return {
        "optical": torch.stack(opt_list),   # [B, 3, H, W]
        "sar": torch.stack(sar_list),        # [B, 1, H, W]
        "targets": torch.stack(tgt_list),    # [B, 3, H, W]
    }


# ──────────────────────────────────────────────────────────────────────────────
# FIX 5: Per-class threshold tuning on validation set
# ──────────────────────────────────────────────────────────────────────────────
def tune_thresholds(model: nn.Module, val_loader: DataLoader, device: str) -> list:
    """Sweeps thresholds [0.2..0.8] per class on val split to maximise IoU."""
    model.eval()
    all_probs = [[], [], []]
    all_targets = [[], [], []]

    with torch.no_grad():
        for batch in val_loader:
            opt = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            tgt = batch["targets"].cpu().numpy()

            logits = model(opt, sar)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, 3, H, W]

            for c in range(3):
                all_probs[c].append(probs[:, c])
                all_targets[c].append(tgt[:, c])

    best_thresholds = []
    for c in range(3):
        p = np.concatenate(all_probs[c]).ravel()
        t = np.concatenate(all_targets[c]).ravel()
        best_iou, best_thr = -1.0, 0.5
        for thr in np.arange(0.2, 0.85, 0.05):
            pred = (p > thr).astype(np.float32)
            inter = (pred * t).sum()
            union = (pred + t - pred * t).sum()
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou, best_thr = iou, float(thr)
        best_thresholds.append(best_thr)
        class_names = ["built_up", "water", "vegetation"]
        print(f"  [Threshold] {class_names[c]}: best_thr={best_thr:.2f}, val_iou={best_iou*100:.1f}%")

    return best_thresholds


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_v4(epochs: int = 15, batch_size: int = 8, lr: float = 5e-4,
             max_samples: int = None, img_size: int = 256):
    print("=" * 70)
    print("  SatQuery AI V4 — Improved Optical-SAR Training Pipeline")
    print("  Fixes: HWC→CHW mask, class-weighted loss, augmentation, LR schedule")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Training on: {device.upper()}")

    # Datasets
    train_ds = BigEarthNetDataset(split="train", max_samples=max_samples)
    val_ds = BigEarthNetDataset(split="val", max_samples=None)
    print(f"[Dataset] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_fn_v4(b, img_size=img_size, augment=True)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_fn_v4(b, img_size=img_size, augment=False)
    )

    # ── FIX 2: Compute inverse-frequency class weights from training data ──
    print("[Weights] Computing per-class pixel frequency weights from train set...")
    built_total, water_total, veg_total, all_total = 0, 0, 0, 0
    sample_count = min(100, len(train_ds))
    for idx in range(sample_count):
        item = train_ds[idx]
        tgt = build_target_hwc_to_chw(item,
                                       np.zeros((img_size, img_size, 3), dtype=np.float32),
                                       np.zeros((img_size, img_size), dtype=np.float32),
                                       img_size=120)  # use native res for stats
        built_total += (tgt[0] > 0.5).sum()
        water_total += (tgt[1] > 0.5).sum()
        veg_total += (tgt[2] > 0.5).sum()
        all_total += tgt[0].size

    # Inverse frequency weights (higher weight = rarer class)
    eps = 1e-6
    freq_built = (built_total + eps) / (all_total + eps)
    freq_water = (water_total + eps) / (all_total + eps)
    freq_veg = (veg_total + eps) / (all_total + eps)

    # pos_weight = neg_pixels / pos_pixels per class
    w_built = (1.0 - freq_built) / (freq_built + eps)
    w_water = (1.0 - freq_water) / (freq_water + eps)
    w_veg = (1.0 - freq_veg) / (freq_veg + eps)

    # Cap weights to avoid extreme instability
    w_built = min(w_built, 25.0)
    w_water = min(w_water, 12.0)
    w_veg = min(w_veg, 4.0)

    print(f"  Built-up freq: {freq_built*100:.1f}% → pos_weight: {w_built:.1f}x")
    print(f"  Water freq:    {freq_water*100:.1f}% → pos_weight: {w_water:.1f}x")
    print(f"  Vegetation freq: {freq_veg*100:.1f}% → pos_weight: {w_veg:.1f}x")

    pos_weights = torch.tensor([w_built, w_water, w_veg], dtype=torch.float32)
    criterion = ClassWeightedBCEDiceLoss(pos_weights=pos_weights, dice_weight=0.6).to(device)

    # Model
    model = SatQueryOpticalSARNet(sar_channels=1, num_classes=3).to(device)
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] SatQueryOpticalSARNet parameters: {p_count:,}")

    # ── FIX 4: LR schedule — warmup + cosine annealing ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,          # 10% warmup
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy="cos"
    )

    checkpoint_dir = ML_DIR / "checkpoints" / "v4" / "optical_sar"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_miou = -1.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training ──
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            opt = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            logits = model(opt, sar)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_ious_per_class = [[], [], []]  # built, water, veg

        with torch.no_grad():
            for batch in val_loader:
                opt = batch["optical"].to(device)
                sar = batch["sar"].to(device)
                targets = batch["targets"].to(device)

                logits = model(opt, sar)
                val_loss += criterion(logits, targets).item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.45).float()

                # Per-class IoU
                for c in range(3):
                    inter = (preds[:, c] * targets[:, c]).sum().item()
                    union = (preds[:, c] + targets[:, c] - preds[:, c] * targets[:, c]).sum().item()
                    if union > 0:
                        val_ious_per_class[c].append(inter / union)

        avg_val_loss = val_loss / len(val_loader)
        per_class_ious = [
            float(np.mean(v)) if v else 0.0
            for v in val_ious_per_class
        ]
        avg_val_miou = float(np.mean(per_class_ious))

        cur_lr = scheduler.get_last_lr()[0]
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_miou": round(avg_val_miou, 4),
            "val_iou_built_up": round(per_class_ious[0], 4),
            "val_iou_water": round(per_class_ious[1], 4),
            "val_iou_vegetation": round(per_class_ious[2], 4),
            "lr": round(cur_lr, 6),
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val mIoU: {avg_val_miou*100:.2f}% "
            f"(B:{per_class_ious[0]*100:.1f}% W:{per_class_ious[1]*100:.1f}% V:{per_class_ious[2]*100:.1f}%) "
            f"LR: {cur_lr:.5f}"
        )

        # ── FIX 6: Best checkpoint by val_miou ──
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "SatQueryOpticalSARNet-v4",
                "version": "v4",
                "parameters": p_count,
                "sar_channels": 1,
                "num_classes": 3,
                "classes": ["built_up", "water", "vegetation"],
                "best_val_miou": best_val_miou,
                "best_epoch": best_epoch,
                "history": history,
                "class_weights": [w_built, w_water, w_veg],
                "img_size": img_size,
            }, checkpoint_dir / "optical_sar_net_v4_best.pth")
            print(f"  ↑ New best checkpoint saved (epoch {epoch}, mIoU={best_val_miou*100:.2f}%)")

    total_time = round(time.time() - start_time, 2)

    # ── FIX 5: Tune thresholds on validation ──
    print("\n[Threshold Tuning] Finding optimal per-class prediction thresholds on val split...")
    # Reload best checkpoint for tuning
    best_ckpt = checkpoint_dir / "optical_sar_net_v4_best.pth"
    try:
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    best_thresholds = tune_thresholds(model, val_loader, device)

    # Save final checkpoint with tuned thresholds
    final_ckpt_path = checkpoint_dir / "optical_sar_net_v4.pth"
    torch.save({
        **state,
        "optimal_thresholds": best_thresholds,
        "total_training_time_seconds": total_time,
    }, final_ckpt_path)
    print(f"\n[Saved] Final checkpoint: {final_ckpt_path}")
    print(f"  Optimal thresholds: built_up={best_thresholds[0]:.2f}, water={best_thresholds[1]:.2f}, veg={best_thresholds[2]:.2f}")

    # ── Save training report ──
    report = {
        "model_name": "SatQueryOpticalSARNet-v4",
        "version": "v4",
        "improvements": [
            "Fixed HWC->CHW target mask transposition",
            "Class-weighted BCE+Dice loss (built_up:~{:.0f}x, water:~{:.0f}x, veg:~{:.0f}x)".format(w_built, w_water, w_veg),
            "Data augmentation: random flips, 90-deg rotations, SAR speckle noise",
            "OneCycleLR schedule with 10% warmup + cosine annealing",
            "Best-checkpoint saving by validation mIoU",
            "Per-class threshold tuning on validation split",
        ],
        "parameters": p_count,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "total_time_seconds": total_time,
        "best_epoch": best_epoch,
        "best_val_miou": round(best_val_miou * 100, 2),
        "class_weights": {"built_up": round(w_built, 2), "water": round(w_water, 2), "vegetation": round(w_veg, 2)},
        "optimal_thresholds": {"built_up": best_thresholds[0], "water": best_thresholds[1], "vegetation": best_thresholds[2]},
        "history": history,
    }
    report_path = checkpoint_dir / "training_report_v4.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Saved] Training report: {report_path}")

    print("\n" + "=" * 70)
    print(f"  V4 Training Complete | Best Epoch: {best_epoch} | Best Val mIoU: {best_val_miou*100:.2f}%")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatQueryOpticalSARNet V4")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    train_v4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        img_size=args.img_size,
    )
