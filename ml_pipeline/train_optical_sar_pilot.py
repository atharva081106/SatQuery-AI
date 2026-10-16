#!/usr/bin/env python3
"""
ml_pipeline/train_optical_sar_pilot.py

Pilot training script for SatQueryOpticalSARNet on Authentic BigEarthNet-MM (BEN-GE-800) data.
Uses authentic Sentinel-1 SAR (VV), Sentinel-2 Optical (RGB), and authentic ESA WorldCover 10m
ground truth raster masks.
Never substitutes synthetic or heuristic labels.
Saves checkpoint to a separate directory:
  ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# Enforce UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from cross_modal_model import SatQueryOpticalSARNet

TARGET_CLASSES = ["Built-up", "Water", "Vegetation"]

class AuthenticOpticalSARDataset(Dataset):
    """
    Loads authentic Sentinel-1/Sentinel-2 patch pairs and ESA WorldCover ground truth masks.
    """
    def __init__(self, data_root: Path, split: str = "train", target_size: tuple = (128, 128)):
        self.data_root = data_root
        self.split = split
        self.target_size = target_size
        ann_file = data_root / split / f"{split}_annotations.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations not found: {ann_file}")
        with open(ann_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        
        # 1. Load S2 RGB image
        s2_path = self.data_root / item["s2_image_path"]
        s2_img = Image.open(s2_path).convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)
        s2_arr = np.array(s2_img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
        opt_tensor = torch.from_numpy(s2_arr).permute(2, 0, 1)  # [3, H, W]

        # 2. Load S1 SAR
        s1_path = self.data_root / item["s1_sar_path"]
        s1_arr = np.load(s1_path)  # [H, W, 2] (VV, VH)
        # Use VV channel (channel 0)
        vv = s1_arr[..., 0]
        # Resize SAR to target size
        vv_img = Image.fromarray(vv).resize(self.target_size, Image.Resampling.BILINEAR)
        vv_arr = np.array(vv_img, dtype=np.float32)
        sar_tensor = torch.from_numpy(vv_arr).unsqueeze(0)  # [1, H, W]

        # 3. Load authentic ESA WorldCover target mask
        target_path = self.data_root / item["target_mask_path"]
        tgt_arr = np.load(target_path)  # [120, 120, 3]
        # Resize each channel to target size
        resized_channels = []
        for c in range(3):
            ch_img = Image.fromarray(tgt_arr[..., c]).resize(self.target_size, Image.Resampling.NEAREST)
            resized_channels.append(np.array(ch_img, dtype=np.float32))
        tgt_arr = np.stack(resized_channels, axis=0)  # [3, H, W]
        tgt_tensor = torch.from_numpy(tgt_arr)

        return {
            "patch_id": item["patch_id"],
            "optical": opt_tensor,
            "sar": sar_tensor,
            "target": tgt_tensor,
            "clc_labels": item.get("clc_labels", [])
        }

class MultiLabelDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice)

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Computes per-class and mean IoU and Dice scores.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    num_classes = logits.shape[1]
    class_ious = []
    class_dices = []
    
    for c in range(num_classes):
        p = preds[:, c, :, :]
        t = targets[:, c, :, :]
        
        inter = torch.sum(p * t).item()
        union = torch.sum((p + t).clamp(0, 1)).item()
        card = (torch.sum(p) + torch.sum(t)).item()
        
        iou = (inter + 1e-6) / (union + 1e-6)
        dice = (2.0 * inter + 1e-6) / (card + 1e-6)
        
        class_ious.append(iou)
        class_dices.append(dice)
        
    return {
        "class_ious": class_ious,
        "class_dices": class_dices,
        "mean_iou": float(np.mean(class_ious)),
        "mean_dice": float(np.mean(class_dices))
    }

def collate_fn(batch):
    return {
        "patch_id": [item["patch_id"] for item in batch],
        "optical": torch.stack([item["optical"] for item in batch]),
        "sar": torch.stack([item["sar"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "clc_labels": [item["clc_labels"] for item in batch]
    }

def train_and_eval(
    data_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    checkpoint_out_dir: Path = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  SatQuery AI: OpticalSARNet Authentic-Data Pilot Training")
    print(f"  Dataset: BigEarthNet-MM / BEN-GE-800 + ESA WorldCover 10m Ground Truth")
    print(f"  Hardware Device: {device} | Architecture: SatQueryOpticalSARNet")
    print("=" * 70)

    train_ds = AuthenticOpticalSARDataset(data_dir, split="train", target_size=(128, 128))
    val_ds = AuthenticOpticalSARDataset(data_dir, split="val", target_size=(128, 128))

    print(f"[Dataset] Train samples: {len(train_ds)}, Held-out Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    model = SatQueryOpticalSARNet(sar_channels=1, num_classes=3).to(device)
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {p_count:,}")

    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = MultiLabelDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_train_time = time.time()
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        ep_start = time.time()

        for batch in train_loader:
            opt = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            tgt = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(opt, sar)
            loss = bce_loss_fn(logits, tgt) + 0.5 * dice_loss_fn(logits, tgt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, batches)
        ep_duration = time.time() - ep_start

        # ── Held-out Validation Evaluation ───────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                opt = batch["optical"].to(device)
                sar = batch["sar"].to(device)
                tgt = batch["target"].to(device)

                logits = model(opt, sar)
                loss = bce_loss_fn(logits, tgt) + 0.5 * dice_loss_fn(logits, tgt)
                val_loss += loss.item()
                val_batches += 1

                all_logits.append(logits.cpu())
                all_targets.append(tgt.cpu())

        avg_val_loss = val_loss / max(1, val_batches)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(cat_logits, cat_targets)

        ep_record = {
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_mean_iou": round(metrics["mean_iou"], 4),
            "val_mean_dice": round(metrics["mean_dice"], 4),
            "val_iou_built_up": round(metrics["class_ious"][0], 4),
            "val_iou_water": round(metrics["class_ious"][1], 4),
            "val_iou_vegetation": round(metrics["class_ious"][2], 4),
            "duration_seconds": round(ep_duration, 2)
        }
        history.append(ep_record)

        print(f"Epoch {ep}/{epochs} [{ep_duration:.1f}s] - Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val mIoU: {metrics['mean_iou']*100:.2f}% "
              f"(Built-up: {metrics['class_ious'][0]*100:.1f}%, "
              f"Water: {metrics['class_ious'][1]*100:.1f}%, "
              f"Veg: {metrics['class_ious'][2]*100:.1f}%)")

    total_training_time = round(time.time() - start_train_time, 2)
    print(f"\n[Training Complete] Total time: {total_training_time}s")

    # ── Save New Separate Checkpoint ─────────────────────────────────────────
    if checkpoint_out_dir is None:
        checkpoint_out_dir = ML_DIR / "checkpoints" / "satquery_optical_sar_net_authentic_pilot"
    checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoint_out_dir / "optical_sar_net.pth"

    torch.save({
        "model_name": "SatQueryOpticalSARNet-AuthenticPilot-v1",
        "model_state_dict": model.state_dict(),
        "parameters": p_count,
        "sar_channels": 1,
        "num_classes": 3,
        "classes": TARGET_CLASSES,
        "dataset": "BigEarthNet-MM / BEN-GE-800",
        "provenance": "Zenodo DOI 10.5281/zenodo.12941231; Mommert et al 2023; ESA WorldCover 10m v100",
        "training_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "final_val_loss": history[-1]["val_loss"],
        "final_val_iou": history[-1]["val_mean_iou"],
        "history": history,
        "hardware": f"{device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Intel/AMD CPU'})",
        "total_training_time_seconds": total_training_time
    }, ckpt_file)

    report = {
        "pilot_run_name": "OpticalSARNet Authentic-Data Pilot",
        "checkpoint_path": str(ckpt_file),
        "dataset_name": "BEN-GE-800 (BigEarthNet-MM subset)",
        "ground_truth_source": "ESA WorldCover 10m Land Cover v100 (Authentic Raster GT)",
        "provenance": "Zenodo DOI 10.5281/zenodo.12941231",
        "classes": TARGET_CLASSES,
        "sample_counts": {
            "train": len(train_ds),
            "val_held_out": len(val_ds),
            "total_patches": len(train_ds) + len(val_ds)
        },
        "hardware": f"{device}",
        "total_runtime_seconds": total_training_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_metrics": {
            "val_loss": history[-1]["val_loss"],
            "val_mean_iou": history[-1]["val_mean_iou"],
            "val_mean_dice": history[-1]["val_mean_dice"],
            "val_iou_built_up": history[-1]["val_iou_built_up"],
            "val_iou_water": history[-1]["val_iou_water"],
            "val_iou_vegetation": history[-1]["val_iou_vegetation"]
        },
        "history": history
    }

    report_file = checkpoint_out_dir / "training_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[Checkpoint Saved] -> {ckpt_file}")
    print(f"[Report Saved]     -> {report_file}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpticalSARNet Authentic-Data Pilot")
    parser.add_argument("--data-dir", default=str(ML_DIR / "authentic_data" / "ben_ge_800_prepared"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_and_eval(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
