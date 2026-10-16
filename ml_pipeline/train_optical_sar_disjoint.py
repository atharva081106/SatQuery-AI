#!/usr/bin/env python3
"""
ml_pipeline/train_optical_sar_disjoint.py

Trains SatQueryOpticalSARNet on the Parent-Granule-Disjoint partition
of Authentic BigEarthNet-MM (BEN-GE-800).
Zero scene or acquisition leakage between training, validation, and test.

Saves isolated checkpoint to:
  ml_pipeline/checkpoints/satquery_optical_sar_net_disjoint/optical_sar_net.pth
"""
import os
import sys
import json
import time
import hashlib
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from cross_modal_model import SatQueryOpticalSARNet

TARGET_CLASSES = ["Built-up", "Water", "Vegetation"]

def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

class DisjointOpticalSARDataset(Dataset):
    """
    Loads authentic Sentinel-1/Sentinel-2 patch pairs and authentic ESA WorldCover ground truth masks
    from parent-granule-disjoint annotation manifests.
    """
    def __init__(self, annotation_path: Path, data_root: Path, target_size: tuple = (128, 128)):
        self.data_root = data_root
        self.target_size = target_size
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotations missing: {annotation_path}")
        with open(annotation_path, "r", encoding="utf-8") as f:
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
        vv = s1_arr[..., 0]  # VV channel
        vv_img = Image.fromarray(vv).resize(self.target_size, Image.Resampling.BILINEAR)
        vv_arr = np.array(vv_img, dtype=np.float32)
        sar_tensor = torch.from_numpy(vv_arr).unsqueeze(0)  # [1, H, W]

        # 3. Load authentic ESA WorldCover target mask
        target_path = self.data_root / item["target_mask_path"]
        tgt_arr = np.load(target_path)  # [120, 120, 3]
        resized_channels = []
        for c in range(3):
            ch_img = Image.fromarray(tgt_arr[..., c]).resize(self.target_size, Image.Resampling.NEAREST)
            resized_channels.append(np.array(ch_img, dtype=np.float32))
        tgt_arr = np.stack(resized_channels, axis=0)  # [3, H, W]
        tgt_tensor = torch.from_numpy(tgt_arr)

        return {
            "patch_id": item["patch_id"],
            "parent_s2_granule": item.get("parent_s2_granule", ""),
            "parent_s1_acquisition": item.get("parent_s1_acquisition", ""),
            "optical": opt_tensor,
            "sar": sar_tensor,
            "target": tgt_tensor,
            "clc_labels": item.get("clc_labels", [])
        }

class MultiLabelDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = (probs + targets).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    num_classes = logits.shape[1]
    class_ious = []
    class_dices = []

    for c in range(num_classes):
        p_c = preds[:, c, :, :]
        t_c = targets[:, c, :, :]

        intersection = (p_c * t_c).sum().item()
        union = ((p_c + t_c) > 0).float().sum().item()

        if union == 0:
            iou = 1.0 if (p_c == t_c).all() else 0.0
        else:
            iou = intersection / union

        cardinality = p_c.sum().item() + t_c.sum().item()
        dice = (2.0 * intersection) / (cardinality + 1e-6) if cardinality > 0 else 1.0

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
        "parent_s2_granule": [item.get("parent_s2_granule", "") for item in batch],
        "parent_s1_acquisition": [item.get("parent_s1_acquisition", "") for item in batch],
        "optical": torch.stack([item["optical"] for item in batch]),
        "sar": torch.stack([item["sar"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "clc_labels": [item["clc_labels"] for item in batch]
    }

def train_disjoint_experiment(
    splits_dir: Path,
    data_root: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 42
):
    print("=" * 80)
    print("SATQUERY AI — PARENT-GRANULE-DISJOINT EXPERIMENTAL TRAINING")
    print("=" * 80)

    # Set seeds for strict reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cpu")
    print(f"Compute hardware: {device}")
    print(f"Random seed:      {seed}")
    print(f"Batch size:       {batch_size}")
    print(f"Epochs:           {epochs}")
    print(f"Learning rate:    {lr}")

    train_ann_file = splits_dir / "train_disjoint.json"
    val_ann_file = splits_dir / "val_disjoint.json"
    test_ann_file = splits_dir / "test_disjoint.json"

    train_ds = DisjointOpticalSARDataset(train_ann_file, data_root=data_root)
    val_ds = DisjointOpticalSARDataset(val_ann_file, data_root=data_root)
    test_ds = DisjointOpticalSARDataset(test_ann_file, data_root=data_root)

    print(f"Partition sample counts:")
    print(f"  Train:      {len(train_ds)} patches (parent-granule disjoint)")
    print(f"  Validation: {len(val_ds)} patches (parent-granule disjoint)")
    print(f"  Test:       {len(test_ds)} patches (held-out disjoint evaluation set)")

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    model = SatQueryOpticalSARNet(sar_channels=1, num_classes=3).to(device)
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized SatQueryOpticalSARNet parameters: {p_count:,}")

    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = MultiLabelDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    start_train_time = time.time()

    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        batches = 0

        for batch in train_loader:
            opt = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            tgt = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(opt, sar)
            loss_bce = bce_loss_fn(logits, tgt)
            loss_dice = dice_loss_fn(logits, tgt)
            loss = loss_bce + 0.5 * loss_dice

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, batches)
        ep_duration = time.time() - ep_start

        # Validation evaluation
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

    # Save isolated experimental checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = output_dir / "optical_sar_net.pth"

    torch.save({
        "model_name": "SatQueryOpticalSARNet-GranuleDisjoint-v1",
        "model_state_dict": model.state_dict(),
        "parameters": p_count,
        "sar_channels": 1,
        "num_classes": 3,
        "classes": TARGET_CLASSES,
        "dataset": "BigEarthNet-MM / BEN-GE-800",
        "protocol": "parent_granule_disjoint",
        "provenance": "Zenodo DOI 10.5281/zenodo.12941231; Mommert et al 2023; ESA WorldCover 10m v100",
        "training_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "test_samples": len(test_ds),
        "random_seed": seed,
        "final_val_loss": history[-1]["val_loss"],
        "final_val_iou": history[-1]["val_mean_iou"],
        "history": history,
        "hardware": f"{device}",
        "total_training_time_seconds": total_training_time
    }, ckpt_file)

    ckpt_sha256 = get_file_sha256(ckpt_file)

    report = {
        "experiment_name": "SatQueryOpticalSARNet Parent-Granule-Disjoint Evaluation",
        "protocol": "parent_granule_disjoint",
        "checkpoint_path": str(ckpt_file),
        "checkpoint_sha256": ckpt_sha256,
        "dataset_name": "BEN-GE-800 (BigEarthNet-MM subset)",
        "ground_truth_source": "ESA WorldCover 10m Land Cover v100 (Authentic Raster GT)",
        "provenance": "Zenodo DOI 10.5281/zenodo.12941231",
        "classes": TARGET_CLASSES,
        "random_seed": seed,
        "sample_counts": {
            "train": len(train_ds),
            "val_held_out": len(val_ds),
            "test_held_out": len(test_ds),
            "total_patches": len(train_ds) + len(val_ds) + len(test_ds)
        },
        "hardware": f"{device}",
        "total_runtime_seconds": total_training_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_validation_metrics": {
            "val_loss": history[-1]["val_loss"],
            "val_mean_iou": history[-1]["val_mean_iou"],
            "val_mean_dice": history[-1]["val_mean_dice"],
            "val_iou_built_up": history[-1]["val_iou_built_up"],
            "val_iou_water": history[-1]["val_iou_water"],
            "val_iou_vegetation": history[-1]["val_iou_vegetation"]
        },
        "history": history
    }

    report_file = output_dir / "training_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Checkpoint Saved] -> {ckpt_file}")
    print(f"  Checkpoint SHA-256: {ckpt_sha256}")
    print(f"[Report Saved]     -> {report_file}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpticalSARNet Parent-Granule-Disjoint")
    parser.add_argument("--splits-dir", default="ml_pipeline/authentic_data/granule_disjoint_splits")
    parser.add_argument("--data-root", default="ml_pipeline/authentic_data/ben_ge_800_prepared")
    parser.add_argument("--output-dir", default="ml_pipeline/checkpoints/satquery_optical_sar_net_disjoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_disjoint_experiment(
        splits_dir=Path(args.splits_dir),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
