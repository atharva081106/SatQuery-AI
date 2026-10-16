#!/usr/bin/env python3
"""
ml_pipeline/train_change_net_finetune_v1.py

Controlled Fine-Tuning Experiment v1 for SatQueryChangeNet.
Based on the completed audit of the authentic CDVQA / SECOND change-detection pipeline.

Experiment v1 Specifications:
- Initialized from baseline checkpoint:
  ml_pipeline/checkpoints/satquery_change_net_authentic/best_change_net.pth
  (SHA-256: 9b94320837db808ed284da285568a29b2cca52d2bb51d89d97a479895b11b332)
- Training data: ALL 1,600 authentic CDVQA / SECOND training pairs (100% split coverage).
- Validation data: ALL 400 authentic CDVQA / SECOND validation pairs (100% split coverage).
- Input resolution: 256x256 (identical to baseline).
- Loss: Standard BCEWithLogitsLoss + 0.5 * DiceLoss (identical to baseline, no class weighting).
- Optimizer: AdamW(lr=2e-4, weight_decay=1e-4) with CosineAnnealingLR(eta_min=1e-6).
- Augmentations: Synchronized RandomHorizontalFlip, RandomVerticalFlip, Random90Rotation on (T1, T2, mask).
- Epochs: 4 epochs, checkpoint strictly selected by validation foreground Change IoU.
- Held-out test set: UNTOUCHED during training and model selection.
- Destination: ml_pipeline/checkpoints/satquery_change_net_finetune/v1/
"""

import os
import sys
import json
import time
import hashlib
import argparse
import random
from pathlib import Path

# Enforce UTF-8 encoding on Windows
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

from change_model import SatQueryChangeNet

class DiceLoss(nn.Module):
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

def derive_label2_path(label1_path: Path) -> Path:
    return Path(str(label1_path).replace(
        str(label1_path.parent),
        str(label1_path.parent).replace("label1", "label2"),
        1
    ))

def compute_binary_change_mask_np(label1_rgb_path: Path, label2_rgb_path: Path, img_size=(256, 256)) -> np.ndarray:
    l1 = np.array(Image.open(label1_rgb_path).resize(img_size, Image.NEAREST))
    l2 = np.array(Image.open(label2_rgb_path).resize(img_size, Image.NEAREST))
    if l1.ndim == 3:
        change = np.any(l1 != l2, axis=-1).astype(np.float32)
    else:
        change = (l1 != l2).astype(np.float32)
    return change

class AugmentedCDVQAPairDataset(Dataset):
    """
    Dataset loader for authentic CDVQA / SECOND image pairs.
    Applies synchronized spatial data augmentations (flips, rotations) during training.
    """
    def __init__(self, split: str = "train", img_size=(256, 256), augment: bool = True):
        self.split = split
        self.img_size = img_size
        self.augment = augment

        prep_dir = ML_DIR / "authentic_data" / "cdvqa_prepared" / split
        with open(prep_dir / "cdvqa_annotations.json", "r", encoding="utf-8") as f:
            annots = json.load(f)

        seen_pairs = set()
        self.pairs = []
        for a in annots:
            pid = a["pair_id"]
            if pid not in seen_pairs:
                seen_pairs.add(pid)
                label1_path = ROOT_DIR / a["mask_path"]
                label2_path = derive_label2_path(label1_path)
                self.pairs.append({
                    "pair_id": pid,
                    "t1_path": ROOT_DIR / a["t1_path"],
                    "t2_path": ROOT_DIR / a["t2_path"],
                    "label1_path": label1_path,
                    "label2_path": label2_path,
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        t1_img = Image.open(item["t1_path"]).convert("RGB").resize(self.img_size, Image.BILINEAR)
        t2_img = Image.open(item["t2_path"]).convert("RGB").resize(self.img_size, Image.BILINEAR)

        t1_arr = np.array(t1_img, dtype=np.float32) / 255.0
        t2_arr = np.array(t2_img, dtype=np.float32) / 255.0

        mask_arr = compute_binary_change_mask_np(
            item["label1_path"], item["label2_path"], self.img_size
        )

        # Synchronous Spatial Augmentations (Train Split only)
        if self.augment and self.split == "train":
            # Random Horizontal Flip
            if random.random() > 0.5:
                t1_arr = np.fliplr(t1_arr)
                t2_arr = np.fliplr(t2_arr)
                mask_arr = np.fliplr(mask_arr)

            # Random Vertical Flip
            if random.random() > 0.5:
                t1_arr = np.flipud(t1_arr)
                t2_arr = np.flipud(t2_arr)
                mask_arr = np.flipud(mask_arr)

            # Random 90-degree Rotation
            k = random.randint(0, 3)
            if k > 0:
                t1_arr = np.rot90(t1_arr, k)
                t2_arr = np.rot90(t2_arr, k)
                mask_arr = np.rot90(mask_arr, k)

            # Ensure contiguous arrays after slicing/rotations
            t1_arr = np.ascontiguousarray(t1_arr)
            t2_arr = np.ascontiguousarray(t2_arr)
            mask_arr = np.ascontiguousarray(mask_arr)

        t1_t = torch.from_numpy(t1_arr).permute(2, 0, 1)
        t2_t = torch.from_numpy(t2_arr).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

        return {
            "pair_id": item["pair_id"],
            "t1": t1_t,
            "t2": t2_t,
            "mask": mask_t
        }

def run_finetune_v1(
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 42
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75, flush=True)
    print("SatQuery AI — SatQueryChangeNet Controlled Fine-Tuning Experiment [v1]", flush=True)
    print(f"Device: {device.upper()} | PyTorch: {torch.__version__}", flush=True)
    print("=" * 75, flush=True)

    # Seed for strict reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Output directory
    out_dir = ML_DIR / "checkpoints" / "satquery_change_net_finetune" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_change_net_finetuned.pth"
    report_path = out_dir / "training_report.json"
    config_path = out_dir / "experiment_config.json"

    # Step 1: Pre-training verification of baseline checkpoint
    base_ckpt_path = ML_DIR / "checkpoints" / "satquery_change_net_authentic" / "best_change_net.pth"
    assert base_ckpt_path.exists(), f"Missing baseline checkpoint: {base_ckpt_path}"
    with open(base_ckpt_path, "rb") as f:
        base_sha256 = hashlib.sha256(f.read()).hexdigest()
    expected_sha256 = "9b94320837db808ed284da285568a29b2cca52d2bb51d89d97a479895b11b332"
    print(f"[CHECKPOINT AUDIT] Base Checkpoint: {base_ckpt_path.name}", flush=True)
    print(f"  SHA-256: {base_sha256}", flush=True)
    assert base_sha256 == expected_sha256, f"Checkpoint SHA mismatch: {base_sha256} != {expected_sha256}"
    print("  Verification Status: PASS (Matches authentic reference)", flush=True)

    # Step 2: Load Datasets
    print("\n[DATASET AUDIT] Initializing authentic train and val datasets...", flush=True)
    train_ds = AugmentedCDVQAPairDataset(split="train", img_size=(256, 256), augment=True)
    val_ds = AugmentedCDVQAPairDataset(split="val", img_size=(256, 256), augment=False)

    train_pids = set(p["pair_id"] for p in train_ds.pairs)
    val_pids = set(p["pair_id"] for p in val_ds.pairs)

    print(f"  Train: {len(train_ds)} unique physical pairs (Expected: 1600)", flush=True)
    print(f"  Val:   {len(val_ds)} unique physical pairs (Expected: 400)", flush=True)
    assert len(train_ds) == 1600, f"Expected 1600 train pairs, found {len(train_ds)}"
    assert len(val_ds) == 400, f"Expected 400 val pairs, found {len(val_ds)}"

    overlap = train_pids & val_pids
    print(f"  Physical Scene Overlap: {len(overlap)} pairs", flush=True)
    assert len(overlap) == 0, f"Critical error: split leakage detected! ({len(overlap)} overlapping pairs)"
    print("  Split Separation Status: PASS (Strictly disjoint)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Step 3: Initialize Model & Load Baseline Weights
    model = SatQueryChangeNet(in_channels=3).to(device)
    base_state = torch.load(base_ckpt_path, map_location=device)
    if isinstance(base_state, dict) and "model_state_dict" in base_state:
        model.load_state_dict(base_state["model_state_dict"])
    else:
        model.load_state_dict(base_state)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] Loaded baseline weights into SatQueryChangeNet ({param_count:,} parameters)", flush=True)

    # Save Experiment Config
    config_dict = {
        "experiment_name": "satquery_change_net_finetune_v1",
        "description": "Controlled fine-tuning of SatQueryChangeNet on 100% authentic SECOND/CDVQA train pairs with spatial augmentations",
        "seed": seed,
        "base_checkpoint": str(base_ckpt_path.relative_to(ROOT_DIR)),
        "base_checkpoint_sha256": base_sha256,
        "baseline_val_foreground_iou_pct": 31.15,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "held_out_test_samples": 968,
        "image_resolution": [256, 256],
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR (eta_min=1e-6)",
        "loss_formulation": "BCEWithLogitsLoss() + 0.5 * DiceLoss()",
        "augmentations": [
            "RandomHorizontalFlip(p=0.5)",
            "RandomVerticalFlip(p=0.5)",
            "Random90Rotation(p=0.5)"
        ],
        "selection_metric": "val_mean_foreground_change_iou",
        "device": device
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Step 4: Setup Loss, Optimizer, and Scheduler
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_iou = -1.0
    best_epoch = 0
    history = []
    start_total_time = time.time()

    print("\n[TRAINING] Commencing Fine-Tuning across 4 Epochs...", flush=True)

    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for b_idx, batch in enumerate(train_loader):
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = bce(logits, masks) + 0.5 * dice(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Epoch {ep}/{epochs} | Batch {b_idx + 1}/{len(train_loader)} | Running Loss: {train_loss / train_batches:.4f}", flush=True)

        scheduler.step()
        avg_train_loss = train_loss / max(1, train_batches)

        # Validation Loop (100% of 400 validation pairs)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_ious = []
        val_f1s = []

        with torch.no_grad():
            for batch in val_loader:
                t1 = batch["t1"].to(device)
                t2 = batch["t2"].to(device)
                masks = batch["mask"].to(device)

                logits = model(t1, t2)
                loss = bce(logits, masks) + 0.5 * dice(logits, masks)
                val_loss += loss.item()
                val_batches += 1

                preds = (torch.sigmoid(logits) > 0.5).float()
                inter = (preds * masks).sum(dim=[1, 2, 3])
                union = (preds + masks).clamp(0, 1).sum(dim=[1, 2, 3])
                iou = (inter + 1e-6) / (union + 1e-6)
                dice_score = (2.0 * inter + 1e-6) / (preds.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3]) + 1e-6)

                val_ious.extend(iou.cpu().numpy().tolist())
                val_f1s.extend(dice_score.cpu().numpy().tolist())

        avg_val_loss = val_loss / max(1, val_batches)
        mean_val_iou = float(np.mean(val_ious))
        mean_val_f1 = float(np.mean(val_f1s))
        ep_duration = round(time.time() - ep_start, 2)

        is_best = (mean_val_iou > best_val_iou)
        if is_best:
            best_val_iou = mean_val_iou
            best_epoch = ep
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep,
                "val_foreground_iou": mean_val_iou,
                "val_f1": mean_val_f1,
                "parameters": param_count,
                "architecture": "SatQueryChangeNet (Siamese Feature Diff U-Net)",
                "base_checkpoint_sha256": base_sha256,
                "experiment": "satquery_change_net_finetune_v1"
            }, best_ckpt_path)

        ep_record = {
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_mean_foreground_iou_pct": round(mean_val_iou * 100.0, 2),
            "val_mean_dice_f1": round(mean_val_f1, 4),
            "is_best": is_best,
            "duration_seconds": ep_duration
        }
        history.append(ep_record)

        star = " [*] NEW BEST" if is_best else ""
        print(f"--> Epoch {ep} Complete [{ep_duration}s]: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Change IoU={mean_val_iou * 100.0:.2f}% | Val Dice={mean_val_f1:.4f}{star}", flush=True)

    total_training_time = round(time.time() - start_total_time, 2)
    print(f"\n[DONE] Fine-Tuning v1 completed in {total_training_time}s.", flush=True)
    print(f"Baseline Val Change IoU: 31.15%", flush=True)
    print(f"Best Fine-Tuned Val Change IoU: {best_val_iou * 100.0:.2f}% (Epoch {best_epoch})", flush=True)

    # Checksum of best fine-tuned checkpoint
    with open(best_ckpt_path, "rb") as f:
        finetuned_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"Fine-Tuned Checkpoint SHA-256: {finetuned_sha256}", flush=True)

    # Save comprehensive training report
    report_dict = {
        "experiment": "satquery_change_net_finetune_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_checkpoint": str(base_ckpt_path.relative_to(ROOT_DIR)),
        "base_checkpoint_sha256": base_sha256,
        "finetuned_checkpoint": str(best_ckpt_path.relative_to(ROOT_DIR)),
        "finetuned_checkpoint_sha256": finetuned_sha256,
        "dataset": "SECOND / CDVQA (Yang et al. 2021)",
        "train_pairs_count": len(train_ds),
        "val_pairs_count": len(val_ds),
        "test_pairs_count": 968,
        "test_set_touched": False,
        "baseline_val_change_iou_pct": 31.15,
        "best_epoch": best_epoch,
        "best_val_change_iou_pct": round(best_val_iou * 100.0, 2),
        "relative_val_improvement_pct": round(((best_val_iou * 100.0 - 31.15) / 31.15) * 100.0, 2),
        "total_duration_seconds": total_training_time,
        "hardware_device": device,
        "history": history
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Report saved to: {report_path}", flush=True)
    return report_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQueryChangeNet Controlled Fine-Tuning v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_finetune_v1(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed
    )
