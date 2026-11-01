#!/usr/bin/env python3
"""
ml_pipeline/tune_v4_thresholds.py
Tuning per-class thresholds for SatQueryOpticalSARNet-v4 and saving final checkpoint.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from data.bigearthnet_loader import BigEarthNetDataset
from cross_modal_model import SatQueryOpticalSARNet
from train_cross_modal_v4 import collate_fn_v4, tune_thresholds

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = ML_DIR / "checkpoints" / "v4" / "optical_sar"
    best_ckpt = checkpoint_dir / "optical_sar_net_v4_best.pth"
    
    print(f"[Device] Running on: {device.upper()}")
    print(f"[Loading] Loading {best_ckpt}...")
    try:
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(best_ckpt, map_location=device)
        
    model = SatQueryOpticalSARNet(sar_channels=1, num_classes=3).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    
    val_ds = BigEarthNetDataset(split="val", max_samples=None)
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_fn_v4(b, img_size=256, augment=False)
    )
    
    print(f"[Tuning] Tuning thresholds on {len(val_ds)} validation samples...")
    t0 = time.time()
    best_thresholds = tune_thresholds(model, val_loader, device)
    tuning_time = time.time() - t0
    
    final_ckpt_path = checkpoint_dir / "optical_sar_net_v4.pth"
    torch.save({
        **state,
        "optimal_thresholds": best_thresholds,
        "tuning_time_seconds": round(tuning_time, 2),
    }, final_ckpt_path)
    print(f"[Saved] Final checkpoint: {final_ckpt_path}")
    print(f"  Optimal thresholds: built_up={best_thresholds[0]:.2f}, water={best_thresholds[1]:.2f}, veg={best_thresholds[2]:.2f}")

    # Training report
    report = {
        "model_name": "SatQueryOpticalSARNet-v4",
        "version": "v4",
        "improvements": [
            "Fixed HWC->CHW target mask transposition",
            "Class-weighted BCE+Dice loss",
            "Data augmentation: random flips, 90-deg rotations, SAR speckle noise",
            "OneCycleLR schedule with 10% warmup + cosine annealing",
            "Best-checkpoint saving by validation mIoU",
            "Optimal per-class prediction threshold tuning on validation split",
        ],
        "best_epoch": state.get("best_epoch"),
        "best_val_miou": state.get("best_val_miou"),
        "optimal_thresholds": {
            "built_up": best_thresholds[0],
            "water": best_thresholds[1],
            "vegetation": best_thresholds[2],
        },
        "class_weights": state.get("class_weights"),
        "history": state.get("history", []),
    }
    report_path = checkpoint_dir / "training_report_v4.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Saved] Training report: {report_path}")

if __name__ == "__main__":
    main()

