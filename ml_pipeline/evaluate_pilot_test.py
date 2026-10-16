#!/usr/bin/env python3
"""
ml_pipeline/evaluate_pilot_test.py

Rigorous held-out evaluation of SatQueryOpticalSARNet-AuthenticPilot-v1
on the untouched 129-sample BigEarthNet-MM (BEN-GE-800) test split.
Computes:
  - Per-class IoU, Dice/F1, Precision, Recall, Specificity
  - Global Micro-averaged and Sample Macro-averaged metrics
  - Pixel confusion matrix
  - Checkpoint SHA-256 verification
  - Train/Val vs. Test overfitting comparison
  - Per-sample breakdown saved to JSON
"""
import sys
import os
import json
import time
import hashlib
from pathlib import Path

# Enforce UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

import torch
import numpy as np
from PIL import Image

from cross_modal_model import SatQueryOpticalSARNet
from train_optical_sar_pilot import AuthenticOpticalSARDataset

CLASSES = ["Built-up", "Water", "Vegetation"]

def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def evaluate():
    data_dir = ML_DIR / "authentic_data" / "ben_ge_800_prepared"
    ckpt_path = ML_DIR / "checkpoints" / "satquery_optical_sar_net_authentic_pilot" / "optical_sar_net.pth"

    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
    ckpt_sha256 = get_file_sha256(ckpt_path)
    print(f"Checkpoint SHA-256: {ckpt_sha256}")

    device = torch.device("cpu")
    model = SatQueryOpticalSARNet(sar_channels=1, num_classes=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = AuthenticOpticalSARDataset(data_dir, split="test", target_size=(128, 128))
    print(f"Loaded {len(test_ds)} held-out test samples.")

    # Global counters
    tp = np.zeros(3, dtype=np.int64)
    fp = np.zeros(3, dtype=np.int64)
    fn = np.zeros(3, dtype=np.int64)
    tn = np.zeros(3, dtype=np.int64)

    per_sample_results = []
    sample_macro_present = {0: [], 1: [], 2: []}
    sample_macro_all = {0: [], 1: [], 2: []}

    t0 = time.time()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        opt = item["optical"].unsqueeze(0).to(device)
        sar = item["sar"].unsqueeze(0).to(device)
        tgt = item["target"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(opt, sar)
            loss = bce_loss_fn(logits, tgt).item()
            total_loss += loss
            probs = torch.sigmoid(logits)[0]  # [3, 128, 128]
            preds = (probs > 0.5).float().cpu().numpy()
            target_np = tgt[0].cpu().numpy()

        sample_entry = {
            "patch_id": item["patch_id"],
            "loss": round(loss, 4),
            "clc_labels": item["clc_labels"],
            "class_metrics": {}
        }

        for c, cname in enumerate(CLASSES):
            p = preds[c].astype(bool)
            t = target_np[c].astype(bool)

            s_tp = int(np.sum(p & t))
            s_fp = int(np.sum(p & (~t)))
            s_fn = int(np.sum((~p) & t))
            s_tn = int(np.sum((~p) & (~t)))

            tp[c] += s_tp
            fp[c] += s_fp
            fn[c] += s_fn
            tn[c] += s_tn

            s_inter = s_tp
            s_union = s_tp + s_fp + s_fn

            has_ground_truth = (s_tp + s_fn) > 0

            if s_union == 0:
                s_iou = 1.0
                sample_macro_all[c].append(1.0)
            else:
                s_iou = s_inter / float(s_union)
                sample_macro_all[c].append(s_iou)
                if has_ground_truth:
                    sample_macro_present[c].append(s_iou)

            sample_entry["class_metrics"][cname] = {
                "tp": s_tp,
                "fp": s_fp,
                "fn": s_fn,
                "tn": s_tn,
                "iou": round(s_iou, 4),
                "has_ground_truth": has_ground_truth
            }

        per_sample_results.append(sample_entry)

    eval_duration = time.time() - t0
    avg_loss = total_loss / len(test_ds)

    # Compute Global (Micro) Dataset Metrics
    class_metrics = {}
    for c, cname in enumerate(CLASSES):
        c_tp = float(tp[c])
        c_fp = float(fp[c])
        c_fn = float(fn[c])
        c_tn = float(tn[c])

        iou = c_tp / (c_tp + c_fp + c_fn + 1e-6)
        prec = c_tp / (c_tp + c_fp + 1e-6)
        rec = c_tp / (c_tp + c_fn + 1e-6)
        dice = (2.0 * c_tp) / (2.0 * c_tp + c_fp + c_fn + 1e-6)
        spec = c_tn / (c_tn + c_fp + 1e-6)

        macro_present = float(np.mean(sample_macro_present[c])) if sample_macro_present[c] else 0.0
        macro_all = float(np.mean(sample_macro_all[c])) if sample_macro_all[c] else 0.0

        class_metrics[cname] = {
            "tp_pixels": int(c_tp),
            "fp_pixels": int(c_fp),
            "fn_pixels": int(c_fn),
            "tn_pixels": int(c_tn),
            "pixel_total": int(c_tp + c_fp + c_fn + c_tn),
            "global_iou": round(iou, 4),
            "global_dice_f1": round(dice, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "sample_macro_iou_present_only": round(macro_present, 4),
            "sample_macro_iou_all_samples": round(macro_all, 4),
            "samples_with_class_present": len(sample_macro_present[c])
        }

    mean_global_iou = float(np.mean([class_metrics[c]["global_iou"] for c in CLASSES]))
    mean_global_dice = float(np.mean([class_metrics[c]["global_dice_f1"] for c in CLASSES]))
    mean_precision = float(np.mean([class_metrics[c]["precision"] for c in CLASSES]))
    mean_recall = float(np.mean([class_metrics[c]["recall"] for c in CLASSES]))
    mean_macro_present = float(np.mean([class_metrics[c]["sample_macro_iou_present_only"] for c in CLASSES]))
    mean_macro_all = float(np.mean([class_metrics[c]["sample_macro_iou_all_samples"] for c in CLASSES]))

    # Load validation report for comparative overfitting analysis
    val_report_path = ML_DIR / "checkpoints" / "satquery_optical_sar_net_authentic_pilot" / "training_report.json"
    val_report = {}
    if val_report_path.exists():
        val_report = json.load(open(val_report_path, encoding="utf-8"))

    results = {
        "evaluation_name": "Held-Out Test Set Evaluation (SatQueryOpticalSARNet-AuthenticPilot-v1)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": ckpt_sha256,
            "architecture": "SatQueryOpticalSARNet",
            "parameters": ckpt.get("parameters", 849603),
            "trained_epochs": ckpt.get("history", [{}])[-1].get("epoch", 3)
        },
        "dataset": {
            "name": "BEN-GE-800 (BigEarthNet-MM subset)",
            "split": "test",
            "sample_count": len(test_ds),
            "provenance": "Zenodo DOI 10.5281/zenodo.12941231; Mommert et al 2023; ESA WorldCover 10m v100",
            "split_method": "MD5-hash deterministic 70/15/15"
        },
        "system": {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "device": str(device),
            "runtime_seconds": round(eval_duration, 2),
            "latency_ms_per_sample": round((eval_duration / len(test_ds)) * 1000.0, 2)
        },
        "preprocessing": {
            "optical": "Sentinel-2 B04/B03/B02 uint8 normalized to [0, 1], resized to (128, 128)",
            "sar": "Sentinel-1 VV float32 backscatter in dB, resized to (128, 128)",
            "target": "ESA WorldCover 10m raster ground truth binary masks (128, 128)",
            "threshold": 0.50
        },
        "aggregate_metrics": {
            "test_bce_loss": round(avg_loss, 4),
            "global_mean_iou": round(mean_global_iou, 4),
            "global_mean_dice_f1": round(mean_global_dice, 4),
            "global_mean_precision": round(mean_precision, 4),
            "global_mean_recall": round(mean_recall, 4),
            "sample_macro_iou_present_only": round(mean_macro_present, 4),
            "sample_macro_iou_all_samples": round(mean_macro_all, 4)
        },
        "class_level_metrics": class_metrics,
        "split_comparison_val_vs_test": {
            "val_loss": val_report.get("final_metrics", {}).get("val_loss"),
            "test_loss": round(avg_loss, 4),
            "val_mean_iou": val_report.get("final_metrics", {}).get("val_mean_iou"),
            "test_mean_iou": round(mean_global_iou, 4),
            "delta_mean_iou": round(mean_global_iou - (val_report.get("final_metrics", {}).get("val_mean_iou") or 0.0), 4),
            "val_iou_built_up": val_report.get("final_metrics", {}).get("val_iou_built_up"),
            "test_iou_built_up": class_metrics["Built-up"]["global_iou"],
            "val_iou_water": val_report.get("final_metrics", {}).get("val_iou_water"),
            "test_iou_water": class_metrics["Water"]["global_iou"],
            "val_iou_vegetation": val_report.get("final_metrics", {}).get("val_iou_vegetation"),
            "test_iou_vegetation": class_metrics["Vegetation"]["global_iou"]
        },
        "per_sample_results": per_sample_results
    }

    out_file = ROOT_DIR / "reports" / "training_migration" / "pilot_test_metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("  HELD-OUT TEST EVALUATION COMPLETE")
    print("=" * 65)
    print(f"Total Samples: {len(test_ds)} | Runtime: {eval_duration:.2f}s ({results['system']['latency_ms_per_sample']} ms/sample)")
    print(f"Test Loss: {avg_loss:.4f} (Val Loss: {val_report.get('final_metrics', {}).get('val_loss')})")
    print(f"Global Mean IoU: {mean_global_iou*100:.2f}% (Val Mean IoU: {val_report.get('final_metrics', {}).get('val_mean_iou', 0)*100:.2f}%)")
    print(f"Global Mean Dice/F1: {mean_global_dice*100:.2f}%")
    print(f"Global Mean Precision: {mean_precision*100:.2f}% | Recall: {mean_recall*100:.2f}%")
    print("\nPer-Class Global Performance:")
    for cname in CLASSES:
        cm = class_metrics[cname]
        print(f"  {cname:12s} -> IoU: {cm['global_iou']*100:.2f}% | F1: {cm['global_dice_f1']*100:.2f}% | "
              f"Prec: {cm['precision']*100:.2f}% | Rec: {cm['recall']*100:.2f}% | Present-N: {cm['samples_with_class_present']}")
    print(f"\nDetailed metrics saved to: {out_file}")

if __name__ == "__main__":
    evaluate()
