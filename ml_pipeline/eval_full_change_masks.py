#!/usr/bin/env python3
"""
ml_pipeline/eval_full_change_masks.py

Full-scale held-out evaluation of SatQueryChangeNet across 100% of the authentic
CDVQA / SECOND Test and Test2 splits (968 pairs each).
Produces official benchmark metrics:
- Macro mIoU & Micro IoU
- Macro Dice F1
- Precision, Recall
- Confusion Matrix (TP, FP, FN, TN)
- Per-pair audit records
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# Enforce UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from change_model import SatQueryChangeNet


def derive_label2_path(label1_path: Path) -> Path:
    """Derive T2 label path from T1 label path (label1/ -> label2/)."""
    return Path(str(label1_path).replace(
        str(label1_path.parent),
        str(label1_path.parent).replace("label1", "label2"),
        1
    ))


def compute_binary_change_mask_np(label1_path: Path, label2_path: Path, size=(256, 256)) -> np.ndarray:
    """
    Correct binary change GT for CDVQA/SECOND:
        change[y,x] = 1  iff  label1_rgb[y,x] != label2_rgb[y,x]
    This is the canonical definition in the SECOND dataset.
    The previous implementation (label1.convert('L') > 128) is INCORRECT
    because it measures T1 semantic foreground density (~80-98% positive rate),
    not actual between-epoch land-cover change.
    """
    l1 = np.array(Image.open(label1_path).resize(size, Image.NEAREST))
    l2 = np.array(Image.open(label2_path).resize(size, Image.NEAREST))
    if l1.ndim == 3:
        return np.any(l1 != l2, axis=-1)
    return (l1 != l2)

def load_unique_pairs(split: str = "test", max_pairs: int = None):
    prep_dir = ML_DIR / "authentic_data" / "cdvqa_prepared" / split
    with open(prep_dir / "cdvqa_annotations.json", "r", encoding="utf-8") as f:
        annots = json.load(f)

    seen = set()
    pairs = []
    for a in annots:
        pid = a["pair_id"]
        if pid not in seen:
            seen.add(pid)
            label1_path = ROOT_DIR / a["mask_path"]
            label2_path = derive_label2_path(label1_path)
            pairs.append({
                "pair_id": pid,
                "t1_path": ROOT_DIR / a["t1_path"],
                "t2_path": ROOT_DIR / a["t2_path"],
                "label1_path": label1_path,
                "label2_path": label2_path,
            })

    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs

def evaluate_checkpoint_on_split(
    checkpoint_path: Path,
    split: str = "test",
    max_pairs: int = None,
    threshold: float = 0.5
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(f"SatQuery AI: Full Authentic CDVQA Mask Evaluation [{split.upper()} SPLIT]", flush=True)
    print(f"Model Checkpoint: {checkpoint_path}", flush=True)
    print(f"Hardware Device: {device.upper()}", flush=True)
    print("=" * 70, flush=True)

    model = SatQueryChangeNet(in_channels=3).to(device)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print("[CHECKPOINT] Weights loaded successfully.", flush=True)
    else:
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}", flush=True)
        return None

    model.eval()
    pairs = load_unique_pairs(split, max_pairs)
    total_pairs = len(pairs)
    print(f"[DATASET] Loaded {total_pairs} unique physical pairs from '{split}'.", flush=True)
    print("[GT MASK] Using correct binary change mask: any pixel where label1 class != label2 class.", flush=True)
    print("[GT MASK] (Previous incorrect method — label1.convert('L')>128 — invalidated.)", flush=True)
    print("[GT MASK] Each pair loads label1/ and label2/ semantic RGB maps from SECOND dataset.", flush=True)

    ious = []
    f1s = []
    precisions = []
    recalls = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    per_pair_results = []
    start_time = time.time()

    for idx, p in enumerate(pairs):
        t1_img = Image.open(p["t1_path"]).convert("RGB").resize((256, 256), Image.BILINEAR)
        t2_img = Image.open(p["t2_path"]).convert("RGB").resize((256, 256), Image.BILINEAR)

        t1_arr = np.array(t1_img, dtype=np.float32) / 255.0
        t2_arr = np.array(t2_img, dtype=np.float32) / 255.0
        t1_t = torch.from_numpy(t1_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        t2_t = torch.from_numpy(t2_arr).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(t1_t, t2_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        pred_bin = (probs > threshold).astype(bool)
        # CORRECTED: derive binary change GT from label1/label2 RGB comparison
        gt_bin = compute_binary_change_mask_np(p["label1_path"], p["label2_path"], size=(256, 256))

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        tp = int(intersection)
        fp = int(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
        fn = int(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())
        tn = int(np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        if union == 0:
            iou = 1.0
            f1 = 1.0
            prec = 1.0
            rec = 1.0
        else:
            iou = float(tp / union) if union > 0 else 0.0
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0

        bg_union = tn + fp + fn
        bg_iou = float(tn / bg_union) if bg_union > 0 else 0.0
        two_class_iou = (iou + bg_iou) / 2.0

        ious.append(iou)
        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)

        per_pair_results.append({
            "pair_id": p["pair_id"],
            "change_iou": round(iou, 4),
            "no_change_iou": round(bg_iou, 4),
            "two_class_miou": round(two_class_iou, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == total_pairs:
            cur_miou = float(np.mean(ious))
            print(f"  Processed {idx + 1}/{total_pairs} pairs | Running Change IoU: {cur_miou*100:.2f}% | Elapsed: {time.time() - start_time:.1f}s", flush=True)

    elapsed_time = round(time.time() - start_time, 2)

    # Compute Macro and Micro metrics
    macro_change_miou = float(np.mean(ious))
    macro_dice = float(np.mean(f1s))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))

    macro_bg_iou = float(np.mean([r["no_change_iou"] for r in per_pair_results]))
    macro_two_class_miou = (macro_change_miou + macro_bg_iou) / 2.0

    micro_union = total_tp + total_fp + total_fn
    micro_iou = float(total_tp / micro_union) if micro_union > 0 else 0.0
    micro_dice = float(2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn)) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0
    micro_precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0

    micro_bg_union = total_tn + total_fp + total_fn
    micro_bg_iou = float(total_tn / micro_bg_union) if micro_bg_union > 0 else 0.0
    micro_two_class_miou = (micro_iou + micro_bg_iou) / 2.0

    accuracy = float((total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn))

    summary = {
        "benchmark": "CDVQA / SECOND (arXiv:2112.06343)",
        "task": "Binary Change Delineation (Mask Segmentation)",
        "gt_mask_protocol": "CORRECTED: binary change = any pixel where label1_rgb != label2_rgb",
        "gt_mask_protocol_note": "Previous GT (label1.convert('L') > 128) was INCORRECT and is superseded.",
        "split": split,
        "total_pairs_evaluated": total_pairs,
        "runtime_seconds": elapsed_time,
        "throughput_fps": round(total_pairs / max(0.01, elapsed_time), 2),
        "threshold": threshold,
        "metrics": {
            "macro_change_miou_pct": round(macro_change_miou * 100.0, 2),
            "macro_two_class_miou_pct": round(macro_two_class_miou * 100.0, 2),
            "macro_dice_f1": round(macro_dice, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "micro_change_iou_pct": round(micro_iou * 100.0, 2),
            "micro_two_class_miou_pct": round(micro_two_class_miou * 100.0, 2),
            "micro_dice_f1": round(micro_dice, 4),
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "pixel_accuracy_pct": round(accuracy * 100.0, 2)
        },
        "confusion_matrix": {
            "true_positive_pixels": total_tp,
            "false_positive_pixels": total_fp,
            "false_negative_pixels": total_fn,
            "true_negative_pixels": total_tn
        },
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT_DIR)) if checkpoint_path.is_relative_to(ROOT_DIR) else str(checkpoint_path),
        "per_pair_results": per_pair_results
    }

    print("\n" + "=" * 70, flush=True)
    print(f"  FINAL {split.upper()} EVALUATION RESULTS (100% COVERAGE, {total_pairs} PAIRS)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Macro Change IoU (Foreground): {macro_change_miou * 100.0:.2f}%", flush=True)
    print(f"  Macro 2-Class mIoU (Mean):    {macro_two_class_miou * 100.0:.2f}%", flush=True)
    print(f"  Macro Dice F1 Score:          {macro_dice:.4f}", flush=True)
    print(f"  Macro Precision:              {macro_precision:.4f}", flush=True)
    print(f"  Macro Recall:                 {macro_recall:.4f}", flush=True)
    print(f"  Micro Change IoU:             {micro_iou * 100.0:.2f}%", flush=True)
    print(f"  Micro 2-Class mIoU:           {micro_two_class_miou * 100.0:.2f}%", flush=True)
    print(f"  Micro Dice F1 Score:          {micro_dice:.4f}", flush=True)
    print(f"  Pixel Accuracy:               {accuracy * 100.0:.2f}%", flush=True)
    print(f"  Total Runtime:                {elapsed_time}s ({summary['throughput_fps']} pairs/sec)", flush=True)
    print("=" * 70, flush=True)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT_DIR / ckpt

    res = evaluate_checkpoint_on_split(
        checkpoint_path=ckpt,
        split=args.split,
        max_pairs=args.max_pairs,
        threshold=args.threshold
    )

    if args.out_json and res:
        out_p = Path(args.out_json)
        if not out_p.is_absolute():
            out_p = ROOT_DIR / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[SAVED] Results saved to {out_p}", flush=True)
