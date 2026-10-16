#!/usr/bin/env python3
"""
ml_pipeline/eval_comprehensive_vqa.py

Comprehensive Authentic Benchmark Evaluator for SatQuery AI CDVQA.
Evaluates across all 8 official IEEE TGRS 2022 categories:
- change_or_not
- increase_or_not
- decrease_or_not
- smallest_change
- largest_change
- change_to_what
- change_ratio
- change_ratio_types

Produces rigorous metrics:
- Overall Accuracy (OA) & Average Per-Type Accuracy (AA)
- Precision, Recall, F1, Confusion Matrix for classification
- MAE & RMSE for ratio estimation
- Complete audit of all evaluated samples
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import numpy as np
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
REPORTS_DIR = ROOT_DIR / "reports" / "benchmark_evaluation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

def normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def parse_ratio_to_float(val_str: str) -> float:
    """Parses CDVQA ratio string to numeric percentage (0.0 to 100.0)."""
    s = val_str.lower().strip()
    if s == "0":
        return 0.0
    m = re.match(r'^(\d+)_to_(\d+)$', s)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return (low + high) / 2.0
    # Try finding integers in string
    nums = re.findall(r'\d+', s)
    if nums:
        return float(nums[0])
    return 0.0

def load_vlm(device: str):
    from transformers import BlipProcessor, BlipForQuestionAnswering
    vlm_ckpt = ML_DIR / "checkpoints" / "satquery_rs_vlm"
    if vlm_ckpt.exists() and (vlm_ckpt / "model.safetensors").exists():
        processor = BlipProcessor.from_pretrained(str(vlm_ckpt))
        vlm = BlipForQuestionAnswering.from_pretrained(str(vlm_ckpt)).to(device)
        prov = "SatQuery-RS-VLM-v1.0 (Adapted Checkpoint)"
    else:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        vlm = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
        prov = "Salesforce/blip-vqa-base"
    vlm.eval()
    return vlm, processor, prov

def run_vqa_evaluation(
    split: str = "test",
    samples_per_category: int = 25,
    max_total_samples: int = 200
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(f"SatQuery AI: Comprehensive Authentic CDVQA Evaluation [{split.upper()} SPLIT]", flush=True)
    print(f"Hardware Device: {device.upper()}", flush=True)
    print("=" * 70, flush=True)

    vlm, processor, vlm_prov = load_vlm(device)
    print(f"[MODEL] Loaded VLM: {vlm_prov}", flush=True)

    data_path = ML_DIR / "authentic_data" / "cdvqa_prepared" / split / "cdvqa_annotations.json"
    with open(data_path, "r", encoding="utf-8") as f:
        all_annots = json.load(f)

    # Stratify by category to ensure balanced coverage of all 8 types
    by_category = defaultdict(list)
    for a in all_annots:
        by_category[a["question_type"]].append(a)

    selected_samples = []
    for cat, items in sorted(by_category.items()):
        take_n = min(samples_per_category, len(items))
        selected_samples.extend(items[:take_n])

    if max_total_samples and len(selected_samples) > max_total_samples:
        selected_samples = selected_samples[:max_total_samples]

    total_eval = len(selected_samples)
    print(f"[DATASET] Selected {total_eval} balanced samples across {len(by_category)} categories from '{split}'.", flush=True)

    results = []
    cat_results = defaultdict(list)
    start_time = time.time()

    # Cache loaded difference images
    diff_cache = {}
    def get_diff_img(t1_path, t2_path):
        key = (t1_path, t2_path)
        if key not in diff_cache:
            t1 = Image.open(ROOT_DIR / t1_path).convert("RGB")
            t2 = Image.open(ROOT_DIR / t2_path).convert("RGB")
            diff_arr = np.abs(np.array(t2, dtype=np.float32) - np.array(t1, dtype=np.float32)).astype(np.uint8)
            diff_cache[key] = Image.fromarray(diff_arr)
        return diff_cache[key]

    for idx, item in enumerate(selected_samples):
        diff_img = get_diff_img(item["t1_path"], item["t2_path"])
        q = item["question"]
        gt = item["answer"]
        q_type = item["question_type"]

        inputs = processor(diff_img, q, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = vlm.generate(**inputs, max_new_tokens=32)
            pred = processor.decode(gen_ids[0], skip_special_tokens=True).strip()

        norm_pred = normalize_text(pred)
        norm_gt = normalize_text(gt)
        exact = (norm_pred == norm_gt)
        partial = exact or (norm_gt in norm_pred) or (norm_pred in norm_gt and len(norm_pred) > 2)

        # Token F1
        pred_toks = norm_pred.split()
        gt_toks = norm_gt.split()
        common = set(pred_toks) & set(gt_toks)
        p_f1 = 2 * len(common) / (len(pred_toks) + len(gt_toks)) if (pred_toks and gt_toks and common) else 0.0

        rec = {
            "index": idx,
            "pair_id": item["pair_id"],
            "question_type": q_type,
            "question": q,
            "ground_truth": gt,
            "predicted": pred,
            "exact_match": exact,
            "partial_match": partial,
            "token_f1": p_f1,
            "gt_numeric": parse_ratio_to_float(gt) if "ratio" in q_type else None,
            "pred_numeric": parse_ratio_to_float(pred) if "ratio" in q_type else None
        }
        results.append(rec)
        cat_results[q_type].append(rec)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_eval:
            cur_acc = np.mean([r["partial_match"] for r in results]) * 100.0
            print(f"  Processed {idx + 1}/{total_eval} | Running Norm Acc: {cur_acc:.2f}% | Elapsed: {time.time() - start_time:.1f}s", flush=True)

    elapsed_time = round(time.time() - start_time, 2)

    # Compute Detailed Metrics per Category
    category_summary = {}
    type_accuracies = []

    for q_type, items in sorted(cat_results.items()):
        n = len(items)
        exact_acc = float(np.mean([r["exact_match"] for r in items])) * 100.0
        norm_acc = float(np.mean([r["partial_match"] for r in items])) * 100.0
        mean_f1 = float(np.mean([r["token_f1"] for r in items]))
        type_accuracies.append(norm_acc)

        cat_stat = {
            "evaluated_samples": n,
            "exact_accuracy_pct": round(exact_acc, 2),
            "normalized_accuracy_pct": round(norm_acc, 2),
            "token_f1": round(mean_f1, 4)
        }

        # Binary confusion matrix metrics
        if q_type in ["change_or_not", "increase_or_not", "decrease_or_not"]:
            tp = sum(1 for r in items if r["ground_truth"] == "yes" and r["predicted"].lower().strip() == "yes")
            fp = sum(1 for r in items if r["ground_truth"] == "no" and r["predicted"].lower().strip() == "yes")
            fn = sum(1 for r in items if r["ground_truth"] == "yes" and r["predicted"].lower().strip() != "yes")
            tn = sum(1 for r in items if r["ground_truth"] == "no" and r["predicted"].lower().strip() != "yes")

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_bin = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            cat_stat["classification_metrics"] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "binary_f1": round(f1_bin, 4),
                "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
            }

        # Ratio regression metrics
        if "ratio" in q_type:
            gt_nums = np.array([r["gt_numeric"] for r in items])
            pred_nums = np.array([r["pred_numeric"] for r in items])
            errors = np.abs(pred_nums - gt_nums)
            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            cat_stat["numerical_metrics"] = {
                "mean_absolute_error": round(mae, 2),
                "root_mean_squared_error": round(rmse, 2)
            }

        category_summary[q_type] = cat_stat

    overall_exact = float(np.mean([r["exact_match"] for r in results])) * 100.0
    overall_norm = float(np.mean([r["partial_match"] for r in results])) * 100.0
    overall_f1 = float(np.mean([r["token_f1"] for r in results]))
    average_accuracy = float(np.mean(type_accuracies))

    report = {
        "benchmark": "CDVQA (arXiv:2112.06343 / IEEE TGRS 2022)",
        "task": "Change Visual Question Answering (VQA)",
        "split": split,
        "total_evaluated_samples": total_eval,
        "runtime_seconds": elapsed_time,
        "overall_metrics": {
            "overall_exact_accuracy_pct": round(overall_exact, 2),
            "overall_normalized_accuracy_pct": round(overall_norm, 2),
            "average_per_type_accuracy_pct": round(average_accuracy, 2),
            "overall_token_f1": round(overall_f1, 4)
        },
        "per_category_metrics": category_summary,
        "evaluator_model": vlm_prov,
        "evaluation_predictions": results
    }

    print("\n" + "=" * 70, flush=True)
    print("  COMPREHENSIVE CDVQA EVALUATION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Overall Exact Accuracy (OA):     {overall_exact:.2f}%", flush=True)
    print(f"  Overall Normalized Accuracy:     {overall_norm:.2f}%", flush=True)
    print(f"  Average Per-Type Accuracy (AA):  {average_accuracy:.2f}%", flush=True)
    print(f"  Overall Token F1:                {overall_f1:.4f}", flush=True)
    print("=" * 70, flush=True)

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples-per-category", type=int, default=25)
    parser.add_argument("--max-total-samples", type=int, default=200)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    report = run_vqa_evaluation(
        split=args.split,
        samples_per_category=args.samples_per_category,
        max_total_samples=args.max_total_samples
    )

    if args.out_json and report:
        out_p = Path(args.out_json)
        if not out_p.is_absolute():
            out_p = ROOT_DIR / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[SAVED] Results saved to {out_p}", flush=True)
