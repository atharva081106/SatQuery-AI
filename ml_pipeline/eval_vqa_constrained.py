#!/usr/bin/env python3
"""
ml_pipeline/eval_vqa_constrained.py

Constrained Vocabulary Inference Adapter & Evaluator for SatQuery AI CDVQA.
Official Reference: Yuan et al. (IEEE TGRS 2022 / arXiv:2112.06343)

Eliminates open-ended free-form VLM text generation defects by constraining
predictions strictly to the official 19-token CDVQA vocabulary:
- Binary: ['no', 'yes']
- Semantic Classes (6): ['NVG_surface', 'buildings', 'low_vegetation', 'playgrounds', 'trees', 'water']
- Ratio Intervals (11): ['0', '0_to_10', '10_to_20', '20_to_30', '30_to_40', '40_to_50',
                         '50_to_60', '60_to_70', '70_to_80', '80_to_90', '90_to_100']

Evaluates strictly on the held-out authentic CDVQA Test partition across all 8 categories:
1. change_or_not
2. increase_or_not
3. decrease_or_not
4. change_to_what
5. smallest_change
6. largest_change
7. change_ratio
8. change_ratio_types
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

# Official CDVQA Closed Vocabulary
CATEGORY_CANDIDATES = {
    "change_or_not": [
        ("no", "no"),
        ("yes", "yes"),
    ],
    "increase_or_not": [
        ("no", "no"),
        ("yes", "yes"),
    ],
    "decrease_or_not": [
        ("no", "no"),
        ("yes", "yes"),
    ],
    "change_to_what": [
        ("NVG surface", "NVG_surface"),
        ("buildings", "buildings"),
        ("low vegetation", "low_vegetation"),
        ("playgrounds", "playgrounds"),
        ("trees", "trees"),
        ("water", "water"),
    ],
    "smallest_change": [
        ("NVG surface", "NVG_surface"),
        ("buildings", "buildings"),
        ("low vegetation", "low_vegetation"),
        ("playgrounds", "playgrounds"),
        ("trees", "trees"),
        ("water", "water"),
    ],
    "largest_change": [
        ("NVG surface", "NVG_surface"),
        ("buildings", "buildings"),
        ("low vegetation", "low_vegetation"),
        ("playgrounds", "playgrounds"),
        ("trees", "trees"),
        ("water", "water"),
    ],
    "change_ratio": [
        ("0 percent", "0"),
        ("0 to 10 percent", "0_to_10"),
        ("10 to 20 percent", "10_to_20"),
        ("20 to 30 percent", "20_to_30"),
        ("30 to 40 percent", "30_to_40"),
        ("40 to 50 percent", "40_to_50"),
        ("50 to 60 percent", "50_to_60"),
        ("60 to 70 percent", "60_to_70"),
        ("70 to 80 percent", "70_to_80"),
        ("80 to 90 percent", "80_to_90"),
        ("90 to 100 percent", "90_to_100"),
    ],
    "change_ratio_types": [
        ("0 percent", "0"),
        ("0 to 10 percent", "0_to_10"),
        ("10 to 20 percent", "10_to_20"),
        ("20 to 30 percent", "20_to_30"),
        ("30 to 40 percent", "30_to_40"),
        ("40 to 50 percent", "40_to_50"),
        ("50 to 60 percent", "50_to_60"),
        ("60 to 70 percent", "60_to_70"),
        ("70 to 80 percent", "70_to_80"),
        ("80 to 90 percent", "80_to_90"),
        ("90 to 100 percent", "90_to_100"),
    ]
}

def parse_ratio_to_float(val_str: str) -> float:
    s = val_str.lower().strip()
    if s == "0":
        return 0.0
    m = re.match(r"^(\d+)_to_(\d+)$", s)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return (low + high) / 2.0
    nums = re.findall(r"\d+", s)
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

def run_constrained_vqa_evaluation(
    split: str = "test",
    samples_per_category: int = 25,
    max_total_samples: int = 200,
    output_path: str = None
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75, flush=True)
    print(f"SatQuery AI: Constrained Vocabulary CDVQA Evaluation [{split.upper()} SPLIT]", flush=True)
    print(f"Hardware Device: {device.upper()}", flush=True)
    print("=" * 75, flush=True)

    vlm, processor, vlm_prov = load_vlm(device)
    print(f"[MODEL] Loaded VLM: {vlm_prov}", flush=True)

    # Pre-tokenize all candidate sets
    tokenized_candidates = {}
    for cat, cand_list in CATEGORY_CANDIDATES.items():
        text_phrases = [t[0] for t in cand_list]
        canon_tokens = [t[1] for t in cand_list]
        enc = processor.tokenizer(text_phrases, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        valid_tok_counts = (labels != -100).sum(dim=1).float().to(device)

        tokenized_candidates[cat] = {
            "text_phrases": text_phrases,
            "canon_tokens": canon_tokens,
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "labels": labels.to(device),
            "valid_tok_counts": valid_tok_counts
        }

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

        cand_info = tokenized_candidates[q_type]
        canon_cands = cand_info["canon_tokens"]
        K = len(canon_cands)

        inputs = processor(images=diff_img, text=q, return_tensors="pt").to(device)

        with torch.no_grad():
            vision_out = vlm.vision_model(pixel_values=inputs["pixel_values"])
            img_embeds = vision_out.last_hidden_state
            img_mask = torch.ones(img_embeds.size()[:-1], dtype=torch.long, device=device)

            q_out = vlm.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                encoder_hidden_states=img_embeds,
                encoder_attention_mask=img_mask,
            )
            q_embeds = q_out[0].expand(K, -1, -1)
            q_mask = inputs["attention_mask"].expand(K, -1)

            dec_out = vlm.text_decoder(
                input_ids=cand_info["input_ids"],
                attention_mask=cand_info["attention_mask"],
                encoder_hidden_states=q_embeds,
                encoder_attention_mask=q_mask,
                labels=cand_info["labels"],
                reduction="none",
            )
            # Length-normalized loss
            norm_loss = dec_out.loss / cand_info["valid_tok_counts"]
            best_idx = int(torch.argmin(norm_loss).item())
            pred = canon_cands[best_idx]

        exact = (pred.strip().lower() == gt.strip().lower())

        # Numeric parsing for ratios
        gt_num = parse_ratio_to_float(gt) if "ratio" in q_type else None
        pred_num = parse_ratio_to_float(pred) if "ratio" in q_type else None

        rec = {
            "index": idx,
            "pair_id": item["pair_id"],
            "question_type": q_type,
            "question": q,
            "ground_truth": gt,
            "predicted": pred,
            "exact_match": exact,
            "gt_numeric": gt_num,
            "pred_numeric": pred_num
        }
        results.append(rec)
        cat_results[q_type].append(rec)

        if (idx + 1) % 25 == 0 or (idx + 1) == total_eval:
            cur_acc = np.mean([r["exact_match"] for r in results]) * 100.0
            print(f"  Processed {idx + 1}/{total_eval} | Running Exact Acc: {cur_acc:.2f}% | Elapsed: {time.time() - start_time:.1f}s", flush=True)

    elapsed_time = round(time.time() - start_time, 2)

    # Category summaries
    category_summary = {}
    type_accuracies = []

    for q_type, items in sorted(cat_results.items()):
        n = len(items)
        exact_acc = float(np.mean([r["exact_match"] for r in items])) * 100.0
        type_accuracies.append(exact_acc)

        cat_stat = {
            "evaluated_samples": n,
            "exact_accuracy_pct": round(exact_acc, 2),
        }

        # Binary classification confusion matrix
        if q_type in ["change_or_not", "increase_or_not", "decrease_or_not"]:
            tp = sum(1 for r in items if r["ground_truth"] == "yes" and r["predicted"] == "yes")
            tn = sum(1 for r in items if r["ground_truth"] == "no" and r["predicted"] == "no")
            fp = sum(1 for r in items if r["ground_truth"] == "no" and r["predicted"] == "yes")
            fn = sum(1 for r in items if r["ground_truth"] == "yes" and r["predicted"] == "no")
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec_val / (prec + rec_val) if (prec + rec_val) > 0 else 0.0
            cat_stat["classification_metrics"] = {
                "precision": round(prec, 4),
                "recall": round(rec_val, 4),
                "binary_f1": round(f1, 4),
                "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
            }

        # Numeric ratio metrics
        if "ratio" in q_type:
            diffs = [abs(r["pred_numeric"] - r["gt_numeric"]) for r in items if r["pred_numeric"] is not None and r["gt_numeric"] is not None]
            if diffs:
                mae = float(np.mean(diffs))
                rmse = float(np.sqrt(np.mean(np.array(diffs) ** 2)))
                cat_stat["numerical_metrics"] = {
                    "mean_absolute_error": round(mae, 2),
                    "root_mean_squared_error": round(rmse, 2)
                }

        category_summary[q_type] = cat_stat

    overall_exact = float(np.mean([r["exact_match"] for r in results])) * 100.0
    average_type_acc = float(np.mean(type_accuracies))

    summary = {
        "benchmark": "CDVQA (arXiv:2112.06343 / IEEE TGRS 2022)",
        "task": "Change Visual Question Answering (VQA) — Constrained Vocabulary Inference",
        "split": split,
        "total_evaluated_samples": total_eval,
        "runtime_seconds": elapsed_time,
        "inference_adapter": "Candidate Cross-Entropy Scoring with Sequence-Length Normalization",
        "vocabulary_size": 19,
        "overall_metrics": {
            "overall_exact_accuracy_pct": round(overall_exact, 2),
            "average_per_type_accuracy_pct": round(average_type_acc, 2),
        },
        "per_category_metrics": category_summary,
        "predictions": results
    }

    if output_path is None:
        output_path = REPORTS_DIR / f"cdvqa_vqa_constrained_{split}_results.json"
    else:
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Constrained VQA Evaluation complete in {elapsed_time}s.")
    print(f"Overall Exact Accuracy (OA): {overall_exact:.2f}%")
    print(f"Average Per-Type Accuracy (AA): {average_type_acc:.2f}%")
    print(f"Results written to: {output_path}", flush=True)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constrained VQA Evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "test2"])
    parser.add_argument("--samples_per_category", type=int, default=25)
    parser.add_argument("--max_total_samples", type=int, default=200)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    run_constrained_vqa_evaluation(
        split=args.split,
        samples_per_category=args.samples_per_category,
        max_total_samples=args.max_total_samples,
        output_path=args.output_path
    )
