#!/usr/bin/env python3
"""
scripts/compute_cdvqa_priors_baseline.py

Computes the empirical majority baselines for CDVQA:
1. Per-category majority baseline (trained on CDVQA train split, evaluated on Val and Test 1).
2. Global majority baseline (always predict 'no', evaluated on Val and Test 1).

Generates reports/computed_baselines.json with full citation, exact counts, OA, AA,
and references to reports/cdvqa_answer_priors.json.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT_DIR = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT_DIR / "ml_pipeline" / "authentic_data" / "cdvqa_prepared"
OUT_FILE = ROOT_DIR / "reports" / "computed_baselines.json"
PRIORS_FILE = ROOT_DIR / "reports" / "cdvqa_answer_priors.json"

def main():
    splits = ["train", "val", "test"]
    data = {}
    for s in splits:
        fpath = PREP_DIR / s / "cdvqa_annotations.json"
        with open(fpath, "r", encoding="utf-8") as f:
            data[s] = json.load(f)
        print(f"Loaded {s}: {len(data[s])} records")

    # 1. Determine training set priors
    train_cat_answers = defaultdict(Counter)
    train_overall_answers = Counter()
    for item in data["train"]:
        cat = item.get("question_type", "unknown")
        ans = str(item.get("answer", "")).strip().lower()
        train_cat_answers[cat][ans] += 1
        train_overall_answers[ans] += 1

    train_cat_majority = {cat: counter.most_common(1)[0][0] for cat, counter in train_cat_answers.items()}
    train_global_majority = train_overall_answers.most_common(1)[0][0]

    print("\nTraining Category Majority Answers:")
    for cat, maj in sorted(train_cat_majority.items()):
        print(f"  {cat}: {maj} ({train_cat_answers[cat][maj]}/{sum(train_cat_answers[cat].values())})")
    print(f"Global Majority: {train_global_majority}")

    # Evaluate on val and test (test1)
    results = {}
    for eval_split in ["val", "test"]:
        split_name = "test1" if eval_split == "test" else eval_split
        records = data[eval_split]
        total = len(records)

        # Per-category majority evaluation
        cat_correct = defaultdict(int)
        cat_total = defaultdict(int)

        # Global majority evaluation
        global_correct = 0

        for r in records:
            cat = r.get("question_type", "unknown")
            ans = str(r.get("answer", "")).strip().lower()
            cat_total[cat] += 1

            # Check per-category prediction
            pred_cat = train_cat_majority.get(cat, train_global_majority)
            if pred_cat == ans:
                cat_correct[cat] += 1

            # Check global prediction
            if ans == train_global_majority:
                global_correct += 1

        oa_cat = sum(cat_correct.values()) / total * 100.0
        cat_accs = {c: (cat_correct[c] / cat_total[c] * 100.0) if cat_total[c] > 0 else 0.0 for c in cat_total}
        aa_cat = sum(cat_accs.values()) / len(cat_accs)

        oa_global = global_correct / total * 100.0

        results[split_name] = {
            "per_category_majority": {
                "overall_accuracy": round(oa_cat, 2),
                "average_accuracy": round(aa_cat, 2),
                "category_accuracies": {k: round(v, 2) for k, v in cat_accs.items()}
            },
            "global_majority": {
                "majority_answer": train_global_majority,
                "overall_accuracy": round(oa_global, 2)
            }
        }
        print(f"\n{split_name.upper()} Per-Category Majority Baseline: OA = {oa_cat:.2f}%, AA = {aa_cat:.2f}%")
        print(f"{split_name.upper()} Global Majority Baseline: OA = {oa_global:.2f}%")

    output_payload = {
        "cdvqa_test1_per_category_majority_oa": {
            "dataset": "CDVQA / SECOND",
            "benchmark_split": "Test Set 1 (968 pairs, 39,686 questions)",
            "baseline_type": "Empirical Per-Category Training Prior",
            "metric": "Overall Accuracy (OA)",
            "value": results["test1"]["per_category_majority"]["overall_accuracy"],
            "computation_script": "scripts/compute_cdvqa_priors_baseline.py",
            "priors_source": "reports/cdvqa_answer_priors.json",
            "description": "Always predicting the category-level majority answer observed on the 65,967 training questions (e.g. change_or_not -> yes, decrease_or_not -> no, etc.).",
            "notes": "Relocated from published_baselines.json; this is a computed empirical prior baseline, not a number published in Yuan et al. 2022 Table I."
        },
        "cdvqa_test1_per_category_majority_aa": {
            "dataset": "CDVQA / SECOND",
            "benchmark_split": "Test Set 1 (968 pairs, 39,686 questions)",
            "baseline_type": "Empirical Per-Category Training Prior",
            "metric": "Average Accuracy (AA)",
            "value": results["test1"]["per_category_majority"]["average_accuracy"],
            "computation_script": "scripts/compute_cdvqa_priors_baseline.py",
            "priors_source": "reports/cdvqa_answer_priors.json",
            "description": "Macro-average accuracy across all 8 CDVQA categories under the per-category training prior.",
            "notes": "Relocated from published_baselines.json."
        },
        "cdvqa_val_per_category_majority_oa": {
            "dataset": "CDVQA / SECOND",
            "benchmark_split": "Validation Set (400 pairs, 16,441 questions)",
            "baseline_type": "Empirical Per-Category Training Prior",
            "metric": "Overall Accuracy (OA)",
            "value": results["val"]["per_category_majority"]["overall_accuracy"],
            "computation_script": "scripts/compute_cdvqa_priors_baseline.py",
            "priors_source": "reports/cdvqa_answer_priors.json",
            "description": "Per-category training majority baseline evaluated on the 16,441 validation questions."
        },
        "cdvqa_val_per_category_majority_aa": {
            "dataset": "CDVQA / SECOND",
            "benchmark_split": "Validation Set (400 pairs, 16,441 questions)",
            "baseline_type": "Empirical Per-Category Training Prior",
            "metric": "Average Accuracy (AA)",
            "value": results["val"]["per_category_majority"]["average_accuracy"],
            "computation_script": "scripts/compute_cdvqa_priors_baseline.py",
            "priors_source": "reports/cdvqa_answer_priors.json",
            "description": "Macro-average accuracy across all 8 CDVQA categories on validation set under per-category training prior."
        },
        "details": results
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nWrote computed baselines to: {OUT_FILE}")

if __name__ == "__main__":
    main()
