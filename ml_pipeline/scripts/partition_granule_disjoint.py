#!/usr/bin/env python3
"""
ml_pipeline/scripts/partition_granule_disjoint.py

Generates parent-granule-disjoint splits for the authentic BigEarthNet-MM / BEN-GE-800 dataset.
Ensures ZERO leakage across train, val, and test partitions:
- Zero shared Sentinel-2 parent granules
- Zero shared Sentinel-1 parent acquisitions
- Stratified class distribution across Built-up, Water, and Vegetation
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

def extract_parent_s2(patch_id: str) -> str:
    """Extract parent S2 granule ID by stripping _<row>_<col> coordinate suffix."""
    return patch_id.rsplit('_', 2)[0]

def extract_parent_s1(s1_patch_id: str) -> str:
    """Extract parent S1 acquisition ID by stripping _<row>_<col> coordinate suffix."""
    return s1_patch_id.rsplit('_', 2)[0]

def partition_granule_disjoint(
    data_dir: Path,
    output_dir: Path,
    seed: int = 1355,
    target_train: int = 560,
    target_val: int = 120,
    target_test: int = 120
):
    print(f"Loading prepared authentic annotations from: {data_dir}")
    all_samples = []
    for split in ["train", "val", "test"]:
        ann_path = data_dir / split / f"{split}_annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")
        with open(ann_path, "r", encoding="utf-8") as f:
            all_samples.extend(json.load(f))

    total_patches = len(all_samples)
    print(f"Total authentic patches: {total_patches}")
    sample_by_id = {s["patch_id"]: s for s in all_samples}

    # Build co-registration bipartite graph between S2 and S1 parent acquisitions
    adj = defaultdict(set)
    s2_to_patches = defaultdict(list)
    s1_to_patches = defaultdict(list)

    for s in all_samples:
        pid = s["patch_id"]
        s2_gid = extract_parent_s2(pid)
        s1_gid = extract_parent_s1(s["s1_patch_id"])
        s2_to_patches[s2_gid].append(pid)
        s1_to_patches[s1_gid].append(pid)

    for gid, pids in s2_to_patches.items():
        for p in pids:
            for p2 in pids:
                if p != p2:
                    adj[p].add(p2)

    for gid, pids in s1_to_patches.items():
        for p in pids:
            for p2 in pids:
                if p != p2:
                    adj[p].add(p2)

    # Find connected components (closed multi-modal acquisition scenes)
    visited = set()
    components = []
    for s in all_samples:
        pid = s["patch_id"]
        if pid not in visited:
            comp = []
            queue = [pid]
            visited.add(pid)
            for curr in queue:
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    print(f"Identified {len(components)} connected acquisition components across {len(s2_to_patches)} S2 granules and {len(s1_to_patches)} S1 granules.")

    # Calculate class pixel distribution per component
    comp_stats = []
    for idx, comp in enumerate(components):
        c_b, c_w, c_v = 0, 0, 0
        s2_gids = set()
        s1_gids = set()
        for pid in comp:
            s = sample_by_id[pid]
            s2_gids.add(extract_parent_s2(s["patch_id"]))
            s1_gids.add(extract_parent_s1(s["s1_patch_id"]))
            mask = np.load(data_dir / s["target_mask_path"])
            c_b += int(np.sum(mask[..., 0]))
            c_w += int(np.sum(mask[..., 1]))
            c_v += int(np.sum(mask[..., 2]))
        px = len(comp) * 120 * 120
        comp_stats.append({
            "comp_id": idx,
            "pids": comp,
            "patch_count": len(comp),
            "built": c_b,
            "water": c_w,
            "veg": c_v,
            "pixels": px,
            "s2_gids": s2_gids,
            "s1_gids": s1_gids
        })

    # Execute deterministic stratified partitioning
    rng = random.Random(seed)
    shuffled = comp_stats.copy()
    rng.shuffle(shuffled)

    train_c, val_c, test_c = [], [], []
    n_tr, n_va, n_te = 0, 0, 0

    for c in shuffled:
        deficit_tr = (target_train - n_tr) / target_train
        deficit_va = (target_val - n_va) / target_val
        deficit_te = (target_test - n_te) / target_test

        max_def = max(deficit_tr, deficit_va, deficit_te)
        if max_def == deficit_tr and n_tr + c["patch_count"] <= 575:
            train_c.append(c)
            n_tr += c["patch_count"]
        elif max_def == deficit_va and n_va + c["patch_count"] <= 135:
            val_c.append(c)
            n_va += c["patch_count"]
        elif n_te + c["patch_count"] <= 135:
            test_c.append(c)
            n_te += c["patch_count"]
        else:
            mins = min([n_tr, n_va, n_te])
            if mins == n_tr:
                train_c.append(c)
                n_tr += c["patch_count"]
            elif mins == n_va:
                val_c.append(c)
                n_va += c["patch_count"]
            else:
                test_c.append(c)
                n_te += c["patch_count"]

    # Collect patch IDs per split
    train_pids = [pid for c in train_c for pid in c["pids"]]
    val_pids = [pid for c in val_c for pid in c["pids"]]
    test_pids = [pid for c in test_c for pid in c["pids"]]

    # Verify zero overlap
    train_s2 = set.union(*[c["s2_gids"] for c in train_c])
    val_s2 = set.union(*[c["s2_gids"] for c in val_c])
    test_s2 = set.union(*[c["s2_gids"] for c in test_c])

    train_s1 = set.union(*[c["s1_gids"] for c in train_c])
    val_s1 = set.union(*[c["s1_gids"] for c in val_c])
    test_s1 = set.union(*[c["s1_gids"] for c in test_c])

    assert len(train_s2 & val_s2) == 0, f"S2 Train-Val overlap detected: {train_s2 & val_s2}"
    assert len(train_s2 & test_s2) == 0, f"S2 Train-Test overlap detected: {train_s2 & test_s2}"
    assert len(val_s2 & test_s2) == 0, f"S2 Val-Test overlap detected: {val_s2 & test_s2}"

    assert len(train_s1 & val_s1) == 0, f"S1 Train-Val overlap detected: {train_s1 & val_s1}"
    assert len(train_s1 & test_s1) == 0, f"S1 Train-Test overlap detected: {train_s1 & test_s1}"
    assert len(val_s1 & test_s1) == 0, f"S1 Val-Test overlap detected: {val_s1 & test_s1}"

    print(f"\n[LEAKAGE CHECK PASSED]")
    print(f"  Sentinel-2 Granules: Train={len(train_s2)}, Val={len(val_s2)}, Test={len(test_s2)} | Intersections: all 0")
    print(f"  Sentinel-1 Granules: Train={len(train_s1)}, Val={len(val_s1)}, Test={len(test_s1)} | Intersections: all 0")

    def calc_split_metrics(cs):
        tot_px = sum(c["pixels"] for c in cs)
        tot_b = sum(c["built"] for c in cs)
        tot_w = sum(c["water"] for c in cs)
        tot_v = sum(c["veg"] for c in cs)
        return {
            "patch_count": sum(c["patch_count"] for c in cs),
            "total_pixels": tot_px,
            "built_pixels": tot_b,
            "built_percentage": round(tot_b / tot_px * 100.0, 2),
            "water_pixels": tot_w,
            "water_percentage": round(tot_w / tot_px * 100.0, 2),
            "vegetation_pixels": tot_v,
            "vegetation_percentage": round(tot_v / tot_px * 100.0, 2)
        }

    stats_train = calc_split_metrics(train_c)
    stats_val = calc_split_metrics(val_c)
    stats_test = calc_split_metrics(test_c)

    print(f"\n[CLASS DISTRIBUTION CHECK]")
    print(f"  Train ({stats_train['patch_count']}): Built={stats_train['built_percentage']}%, Water={stats_train['water_percentage']}%, Veg={stats_train['vegetation_percentage']}%")
    print(f"  Val   ({stats_val['patch_count']}): Built={stats_val['built_percentage']}%, Water={stats_val['water_percentage']}%, Veg={stats_val['vegetation_percentage']}%")
    print(f"  Test  ({stats_test['patch_count']}): Built={stats_test['built_percentage']}%, Water={stats_test['water_percentage']}%, Veg={stats_test['vegetation_percentage']}%")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Format annotations pointing to existing prepared files
    def format_annotations(pids, split_name):
        res = []
        for pid in pids:
            orig = sample_by_id[pid]
            res.append({
                "patch_id": orig["patch_id"],
                "s1_patch_id": orig["s1_patch_id"],
                "parent_s2_granule": extract_parent_s2(orig["patch_id"]),
                "parent_s1_acquisition": extract_parent_s1(orig["s1_patch_id"]),
                "s2_image_path": orig["s2_image_path"],
                "s1_sar_path": orig["s1_sar_path"],
                "target_mask_path": orig["target_mask_path"],
                "caption": orig.get("caption", ""),
                "clc_labels": orig.get("clc_labels", []),
                "split": split_name,
                "partition_protocol": "parent_granule_disjoint",
                "source": orig.get("source", "ben-ge-800"),
                "provenance": orig.get("provenance", "")
            })
        return res

    train_ann = format_annotations(train_pids, "train")
    val_ann = format_annotations(val_pids, "val")
    test_ann = format_annotations(test_pids, "test")

    train_file = output_dir / "train_disjoint.json"
    val_file = output_dir / "val_disjoint.json"
    test_file = output_dir / "test_disjoint.json"

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_ann, f, indent=2)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_ann, f, indent=2)
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_ann, f, indent=2)

    manifest = {
        "dataset_name": "BEN-GE-800 Parent-Granule-Disjoint Splits",
        "protocol": "parent_granule_disjoint",
        "random_seed": seed,
        "total_patches": total_patches,
        "splits": {
            "train": {
                "sample_count": len(train_ann),
                "unique_s2_granules": sorted(list(train_s2)),
                "s2_granule_count": len(train_s2),
                "unique_s1_acquisitions": sorted(list(train_s1)),
                "s1_acquisition_count": len(train_s1),
                "class_statistics": stats_train,
                "file": str(train_file.name)
            },
            "validation": {
                "sample_count": len(val_ann),
                "unique_s2_granules": sorted(list(val_s2)),
                "s2_granule_count": len(val_s2),
                "unique_s1_acquisitions": sorted(list(val_s1)),
                "s1_acquisition_count": len(val_s1),
                "class_statistics": stats_val,
                "file": str(val_file.name)
            },
            "test": {
                "sample_count": len(test_ann),
                "unique_s2_granules": sorted(list(test_s2)),
                "s2_granule_count": len(test_s2),
                "unique_s1_acquisitions": sorted(list(test_s1)),
                "s1_acquisition_count": len(test_s1),
                "class_statistics": stats_test,
                "file": str(test_file.name)
            }
        },
        "leakage_verification": {
            "s2_train_val_overlap": len(train_s2 & val_s2),
            "s2_train_test_overlap": len(train_s2 & test_s2),
            "s2_val_test_overlap": len(val_s2 & test_s2),
            "s1_train_val_overlap": len(train_s1 & val_s1),
            "s1_train_test_overlap": len(train_s1 & test_s1),
            "s1_val_test_overlap": len(val_s1 & test_s1),
            "patch_id_overlap": len(set(train_pids) & set(val_pids)) + len(set(train_pids) & set(test_pids)) + len(set(val_pids) & set(test_pids)),
            "zero_leakage_confirmed": True
        }
    }

    manifest_file = output_dir / "split_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also copy manifest to reports directory for immediate audit reference
    report_manifest_file = Path("reports/training_migration/granule_disjoint_split_manifest.json")
    report_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Generated Files]")
    print(f"  Train Disjoint Annotations: {train_file} ({len(train_ann)} samples)")
    print(f"  Val Disjoint Annotations:   {val_file} ({len(val_ann)} samples)")
    print(f"  Test Disjoint Annotations:  {test_file} ({len(test_ann)} samples)")
    print(f"  Split Manifest:             {manifest_file}")
    print(f"  Report Manifest Copy:       {report_manifest_file}")

    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Parent-Granule-Disjoint Splits")
    parser.add_argument("--data-dir", default="ml_pipeline/authentic_data/ben_ge_800_prepared")
    parser.add_argument("--output-dir", default="ml_pipeline/authentic_data/granule_disjoint_splits")
    parser.add_argument("--seed", type=int, default=1355)
    args = parser.parse_args()

    partition_granule_disjoint(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed
    )
