#!/usr/bin/env python3
"""
ml_pipeline/scripts/prepare_cdvqa.py

Converts CDVQA official annotation JSONs + LEVIR-CD image pairs into
the format expected by ml_pipeline/data/cdvqa_loader.py (CDVQADataset).

CDVQA official JSON structure:
  Train_images.json -> {"images": [{"id": int, "file_name": "train_00001.png"}, ...]}
  Train_questions.json -> {"questions": [{"image_id": int, "id": int, "question": str, "type": str}, ...]}
  Train_answers.json -> {"answers": [{"question_id": int, "answer": str}, ...]}

Image pairs: The CDVQA dataset provides bi-temporal pairs from the LEVIR-CD dataset.
The images are NOT in the GitHub repo - they must come from the LEVIR-CD dataset or
equivalent source. This script verifies that images exist; if images are absent, it
flags ALL entries as BLOCKED rather than substituting synthetic data.

Output: ml_pipeline/authentic_data/cdvqa/<split>/cdvqa_annotations.json
  with keys: pair_id, t1_path, t2_path, mask_path, question, answer, question_type,
             has_change, split, source, provenance
"""
import sys, json, argparse, logging, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

QUESTION_TYPES = [
    "change_or_not",
    "change_ratio_types",
    "increase_or_not",
    "decrease_or_not",
    "change_to_what",
    "smallest_change",
    "largest_change",
    "change_ratio"
]

def load_cdvqa_split(ann_dir, split_prefix):
    """Load images, questions, answers for a given split prefix (Train/Val/Test/Test2)."""
    images_f = ann_dir / f"{split_prefix}_images.json"
    questions_f = ann_dir / f"{split_prefix}_questions.json"
    answers_f = ann_dir / f"{split_prefix}_answers.json"
    for f in [images_f, questions_f, answers_f]:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
    with open(images_f, encoding="utf-8") as f:
        images = {img["id"]: img["file_name"] for img in json.load(f)["images"]}
    with open(questions_f, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(answers_f, encoding="utf-8") as f:
        answers = {a["question_id"]: a["answer"] for a in json.load(f)["answers"]}
    return images, questions, answers

def infer_question_type(question_text):
    q = question_text.lower()
    if any(w in q for w in ["what type", "kind of change", "what happened", "what changed"]):
        return "change_to_what"
    if any(w in q for w in ["did", "is there", "are there", "has", "were", "have"]):
        return "change_or_not"
    if any(w in q for w in ["how much", "percentage", "ratio"]):
        return "change_ratio"
    if any(w in q for w in ["smallest"]):
        return "smallest_change"
    if any(w in q for w in ["largest"]):
        return "largest_change"
    return "change_or_not"

def prepare(ann_dir, images_dir, dst_root, max_per_split=None):
    """
    ann_dir: directory containing CDVQA official JSONs
    images_dir: directory with im1/, im2/, label1/, label2/ (from SECOND) or T1/, T2/
                If None or images absent, entries flagged as image_present=False
    dst_root: output directory (loader-ready cdvqa format)
    """
    split_map = {
        "train": "Train",
        "val": "Val",
        "test": "Test",
        "test2": "Test2"
    }

    total_stats = {"ok": 0, "image_missing": 0, "skipped": 0}

    for split, prefix in split_map.items():
        try:
            images, questions, answers = load_cdvqa_split(ann_dir, prefix)
        except FileNotFoundError as e:
            log.warning(f"Skipping {split}: {e}")
            continue

        log.info(f"[{split}] {len(images)} images, {len(questions)} questions")
        if max_per_split:
            questions = questions[:max_per_split]

        split_dir = dst_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        annotations = []
        for q in questions:
            img_id = q.get("img_id", q.get("image_id"))
            qid = q["id"]
            fname = images.get(img_id)
            if fname is None:
                total_stats["skipped"] += 1
                continue

            # Build canonical pair_id from filename (without extension)
            pair_id = Path(fname).stem

            # Check if actual image files exist
            t1_path = None
            t2_path = None
            mask_path = None
            image_present = False

            if images_dir:
                # Support SECOND dataset layout (im1, im2, label1, label2) as well as legacy LEVIR-CD
                for t1_dir_name in ["im1", "T1", "A", "t1", "before"]:
                    t1_cand = images_dir / t1_dir_name / fname
                    if t1_cand.exists():
                        t1_path = str(t1_cand)
                        break
                for t2_dir_name in ["im2", "T2", "B", "t2", "after"]:
                    t2_cand = images_dir / t2_dir_name / fname
                    if t2_cand.exists():
                        t2_path = str(t2_cand)
                        break
                for mask_dir_name in ["label1", "label", "Label", "masks", "GT", "gt"]:
                    mask_cand = images_dir / mask_dir_name / fname
                    if mask_cand.exists():
                        mask_path = str(mask_cand)
                        break
                image_present = (t1_path is not None and t2_path is not None)
            
            if not image_present:
                total_stats["image_missing"] += 1

            q_type = q.get("type", infer_question_type(q["question"]))
            answer = answers.get(qid, "UNKNOWN")
            has_change = (str(answer).lower() in ["yes", "true", "1"] or
                          (q_type == "change_or_not" and str(answer).lower() != "no") or
                          (q_type == "change_ratio" and str(answer) not in ["0", "unchanged"]) or
                          (q_type == "change_to_what" and str(answer).lower() != "no significant change"))

            entry = {
                "pair_id": pair_id,
                "question_id": qid,
                "image_id": img_id,
                "t1_path": t1_path or "IMAGE_NOT_FOUND",
                "t2_path": t2_path or "IMAGE_NOT_FOUND",
                "mask_path": mask_path or "MASK_NOT_FOUND",
                "question": q["question"],
                "answer": answer,
                "question_type": q_type,
                "has_change": has_change,
                "split": split,
                "image_present": image_present,
                "source": "CDVQA",
                "provenance": "Yuan et al 2022 IEEE TGRS / github.com/YZHJessica/CDVQA (Imagery: CAPTAIN WHU SECOND dataset)"
            }
            annotations.append(entry)
            if image_present:
                total_stats["ok"] += 1

        ann_path = split_dir / "cdvqa_annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)
        log.info(f"  {split}: {len(annotations)} entries -> {ann_path}")
        img_present_count = sum(1 for e in annotations if e["image_present"])
        img_missing_count = sum(1 for e in annotations if not e["image_present"])
        log.info(f"    images present: {img_present_count}, MISSING: {img_missing_count}")
        if img_missing_count > 0:
            log.warning(f"    BLOCKED: {img_missing_count} entries have no images - "
                        f"requires authentic imagery from SECOND dataset (captain-whu.github.io/SCD/)")

    meta = {
        "dataset_name": "CDVQA",
        "source": "Yuan, Mou, Xiong, Zhu 2022 IEEE TGRS",
        "github": "https://github.com/YZHJessica/CDVQA",
        "license": "Apache License 2.0 (CDVQA) / Academic Research Use (SECOND)",
        "image_source": "SECOND (Yang et al 2021 IEEE TGRS / CAPTAIN WHU) bi-temporal aerial image pairs (0.1524m GSD, 512x512)",
        "question_types": QUESTION_TYPES,
        "splits": list(split_map.keys()),
        "stats": total_stats,
        "date_prepared": datetime.datetime.utcnow().isoformat(),
        "note": "CDVQA questions generated on SECOND semantic change detection benchmark. "
                "Entries with image_present=False require authentic rasters from second_dataset.zip."
    }
    with open(dst_root / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"\nDone: ok={total_stats['ok']} image_missing={total_stats['image_missing']} "
             f"skipped={total_stats['skipped']}")
    return total_stats

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare CDVQA annotations for CDVQADataset loader")
    p.add_argument("--ann-dir", required=True, help="Dir with CDVQA official JSON files")
    p.add_argument("--images-dir", default=None, help="Dir with T1/T2/label image subdirs")
    p.add_argument("--dst", required=True, help="Output dir for loader-ready data")
    p.add_argument("--max-per-split", type=int, default=None)
    a = p.parse_args()
    prepare(Path(a.ann_dir), Path(a.images_dir) if a.images_dir else None, Path(a.dst), a.max_per_split)
