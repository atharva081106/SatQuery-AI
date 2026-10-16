import sys
import os
import inspect
from pathlib import Path
import pytest
import numpy as np

# Ensure backend and ml_pipeline are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR / "ml_pipeline"))
sys.path.insert(0, str(ROOT_DIR))

from ml_pipeline.eval_single_image import (
    compute_box_iou,
    predict_grounding_box_heuristic,
    evaluate_visual_grounding,
)
from tool_registry import tool_registry


def test_missed_detections_receive_zero_iou():
    """Verify that missed detections (None, empty, or zero boxes) score strictly 0.0 IoU."""
    gt_box = [20, 30, 80, 80]

    # Missed detection represented as None
    assert compute_box_iou(None, gt_box) == 0.0
    assert compute_box_iou(gt_box, None) == 0.0

    # Missed detection represented as empty / invalid list
    assert compute_box_iou([], gt_box) == 0.0
    assert compute_box_iou([0, 0], gt_box) == 0.0
    assert compute_box_iou([0, 0, 0, 0], gt_box) == 0.0

    # Inverted / degenerate boxes
    assert compute_box_iou([80, 80, 20, 30], gt_box) == 0.0


def test_prediction_generation_has_no_access_to_ground_truth():
    """Verify that prediction function signature and runtime operate blindly without ground truth."""
    sig = inspect.signature(predict_grounding_box_heuristic)
    forbidden_terms = ["gt", "ground_truth", "bbox", "target_box", "annotation"]
    for param_name in sig.parameters.keys():
        for forbidden in forbidden_terms:
            assert forbidden not in param_name.lower(), (
                f"Candidate prediction function has forbidden parameter '{param_name}' leaking ground truth!"
            )

    # Runtime test on synthetic image
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    pred = predict_grounding_box_heuristic(dummy_img, "airplane")
    # For a blank image, predictor should return None (missed detection), not crash or fabricate
    assert pred is None


def test_iou_standard_coordinate_conventions():
    """Verify standard mathematical IoU behavior for [xmin, ymin, xmax, ymax]."""
    # Exact overlap
    assert compute_box_iou([10, 10, 50, 50], [10, 10, 50, 50]) == 1.0

    # Completely disjoint
    assert compute_box_iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0

    # Partial overlap: boxA (area 100), boxB (area 100), intersection 5x5=25, union=175
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 15, 15]
    expected_iou = 25.0 / 175.0
    assert abs(compute_box_iou(boxA, boxB) - expected_iou) < 1e-5


def test_evaluate_visual_grounding_reproducibility():
    """Verify that evaluate_visual_grounding reproduces strict no-fallback metrics."""
    res = evaluate_visual_grounding()
    assert res["total_samples"] == 15
    assert res["acc_at_0_5_pct"] == 0.0
    assert res["hits_at_0_5"] == 0
    assert res["missed_detections"] == 14
    assert res["mean_iou"] == 0.0113
    assert res["historical_diagnostic_miou"] == 0.0540
    assert res["evaluation_scope"] == "SYNTHETIC_SCHEMA_PROTOTYPE"
    assert res["evaluation_method"] == "HEURISTIC_CONTOUR_BASELINE"


def test_tool_registry_grounding_specification():
    """Verify tool_registry records single_image_grounding as heuristic sector delineation."""
    tool = tool_registry.get_tool("single_image_grounding")
    assert tool is not None
    assert tool.is_heuristic_fallback is True
    assert "heuristic" in tool.description.lower()
    assert "sector delineation" in tool.description.lower()
    assert "Heuristic" in tool.provenance_label
