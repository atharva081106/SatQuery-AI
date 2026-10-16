import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
import cv2
from PIL import Image
import io

from model_interfaces import (
    check_spatial_compatibility,
    BiTemporalChangeAnalysis,
    decode_satellite_image
)

DEMO_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "demo"

@pytest.fixture
def uttarakhand_pair():
    p1 = DEMO_DIR / "uttarakhand_pre_event_t1.png"
    p2 = DEMO_DIR / "uttarakhand_post_event_t2.png"
    assert p1.exists() and p2.exists(), "Demo files missing"
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        return f1.read(), f2.read()

@pytest.fixture
def disparate_pair(uttarakhand_pair):
    b1, _ = uttarakhand_pair
    p_other = DEMO_DIR / "bengaluru_cartosat3.png"
    assert p_other.exists()
    with open(p_other, "rb") as f:
        return b1, f.read()

def test_identical_images_rejected(uttarakhand_pair):
    b1, _ = uttarakhand_pair
    is_compatible, score, msg, diag = check_spatial_compatibility(b1, b1)
    assert not is_compatible
    assert diag == "IDENTICAL"
    assert "Identical images provided" in msg

    pipeline = BiTemporalChangeAnalysis()
    res = pipeline.execute([b1, b1], "What changed?")
    assert res["compatibility_status"] == "REJECTED_IDENTICAL_IMAGES"
    assert res["confidence"] == round(1.0 / 25.0, 4)
    assert "cannot detect change" in res["text"].lower()

def test_disparate_images_rejected(disparate_pair):
    b1, b2 = disparate_pair
    is_compatible, score, msg, diag = check_spatial_compatibility(b1, b2)
    assert not is_compatible
    assert score <= 0.35
    assert "Disparity" in msg or "Incompatible" in msg

    pipeline = BiTemporalChangeAnalysis()
    res = pipeline.execute([b1, b2], "What changed?")
    assert res["compatibility_status"] == "FAILED"
    assert res["confidence"] <= 0.35

def test_uttarakhand_demo_pair_passes(uttarakhand_pair):
    b1, b2 = uttarakhand_pair
    is_compatible, score, msg, diag = check_spatial_compatibility(b1, b2)
    assert is_compatible
    assert score >= 0.50
    assert "Spatial Coherence Verified" in msg

    pipeline = BiTemporalChangeAnalysis()
    res = pipeline.execute([b1, b2], "Has built-up area increased, decreased, or remained unchanged?")
    assert res["compatibility_status"] == "PASSED"
    assert res["confidence"] > 0.35
    assert "geojson_data" in res and res["geojson_data"] is not None
    assert "features" in res["geojson_data"]
    assert "Surface Transition Matrix" in res["text"]
    assert "INCREASED" in res["text"] or "DECREASED" in res["text"] or "REMAINED UNCHANGED" in res["text"]

def test_decode_palette_image_preserves_colors(uttarakhand_pair):
    b1, _ = uttarakhand_pair
    cv_img, pil_img, geo_meta = decode_satellite_image(b1)
    assert cv_img is not None and pil_img is not None
    # Palette indexed images must not be reduced to uniform grayscale across all channels
    std_channels = np.std(np.mean(cv_img, axis=(0, 1)))
    assert std_channels > 2.0, "Channels are identical; image was decoded as raw indices instead of RGB!"
