"""
SatQuery AI — Spatial Reasoning Layer
Converts pixel/normalized coordinates into human-readable spatial descriptions.
Supports: position descriptions, region partitioning, proximity analysis.

Important: this module clearly distinguishes image-up ("upper portion")
from geographic north ("north") — geographic north is only asserted when
geo_meta is present.
"""
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from data_models import Detection, SpatialRelationship

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Position labels using normalized [0,1] coordinates
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_position(cx_norm: float, cy_norm: float, has_geo: bool = False) -> str:
    """
    Convert normalized center point [0..1] to human-readable position string.
    cy_norm=0 → top of image (which is geographically north IF has_geo=True).

    Returns phrases like "upper-left", "central", "lower-right"
    or "northern" / "southern" when has_geo=True.
    """
    # Horizontal zone
    if cx_norm < 0.33:
        h = "left" if not has_geo else "western"
    elif cx_norm > 0.67:
        h = "right" if not has_geo else "eastern"
    else:
        h = "central"

    # Vertical zone (y=0 is top)
    if cy_norm < 0.33:
        v = "upper" if not has_geo else "northern"
    elif cy_norm > 0.67:
        v = "lower" if not has_geo else "southern"
    else:
        v = "central"

    if v == "central" and h == "central":
        return "central"
    if v == "central":
        return h
    if h == "central":
        return v
    return f"{v}-{h}"


def bbox_to_position(bbox_norm: Dict[str, float], has_geo: bool = False) -> str:
    """Convert a bounding box to a position string using its center point."""
    cx = (bbox_norm["x_min"] + bbox_norm["x_max"]) / 2.0
    cy = (bbox_norm["y_min"] + bbox_norm["y_max"]) / 2.0
    return pixel_to_position(cx, cy, has_geo)


def partition_image_regions(
    img_height: int, img_width: int, has_geo: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Returns normalized bounding boxes for 9 image regions.
    Labeled as north/south/... if has_geo else upper/lower/...
    """
    def label(v_label: str, h_label: str) -> str:
        if v_label == "mid" and h_label == "center":
            return "central"
        if v_label == "mid":
            return h_label
        if h_label == "center":
            return v_label
        return f"{v_label}-{h_label}"

    v_labels = ["upper" if not has_geo else "northern",
                "mid" if not has_geo else "central",
                "lower" if not has_geo else "southern"]
    h_labels = ["left" if not has_geo else "western",
                "center", "right" if not has_geo else "eastern"]

    regions = {}
    for vi, vl in enumerate(v_labels):
        y_start = vi / 3.0
        y_end   = (vi + 1) / 3.0
        for hi, hl in enumerate(h_labels):
            x_start = hi / 3.0
            x_end   = (hi + 1) / 3.0
            name = label(vl, hl)
            regions[name] = {
                "x_min": x_start, "y_min": y_start,
                "x_max": x_end,   "y_max": y_end
            }
    return regions


def get_region_for_bbox(bbox_norm: Dict[str, float],
                         regions: Dict[str, Dict[str, Any]]) -> str:
    """Return which named region the center of a bbox falls in."""
    cx = (bbox_norm["x_min"] + bbox_norm["x_max"]) / 2.0
    cy = (bbox_norm["y_min"] + bbox_norm["y_max"]) / 2.0
    best = "central"
    for name, r in regions.items():
        if r["x_min"] <= cx < r["x_max"] and r["y_min"] <= cy < r["y_max"]:
            return name
    return best


def analyze_proximity(
    detections_a: List[Detection],
    detections_b: List[Detection],
    max_dist_norm: float = 0.15
) -> List[SpatialRelationship]:
    """
    Find objects in detections_a that are within max_dist_norm of any object
    in detections_b. Returns SpatialRelationship list.
    """
    if not detections_a or not detections_b:
        return []

    relationships = []
    a_class = detections_a[0].class_name if detections_a else "object"
    b_class = detections_b[0].class_name if detections_b else "object"
    near_ids = []

    for da in detections_a:
        ca_x = da.centroid_norm.get("x", 0.5)
        ca_y = da.centroid_norm.get("y", 0.5)
        for db in detections_b:
            cb_x = db.centroid_norm.get("x", 0.5)
            cb_y = db.centroid_norm.get("y", 0.5)
            dist = math.hypot(ca_x - cb_x, ca_y - cb_y)
            if dist <= max_dist_norm:
                if da.id not in near_ids:
                    near_ids.append(da.id)

    if near_ids:
        relationships.append(SpatialRelationship(
            subject_class=a_class,
            relation="near",
            object_class=b_class,
            instances=near_ids,
        ))
    return relationships


def describe_distribution(detections: List[Detection], has_geo: bool = False) -> str:
    """
    Produce a sentence describing where detected objects are concentrated.
    Example: "Most buildings are concentrated in the central area, with
    some in the upper-left."
    """
    if not detections:
        return "No objects detected."

    position_counts: Dict[str, int] = {}
    for d in detections:
        pos = d.image_position
        position_counts[pos] = position_counts.get(pos, 0) + 1

    sorted_pos = sorted(position_counts.items(), key=lambda x: -x[1])
    class_name = detections[0].class_name + "s"

    if len(sorted_pos) == 1:
        region, cnt = sorted_pos[0]
        return f"All {len(detections)} {class_name} detected are located in the {region} area."

    dominant_region, dominant_count = sorted_pos[0]
    rest = sorted_pos[1:]

    msg = f"Most {class_name} ({dominant_count} of {len(detections)}) are concentrated in the {dominant_region} area"
    if rest:
        rest_desc = ", ".join(f"{cnt} in the {r}" for r, cnt in rest[:3])
        msg += f", with additional instances in the {rest_desc}"
    msg += "."
    return msg


def pixel_coords_to_geo(
    px_x: float, px_y: float,
    img_w: int, img_h: int,
    geo_meta: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """
    Convert pixel coordinates to WGS84 lat/lon.
    Only works when real geo_meta is present. Returns None otherwise.
    """
    if not geo_meta:
        return None
    try:
        lon_min = geo_meta["west"]
        lon_max = geo_meta["east"]
        lat_max = geo_meta["north"]
        lat_min = geo_meta["south"]
        lon = lon_min + (px_x / float(img_w)) * (lon_max - lon_min)
        lat = lat_max - (px_y / float(img_h)) * (lat_max - lat_min)
        return {"lat": round(lat, 6), "lon": round(lon, 6)}
    except Exception:
        return None


def bbox_norm_to_geo(
    bbox_norm: Dict[str, float],
    img_w: int, img_h: int,
    geo_meta: Optional[Dict[str, Any]]
) -> Optional[Dict[str, float]]:
    """Return geographic centroid of a normalized bbox, or None if no georef."""
    if not geo_meta:
        return None
    cx_px = ((bbox_norm["x_min"] + bbox_norm["x_max"]) / 2.0) * img_w
    cy_px = ((bbox_norm["y_min"] + bbox_norm["y_max"]) / 2.0) * img_h
    return pixel_coords_to_geo(cx_px, cy_px, img_w, img_h, geo_meta)
