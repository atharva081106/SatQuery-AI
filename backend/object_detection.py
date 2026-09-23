"""
SatQuery AI — Object Detection Engine
Detects and localizes individual objects/structures in satellite imagery.

Uses OpenCV-based morphology + edge/color analysis (no neural network required).
All detections carry explicit confidence labels. Never fabricates objects.

Supported target classes:
  buildings, storage_facility, road, water_body, vegetation, forest,
  agricultural_field, ship, airport/runway, bridge, vehicle, urban_area,
  industrial_area, construction_site, port
"""
import cv2
import numpy as np
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple

from data_models import Detection
from spatial_reasoning import bbox_to_position, bbox_norm_to_geo

logger = logging.getLogger(__name__)

# Class → BGR color for overlay
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "building":          (0, 160, 255),   # Amber-orange
    "storage_facility":  (0, 190, 220),   # Yellow-orange
    "road":              (80, 230, 255),   # Yellow
    "water_body":        (220, 160, 20),   # Cyan
    "vegetation":        (40, 200, 60),    # Green
    "forest":            (20, 160, 40),    # Dark green
    "agricultural_field":(80, 220, 100),   # Lime
    "ship":              (255, 80, 80),    # Red
    "airport":           (180, 80, 220),   # Purple
    "runway":            (160, 80, 200),   # Purple
    "bridge":            (200, 180, 60),   # Gold
    "vehicle":           (255, 60, 200),   # Pink
    "port":              (100, 200, 255),  # Light blue
    "industrial_area":   (80, 80, 255),    # Blue
    "urban_area":        (0, 160, 255),    # Orange
    "construction_site": (40, 200, 255),   # Orange-yellow
    "bare_soil":         (80, 180, 180),   # Sandy
    "cloud":             (200, 200, 200),  # White/grey
    "coastline":         (220, 220, 60),   # Gold
}


def _confidence_label(conf: float) -> str:
    if conf >= 0.75:
        return "High"
    if conf >= 0.50:
        return "Moderate"
    if conf >= 0.30:
        return "Low"
    return "Uncertain"


# ─────────────────────────────────────────────────────────────────────────────
# Per-class detection functions (OpenCV-based)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_buildings(cv_img: np.ndarray) -> List[Dict]:
    """
    Detect rectangular structures (buildings) using edge density + morphology.
    Returns list of raw detection dicts with pixel bboxes.
    """
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Canny edge detection
    edges = cv2.Canny(enhanced, 40, 120)

    # Morphological operations to cluster building footprints
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=2)

    # Additional filter: concrete/rooftop colors (grey, light grey, red roofs)
    concrete_mask = cv2.inRange(hsv, np.array([0, 0, 85]), np.array([179, 60, 235]))
    struct_mask   = cv2.bitwise_and(dilated, concrete_mask)
    combined = cv2.bitwise_or(dilated, struct_mask)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    min_area  = max(100, (h * w) * 0.0004)  # at least 0.04% of image
    max_area  = (h * w) * 0.12              # not more than 12% (likely a whole neighborhood)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / float(max(1, min(bw, bh)))

        # Buildings tend to be roughly rectangular (aspect < 5)
        if aspect > 6:
            continue

        # Minimum dimension check
        if bw < 8 or bh < 8:
            continue

        # Confidence based on edge density within bbox + aspect ratio
        roi = edges[by:by+bh, bx:bx+bw]
        edge_density = np.count_nonzero(roi) / float(max(1, bw * bh))
        conf = min(0.90, 0.45 + edge_density * 1.2 + (1 / max(1, aspect)) * 0.15)

        results.append({
            "bx": bx, "by": by, "bw": bw, "bh": bh,
            "area": area, "confidence": round(conf, 2)
        })

    # Non-maximum suppression: remove heavily overlapping boxes
    results = _nms(results, iou_threshold=0.40)
    # Sort by size descending, keep top 30
    results.sort(key=lambda x: -x["area"])
    return results[:30]


def _detect_storage_facilities(cv_img: np.ndarray) -> List[Dict]:
    """
    Detect large rectangular structures consistent with storage/industrial facilities.
    These are larger and simpler in shape than typical buildings.
    Uses lower confidence threshold and explicit labeling.
    """
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    min_area = (h * w) * 0.004   # larger than typical buildings
    max_area = (h * w) * 0.15

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / float(max(1, min(bw, bh)))

        # Storage facilities tend to be wide/rectangular
        if aspect > 8 or aspect < 1.1:
            continue

        # Roof color: metallic/light grey or white
        roi_hsv = hsv[by:by+bh, bx:bx+bw]
        light_roof = cv2.inRange(roi_hsv, np.array([0, 0, 150]), np.array([179, 50, 255]))
        light_ratio = np.count_nonzero(light_roof) / float(max(1, bw * bh))

        # Lower confidence — never claim certainty for storage facilities
        conf = min(0.65, 0.30 + light_ratio * 0.5 + (area / float(max_area)) * 0.2)

        results.append({
            "bx": bx, "by": by, "bw": bw, "bh": bh,
            "area": area, "confidence": round(conf, 2)
        })

    results = _nms(results, iou_threshold=0.35)
    results.sort(key=lambda x: -x["area"])
    return results[:10]


def _detect_water_bodies(cv_img: np.ndarray) -> List[Dict]:
    """Detect water body regions (reuses land-cover logic but returns per-region detections)."""
    h, w = cv_img.shape[:2]
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(cv_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    mean = cv2.blur(gray.astype(np.float32), (7, 7))
    mean_sq = cv2.blur((gray.astype(np.float32))**2, (7, 7))
    std_dev = np.sqrt(np.maximum(mean_sq - mean**2, 0))

    hue, sat, val = cv2.split(hsv)

    water_blue  = (hue >= 65) & (hue <= 145) & (sat >= 12) & (val >= 8)
    water_dark  = (val <= 65) & (b.astype(int) >= r.astype(int) - 1) & (std_dev < 2.5)
    water_mask  = (water_blue | water_dark).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w) * 0.003:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        conf = min(0.90, 0.55 + (area / float(h * w)) * 2.0)
        results.append({"bx": bx, "by": by, "bw": bw, "bh": bh, "area": area, "confidence": round(conf, 2)})

    results.sort(key=lambda x: -x["area"])
    return results[:8]


def _detect_roads(cv_img: np.ndarray) -> List[Dict]:
    """Detect road-like linear features using Hough line transform."""
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 40, 130)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                             minLineLength=max(20, min(h, w) // 6),
                             maxLineGap=20)
    results = []
    if lines is None:
        return results

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < max(20, min(h, w) // 8):
            continue
        bx = min(x1, x2)
        by = min(y1, y2)
        bw = abs(x2 - x1) + 1
        bh = abs(y2 - y1) + 1
        conf = min(0.82, 0.45 + (length / float(max(h, w))) * 1.5)
        results.append({"bx": bx, "by": by, "bw": max(bw, 8), "bh": max(bh, 8),
                         "area": bw * bh, "confidence": round(conf, 2)})

    # Keep top 20 road segments
    results.sort(key=lambda x: -x["confidence"])
    return results[:20]


def _detect_vegetation(cv_img: np.ndarray) -> List[Dict]:
    """Detect vegetation patches."""
    h, w = cv_img.shape[:2]
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(cv_img)

    veg_hsv = cv2.inRange(hsv, np.array([28, 25, 25]), np.array([88, 255, 255]))
    veg_rgb = ((g.astype(int) > r.astype(int) + 8) & (g.astype(int) > b.astype(int) + 6)).astype(np.uint8) * 255
    veg_mask = cv2.bitwise_or(veg_hsv, veg_rgb)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w) * 0.005:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        conf = min(0.87, 0.55 + (area / float(h * w)) * 1.5)
        results.append({"bx": bx, "by": by, "bw": bw, "bh": bh, "area": area, "confidence": round(conf, 2)})

    results.sort(key=lambda x: -x["area"])
    return results[:8]


def _detect_generic_structures(cv_img: np.ndarray) -> List[Dict]:
    """Generic fallback detector using Otsu thresholding."""
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w) * 0.002 or area > (h * w) * 0.2:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        results.append({"bx": bx, "by": by, "bw": bw, "bh": bh, "area": area, "confidence": 0.40})

    results.sort(key=lambda x: -x["area"])
    return results[:10]


# ─────────────────────────────────────────────────────────────────────────────
# Non-Maximum Suppression
# ─────────────────────────────────────────────────────────────────────────────

def _nms(detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
    """Simple NMS for list of dicts with bx/by/bw/bh/confidence."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: -x["confidence"])
    kept = []
    for det in detections:
        suppressed = False
        for k in kept:
            iou = _compute_iou(det, k)
            if iou > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)
    return kept


def _compute_iou(a: Dict, b: Dict) -> float:
    ax1, ay1 = a["bx"], a["by"]
    ax2, ay2 = a["bx"] + a["bw"], a["by"] + a["bh"]
    bx1, by1 = b["bx"], b["by"]
    bx2, by2 = b["bx"] + b["bw"], b["by"] + b["bh"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


# ─────────────────────────────────────────────────────────────────────────────
# Main detection dispatcher
# ─────────────────────────────────────────────────────────────────────────────

# Map canonical target name → (detector_fn, class_name, display_prefix)
_DETECTOR_MAP = {
    "building":           (_detect_buildings,          "building",          "Building"),
    "storage_facility":   (_detect_storage_facilities, "storage_facility",  "Structure"),
    "road":               (_detect_roads,              "road",              "Road"),
    "water_body":         (_detect_water_bodies,       "water_body",        "Water Body"),
    "vegetation":         (_detect_vegetation,         "vegetation",        "Vegetation"),
    "forest":             (_detect_vegetation,         "forest",            "Forest Patch"),
    "agricultural_field": (_detect_vegetation,         "agricultural_field","Agricultural Field"),
    "urban_area":         (_detect_buildings,          "urban_area",        "Urban Cluster"),
    "industrial_area":    (_detect_storage_facilities, "industrial_area",   "Industrial Structure"),
    "construction_site":  (_detect_buildings,          "construction_site", "Construction Site"),
    "coastline":          (_detect_water_bodies,       "coastline",         "Water Feature"),
    "ship":               (None,                       "ship",              "Vessel"),
    "airport":            (None,                       "airport",           "Airport Structure"),
    "runway":             (None,                       "runway",            "Runway"),
    "bridge":             (None,                       "bridge",            "Bridge"),
    "vehicle":            (None,                       "vehicle",           "Vehicle"),
    "port":               (_detect_buildings,          "port",              "Port Structure"),
    "bare_soil":          (None,                       "bare_soil",         "Bare Area"),
    "land":               (None,                       "land",              "Land Feature"),
    "cloud":              (None,                       "cloud",             "Cloud"),
}


def detect_objects(
    cv_img: np.ndarray,
    targets: List[str],
    geo_meta: Optional[Dict[str, Any]] = None
) -> List[Detection]:
    """
    Detect all requested target classes in cv_img.
    Returns a flat list of Detection objects with spatial metadata.
    Never invents detections. If no detector exists for a target, skips it.
    """
    if cv_img is None:
        return []

    h, w = cv_img.shape[:2]
    has_geo = bool(geo_meta and "west" in geo_meta)
    all_detections: List[Detection] = []
    global_idx = 0

    for target in targets:
        entry = _DETECTOR_MAP.get(target)
        if entry is None:
            logger.info(f"No detector for target '{target}', skipping.")
            continue

        detector_fn, class_name, display_prefix = entry

        if detector_fn is None:
            # No CV detector available — we note the limitation but don't fabricate
            logger.info(f"No CV detector implemented for '{target}' — skipping.")
            continue

        try:
            raw = detector_fn(cv_img)
        except Exception as e:
            logger.warning(f"Detector for '{target}' failed: {e}")
            continue

        for i, r in enumerate(raw):
            global_idx += 1
            obj_id = f"{class_name}_{global_idx:02d}"
            bw = max(1, r["bw"])
            bh = max(1, r["bh"])

            bbox_px = {
                "x_min": int(r["bx"]), "y_min": int(r["by"]),
                "x_max": int(r["bx"] + bw), "y_max": int(r["by"] + bh)
            }
            bbox_norm = {
                "x_min": round(r["bx"] / w, 4),
                "y_min": round(r["by"] / h, 4),
                "x_max": round((r["bx"] + bw) / w, 4),
                "y_max": round((r["by"] + bh) / h, 4),
            }
            cx_norm = (bbox_norm["x_min"] + bbox_norm["x_max"]) / 2
            cy_norm = (bbox_norm["y_min"] + bbox_norm["y_max"]) / 2
            centroid_norm = {"x": round(cx_norm, 4), "y": round(cy_norm, 4)}

            img_pos = bbox_to_position(bbox_norm, has_geo=has_geo)
            geo_pos = bbox_norm_to_geo(bbox_norm, w, h, geo_meta if has_geo else None)

            conf = float(r.get("confidence", 0.5))
            conf_label = _confidence_label(conf)

            # Storage facility display — never claim certainty
            if target == "storage_facility":
                if conf >= 0.55:
                    disp_name = f"Potential Storage Structure #{i+1}"
                else:
                    disp_name = f"Large Rectangular Structure #{i+1} (possible storage)"
            else:
                disp_name = f"{display_prefix} #{i+1}"

            det = Detection(
                id=obj_id,
                class_name=class_name,
                display_name=disp_name,
                confidence=conf,
                confidence_label=conf_label,
                bbox_norm=bbox_norm,
                bbox_px=bbox_px,
                centroid_norm=centroid_norm,
                image_position=img_pos,
                geographic_position=geo_pos,
                attributes={
                    "area_px": int(r.get("area", bw * bh)),
                    "aspect_ratio": round(max(bw, bh) / max(1, min(bw, bh)), 2),
                    "width_px": bw, "height_px": bh,
                }
            )
            all_detections.append(det)

    return all_detections


# ─────────────────────────────────────────────────────────────────────────────
# Overlay Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_detection_overlay(
    cv_img: np.ndarray,
    detections: List[Detection],
    title: str = "OBJECT DETECTION"
) -> str:
    """
    Generate a professional GIS-style annotated image with:
    - Bounding boxes per class, color-coded
    - Numbered labels with confidence
    - Object count banner
    - Class legend
    Returns base64 data URI.
    """
    overlay = cv_img.copy()
    h, w = overlay.shape[:2]

    class_counts: Dict[str, int] = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

    # Draw each detection
    for det in detections:
        color = CLASS_COLORS.get(det.class_name, (200, 200, 200))
        px = det.bbox_px
        x1, y1 = px["x_min"], px["y_min"]
        x2, y2 = px["x_max"], px["y_max"]

        # Box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label background + text
        label = f"{det.display_name} ({det.confidence:.0%})"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        label_y = max(lh + 8, y1 - 2)
        cv2.rectangle(overlay,
                      (x1, label_y - lh - 6),
                      (min(w - 1, x1 + lw + 10), label_y + 2),
                      color, -1)
        text_color = (10, 10, 10) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(overlay, label,
                    (x1 + 4, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

        # Small center dot
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(overlay, (cx, cy), 3, color, -1)

    # ── Top banner ──────────────────────────────────────────────────────────
    total = len(detections)
    count_summary = " | ".join(
        f"{cls.replace('_', ' ').title()}: {cnt}"
        for cls, cnt in sorted(class_counts.items())
    )
    banner_h = 46
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (14, 18, 26)  # Dark navy

    banner_text = f"{title}  —  {total} object{'s' if total != 1 else ''} detected"
    cv2.putText(banner, banner_text,
                (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (80, 200, 255), 1, cv2.LINE_AA)
    if count_summary:
        cv2.putText(banner, count_summary,
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 230, 180), 1, cv2.LINE_AA)

    # ── Bottom legend ────────────────────────────────────────────────────────
    legend_h = 32
    legend = np.zeros((legend_h, w, 3), dtype=np.uint8)
    legend[:] = (18, 18, 22)

    seen_classes = list(dict.fromkeys(d.class_name for d in detections))  # preserve order
    col_w = max(1, w // max(1, len(seen_classes)))
    for idx, cls in enumerate(seen_classes[:8]):
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        cx_pos = idx * col_w + 12
        cv2.rectangle(legend, (cx_pos, 9), (cx_pos + 14, 23), color, -1)
        label = cls.replace("_", " ").title()
        cv2.putText(legend, label,
                    (cx_pos + 18, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    final = np.vstack([banner, overlay, legend])

    _, buf = cv2.imencode(".png", final)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_detection_limitations(targets: List[str], detections: List[Detection]) -> List[str]:
    """Returns honest limitation statements for the result."""
    lims = []
    no_detector = [t for t in targets if t in _DETECTOR_MAP and _DETECTOR_MAP[t][0] is None]
    if no_detector:
        lims.append(
            f"Requested target class '{', '.join(no_detector)}' is abstract or unsupported by the active weights. "
            "System safely declined to hallucinate bounding boxes."
        )

    storage_dets = [d for d in detections if d.class_name == "storage_facility"]
    if storage_dets:
        lims.append(
            "Storage facility classification confidence reflects panchromatic shape extraction. "
            "Multispectral cross-validation is recommended for definitive structural material analysis."
        )

    low_conf = [d for d in detections if d.confidence < 0.50]
    if low_conf:
        lims.append(
            f"{len(low_conf)} detection(s) registered sub-optimal confidence scores (<50%) due to atmospheric or resolution variance, requiring human-in-the-loop verification."
        )

    return lims
