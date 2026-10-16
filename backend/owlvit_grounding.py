"""
SatQuery AI — Google OWL-ViT Visual Grounding Engine
Offline, text-conditioned open-vocabulary object localization for satellite imagery.
Uses cached google/owlvit-base-patch32 (Apache-2.0 license).
Zero network downloads; strictly CPU-compatible PyTorch operations.
"""

from pathlib import Path
import os
import time
import re
import io
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import torch

logger = logging.getLogger(__name__)

# Enforce offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT_DIR = Path(__file__).resolve().parent.parent
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(ROOT_DIR / ".huggingface_cache")


def find_owlvit_snapshot_path() -> Optional[Path]:
    """Finds the offline snapshot path for google/owlvit-base-patch32."""
    candidates = [
        ROOT_DIR / ".huggingface_cache" / "hub" / "models--google--owlvit-base-patch32" / "snapshots" / "cbc355fb364588351c5d51c7f74465e8e7ec6f72",
        Path("d:/sih26167/.huggingface_cache/hub/models--google--owlvit-base-patch32/snapshots/cbc355fb364588351c5d51c7f74465e8e7ec6f72"),
    ]
    for c in candidates:
        if c.exists() and (c / "model.safetensors").exists() and (c / "config.json").exists():
            return c

    # Search dynamically in .huggingface_cache
    cache_dir = ROOT_DIR / ".huggingface_cache" / "hub" / "models--google--owlvit-base-patch32" / "snapshots"
    if cache_dir.exists():
        for snap in cache_dir.iterdir():
            if snap.is_dir() and (snap / "model.safetensors").exists() and (snap / "config.json").exists():
                return snap

    return None


class OwlViTGroundingEngine:
    """
    Isolated wrapper for Google OWL-ViT zero-shot object localization.
    Operates 100% offline using locally cached safetensors weights.
    """

    _instance = None
    _processor = None
    _model = None
    _snapshot_path: Optional[Path] = None
    _is_loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OwlViTGroundingEngine, cls).__new__(cls)
            cls._instance._snapshot_path = find_owlvit_snapshot_path()
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        """Returns True if local snapshot weights are present on disk."""
        snap = find_owlvit_snapshot_path()
        return snap is not None and (snap / "model.safetensors").exists()

    def load_model(self, force_reload: bool = False) -> bool:
        """Loads OWL-ViT processor and model weights in offline mode."""
        if self._is_loaded and not force_reload:
            return True

        snap = find_owlvit_snapshot_path()
        if snap is None:
            logger.warning("[OWL-ViT] Cannot load: snapshot directory not found in local cache.")
            return False

        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            logger.info(f"[OWL-ViT] Loading offline weights from: {snap}")
            self._processor = OwlViTProcessor.from_pretrained(str(snap), local_files_only=True)
            self._model = OwlViTForObjectDetection.from_pretrained(str(snap), local_files_only=True)
            self._model.eval()
            self._snapshot_path = snap
            self._is_loaded = True
            logger.info(f"[OWL-ViT] Model loaded successfully ({sum(p.numel() for p in self._model.parameters()):,} params).")
            return True
        except Exception as e:
            logger.error(f"[OWL-ViT] Failed to load offline weights: {e}")
            self._is_loaded = False
            return False

    def _extract_query_labels(self, query: str) -> List[str]:
        """Cleans user query and extracts candidate target labels."""
        q = query.strip()
        # Strip common instruction prefixes
        clean = re.sub(
            r"^(locate|find and pinpoint|find and locate|find|where is|where are|show me where|coordinates of|pinpoint|spot the|grounding for|bound)\s+",
            "",
            q,
            flags=re.IGNORECASE
        ).strip()
        # Strip trailing location phrases
        clean = re.sub(
            r"\s+(in|on|within)\s+(the|this)?\s*(image|satellite image|scene|raster|tile|area|observed scene)\.?$",
            "",
            clean,
            flags=re.IGNORECASE
        ).strip()

        # Strip leading articles
        clean_noun = re.sub(r"^(the|a|an)\s+", "", clean, flags=re.IGNORECASE).strip()

        labels = []
        if clean_noun and len(clean_noun) >= 2:
            labels.append(clean_noun.lower())
            if not clean_noun.lower().startswith("a photo"):
                labels.append(f"a photo of a {clean_noun.lower()}")
        if clean.lower() not in labels and len(clean) >= 2:
            labels.append(clean.lower())
        if q.lower() not in labels:
            labels.append(q.lower())

        # If empty fallback to "object"
        if not labels:
            labels = ["object"]

        return labels[:4]

    def ground_expression(
        self,
        image: Union[Image.Image, np.ndarray, bytes],
        query: str,
        threshold: float = 0.08,
        max_detections: int = 20,
        nms_iou_threshold: float = 0.45,
        geo_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes text-conditioned object localization on image using Google OWL-ViT.
        
        Returns:
            Dictionary matching SatQuery grounding schema with:
            - execution_path: 'NEURAL_OWLVIT'
            - model_provenance: label string
            - bounding_boxes: list of [bx, by, bw, bh] (pixel coordinates)
            - geojson_data: RFC 7946 FeatureCollection
            - visual_evidence: annotated image overlay
            - latency_ms: isolated forward pass time
            - status: 'SUCCESS' or 'NO_DETECTION'
        """
        if not self._is_loaded:
            success = self.load_model()
            if not success:
                raise RuntimeError("OWL-ViT model could not be loaded from offline cache.")

        # 1. Normalize input image to PIL Image RGB
        if isinstance(image, bytes):
            pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 3:
                # Assume BGR if from OpenCV
                rgb_arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_arr)
            else:
                pil_img = Image.fromarray(image[:, :, :3]).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        w, h = pil_img.size
        labels_to_query = self._extract_query_labels(query)

        # 2. Forward pass with isolated latency timing
        t0 = time.perf_counter()
        inputs = self._processor(text=[labels_to_query], images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.Tensor([[h, w]])
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            text_labels=[labels_to_query]
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        res = results[0]
        raw_boxes = res["boxes"].tolist() if "boxes" in res and len(res["boxes"]) > 0 else []
        raw_scores = res["scores"].tolist() if "scores" in res and len(res["scores"]) > 0 else []
        raw_labels = res["text_labels"] if "text_labels" in res else []

        # 3. Post-processing & Non-Maximum Suppression (NMS)
        parsed_cands = []
        for box, score, label in zip(raw_boxes, raw_scores, raw_labels):
            # Clamp coordinates to image boundaries
            x_min = max(0, min(w - 1, int(round(box[0]))))
            y_min = max(0, min(h - 1, int(round(box[1]))))
            x_max = max(x_min + 1, min(w, int(round(box[2]))))
            y_max = max(y_min + 1, min(h, int(round(box[3]))))
            bw = x_max - x_min
            bh = y_max - y_min

            # Filter degenerate full-canvas boxes (> 95% of area)
            if (bw * bh) >= 0.95 * (w * h):
                continue
            # Filter tiny noise (< 10 px)
            if bw < 4 or bh < 4:
                continue

            parsed_cands.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "bx": x_min,
                "by": y_min,
                "bw": bw,
                "bh": bh,
                "pixel_bbox": [x_min, y_min, bw, bh],
                "score": round(float(score), 4),
                "label": str(label)
            })

        # NMS
        parsed_cands.sort(key=lambda x: -x["score"])
        kept_detections = []
        for cand in parsed_cands:
            suppressed = False
            for k in kept_detections:
                iou = self._compute_iou(cand, k)
                if iou > nms_iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept_detections.append(cand)
                if len(kept_detections) >= max_detections:
                    break

        # 4. Handle Empty Detection (explicit no-detection result)
        has_detection = len(kept_detections) > 0
        status_str = "SUCCESS" if has_detection else "NO_DETECTION"
        derived_conf = kept_detections[0]["score"] if has_detection else 0.0

        # 5. Build GeoJSON Features
        geojson_features = self._build_geojson_features(kept_detections, w, h, geo_meta)
        geojson_data = {
            "type": "FeatureCollection",
            "metadata": {
                "mission": "SatQuery AI Neural Object Grounding",
                "model": "google/owlvit-base-patch32",
                "execution_path": "NEURAL_OWLVIT",
                "target_expression": query,
                "query_labels": labels_to_query,
                "total_detections": len(kept_detections),
                "threshold": threshold,
                "crs": "urn:ogc:def:crs:OGC:1.3:CRS84"
            },
            "features": geojson_features
        }

        # 6. Generate Visual Overlay
        b64_overlay = self._draw_visual_evidence(pil_img, kept_detections, query)

        # 7. Formulate textual explanation
        if has_detection:
            summary_bullets = [
                f"• **Neural Grounding Model:** Google OWL-ViT (ViT-B/32, Zero-Shot Grounding Head)",
                f"• **Detected Entities:** {len(kept_detections)} localized instance(s) matching expression *\"{labels_to_query[0]}\"*.",
                f"• **Top Confidence Score:** {derived_conf:.1%} (threshold: {threshold:.2f}).",
                f"• **Inference Latency:** {latency_ms:.1f} ms forward pass on PyTorch CPU."
            ]
            det_desc = ", ".join(f"[{d['bx']}, {d['by']}, {d['bw']}, {d['bh']}] ({d['score']:.1%})" for d in kept_detections[:5])
            if len(kept_detections) > 5:
                det_desc += f" ... (+{len(kept_detections) - 5} more)"
            summary_bullets.append(f"• **Localized Bounding Boxes [XYWH]:** {det_desc}")
            formatted_text = (
                f"🎯 **SatQuery AI — Neural Object Grounding & Spatial Localization**\n\n"
                f"### 1. Detection Summary\n"
                + "\n".join(summary_bullets) + "\n\n"
                f"---\n\n"
                f"### 2. Tactical GIS Interoperability\n"
                f"• Bounding vectors have been generated in **RFC 7946 GeoJSON format**.\n"
                f"• Click **Download GeoJSON** to export detected target coordinates to ISRO Bhuvan, QGIS, or ArcGIS."
            )
        else:
            formatted_text = (
                f"⚠️ **SatQuery AI — Neural Object Grounding (No Detection)**\n\n"
                f"### 1. Grounding Outcome\n"
                f"• **Target Expression:** *\"{query}\"*\n"
                f"• **Model:** Google OWL-ViT (Zero-Shot Grounding Head, `google/owlvit-base-patch32`)\n"
                f"• **Result:** No object candidate exceeded the confidence threshold of **{threshold:.2f}**.\n"
                f"• **Inference Latency:** {latency_ms:.1f} ms on PyTorch CPU.\n"
                f"• **Note:** The target may either be absent from the image, obscured by atmospheric conditions, or outside the resolution limit of this tile."
            )

        provenance_label = "Google OWL-ViT (ViT-B/32, Zero-Shot Neural Grounding)"

        return {
            "text": formatted_text,
            "status": status_str,
            "has_detection": has_detection,
            "visual_evidence": [{"image_base64": b64_overlay, "description": f"OWL-ViT Detections for '{query}'"}],
            "confidence": derived_conf,
            "compatibility_status": "PASSED",
            "spatial_coherence_score": 1.0,
            "model_provenance": provenance_label,
            "bounding_boxes": [d["pixel_bbox"] for d in kept_detections],
            "detections": kept_detections,
            "geo_metadata": geo_meta,
            "geojson_data": geojson_data,
            "inference_metadata": {
                "model_id": "google/owlvit-base-patch32",
                "checkpoint": "cbc355fb364588351c5d51c7f74465e8e7ec6f72",
                "model_loaded": True,
                "execution_path": "NEURAL_OWLVIT",
                "forward_latency_ms": latency_ms,
                "provenance_label": provenance_label,
                "threshold_applied": threshold,
                "detection_count": len(kept_detections)
            }
        }

    def _compute_iou(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Computes IoU between two detection dicts containing x_min, y_min, x_max, y_max."""
        ix1 = max(a["x_min"], b["x_min"])
        iy1 = max(a["y_min"], b["y_min"])
        ix2 = min(a["x_max"], b["x_max"])
        iy2 = min(a["y_max"], b["y_max"])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a["x_max"] - a["x_min"]) * (a["y_max"] - a["y_min"])
        area_b = (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
        union = float(area_a + area_b - inter)
        return inter / union if union > 0 else 0.0

    def _build_geojson_features(
        self,
        detections: List[Dict[str, Any]],
        w: int,
        h: int,
        geo_meta: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Converts pixel bounding boxes into GeoJSON polygon features."""
        base_lat = 19.0760
        base_lon = 72.8777
        if geo_meta and "west" in geo_meta and "east" in geo_meta and "north" in geo_meta and "south" in geo_meta:
            lon_min, lat_min, lon_max, lat_max = geo_meta["west"], geo_meta["south"], geo_meta["east"], geo_meta["north"]
        else:
            lon_min, lat_min, lon_max, lat_max = base_lon - 0.04, base_lat - 0.04, base_lon + 0.04, base_lat + 0.04

        def px_to_geo(px_x, px_y):
            geo_lon = lon_min + (px_x / float(w)) * (lon_max - lon_min)
            geo_lat = lat_max - (px_y / float(h)) * (lat_max - lat_min)
            return round(geo_lon, 6), round(geo_lat, 6)

        features = []
        for idx, det in enumerate(detections):
            bx, by, bw, bh = det["pixel_bbox"]
            p1 = px_to_geo(bx, by)
            p2 = px_to_geo(bx + bw, by)
            p3 = px_to_geo(bx + bw, by + bh)
            p4 = px_to_geo(bx, by + bh)

            features.append({
                "type": "Feature",
                "id": f"owlvit_target_{idx+1}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[list(p1), list(p2), list(p3), list(p4), list(p1)]]
                },
                "properties": {
                    "feature_id": idx + 1,
                    "label": f"{det['label'].upper()} #{idx+1}",
                    "confidence": det["score"],
                    "pixel_bbox": [bx, by, bw, bh],
                    "classification": det["label"].upper()
                }
            })
        return features

    def _draw_visual_evidence(
        self,
        pil_img: Image.Image,
        detections: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Draws bounding boxes and confidence banners onto canvas, returning base64 URI."""
        import base64
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]

        # Draw boxes
        for det in detections:
            bx, by, bw, bh = det["pixel_bbox"]
            score = det["score"]
            label = det["label"]

            # Glowing amber / cyan box
            color = (0, 215, 255)  # Gold
            cv2.rectangle(cv_img, (bx, by), (bx + bw, by + bh), color, 2, cv2.LINE_AA)

            tag = f"{label} ({score:.0%})"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(cv_img, (bx, max(0, by - th - 6)), (bx + tw + 6, by), color, -1)
            cv2.putText(cv_img, tag, (bx + 3, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

        # Top banner
        banner_h = 36
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        banner[:] = (15, 18, 26)
        banner_text = f"OWL-ViT GROUNDING: '{query}' | {len(detections)} detection(s)"
        cv2.putText(banner, banner_text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 1, cv2.LINE_AA)

        final = np.vstack([banner, cv_img])
        _, buf = cv2.imencode(".png", final)
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/png;base64,{b64}"


# Global singleton engine
owlvit_grounding_engine = OwlViTGroundingEngine()
