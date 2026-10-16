"""
backend/tiling.py
Sliding Window Tiling, Stitching, and Large Raster Inference Engine for SatQuery AI.

Supports:
1. Memory-bounded sliding window tiling for large remote sensing imagery (up to 10k x 10k+).
2. Smooth border blending with 2D Hann / Cosine apodization windows (eliminates seam artifacts).
3. Probability / mask stitching with multi-class argmax.
4. Tile-to-global bounding box re-projection with cross-tile Non-Maximum Suppression (NMS).
5. Geographic coordinates interpolation for sub-tile features.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TileWindow:
    tile_id: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    tile_h: int
    tile_w: int
    full_h: int
    full_w: int
    geo_meta: Optional[Dict[str, Any]] = None


def compute_hann_window(height: int, width: int, overlap_h: int, overlap_w: int) -> np.ndarray:
    """
    Generate a 2D apodization weight matrix with smooth cosine transitions
    along the overlap margins to prevent tile stitching seams.
    """
    def _1d_ramp(length: int, overlap: int) -> np.ndarray:
        if overlap <= 0 or overlap * 2 >= length:
            return np.ones(length, dtype=np.float32)
        ramp = np.ones(length, dtype=np.float32)
        # Cosine ramp from 0 to 1
        t = np.linspace(0, np.pi / 2, overlap)
        ramp[:overlap] = np.sin(t) ** 2
        ramp[-overlap:] = np.cos(t) ** 2
        return ramp

    ramp_y = _1d_ramp(height, overlap_h)
    ramp_x = _1d_ramp(width, overlap_w)
    weight_2d = np.outer(ramp_y, ramp_x)
    return np.clip(weight_2d, 1e-4, 1.0).astype(np.float32)


class SlidingWindowTiler:
    """
    Memory-bounded sliding window inference and seamless stitching engine.
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        max_dimension: int = 4096
    ):
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.stride = max(1, self.tile_size - self.overlap)
        self.max_dimension = max_dimension

    def get_windows(
        self,
        height: int,
        width: int,
        geo_meta: Optional[Dict[str, Any]] = None
    ) -> List[TileWindow]:
        """
        Compute grid of overlapping window coordinates covering (height, width).
        Guarantees 100% coverage up to the right and bottom image borders.
        """
        windows: List[TileWindow] = []
        tile_id = 0

        y_starts = list(range(0, max(1, height - self.tile_size + 1), self.stride))
        if not y_starts or y_starts[-1] + self.tile_size < height:
            y_starts.append(max(0, height - self.tile_size))

        x_starts = list(range(0, max(1, width - self.tile_size + 1), self.stride))
        if not x_starts or x_starts[-1] + self.tile_size < width:
            x_starts.append(max(0, width - self.tile_size))

        # Deduplicate and sort
        y_starts = sorted(list(set(y_starts)))
        x_starts = sorted(list(set(x_starts)))

        for y in y_starts:
            for x in x_starts:
                y_end = min(height, y + self.tile_size)
                x_end = min(width, x + self.tile_size)
                tile_h = y_end - y
                tile_w = x_end - x

                tile_geo = None
                if geo_meta and "west" in geo_meta and "north" in geo_meta:
                    west = geo_meta["west"]
                    east = geo_meta["east"]
                    north = geo_meta["north"]
                    south = geo_meta["south"]
                    crs = geo_meta.get("crs", "EPSG:4326")

                    res_x = (east - west) / float(width)
                    res_y = (north - south) / float(height)

                    tile_geo = {
                        "west": west + x * res_x,
                        "east": west + x_end * res_x,
                        "north": north - y * res_y,
                        "south": north - y_end * res_y,
                        "crs": crs
                    }

                windows.append(TileWindow(
                    tile_id=tile_id,
                    y_start=y,
                    y_end=y_end,
                    x_start=x,
                    x_end=x_end,
                    tile_h=tile_h,
                    tile_w=tile_w,
                    full_h=height,
                    full_w=width,
                    geo_meta=tile_geo
                ))
                tile_id += 1

        return windows

    def split_raster(
        self,
        raster: np.ndarray,
        geo_meta: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TileWindow, np.ndarray]]:
        """
        Slice a numpy raster (H, W) or (H, W, C) into overlapping tiles.
        """
        h, w = raster.shape[:2]
        windows = self.get_windows(h, w, geo_meta=geo_meta)
        tiles = []
        for win in windows:
            tile_data = raster[win.y_start:win.y_end, win.x_start:win.x_end]
            tiles.append((win, tile_data))
        return tiles

    def stitch_masks(
        self,
        tile_masks: List[np.ndarray],
        windows: List[TileWindow],
        full_height: int,
        full_width: int,
        num_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        Stitch tile masks back into a full-sized raster using weighted Hann blending.
        Handles either:
        - 2D integer class masks (H_t, W_t) -> blended via one-hot votes
        - 3D probability maps (C, H_t, W_t) or (H_t, W_t, C) -> accumulated and normalized
        """
        if not tile_masks or not windows:
            return np.zeros((full_height, full_width), dtype=np.uint8)

        first = tile_masks[0]
        is_prob_map = (first.ndim == 3)

        if is_prob_map:
            # Assume channels are either axis 0 or axis 2
            if first.shape[0] < first.shape[2]:  # (C, H, W)
                c = first.shape[0]
                channel_first = True
            else:  # (H, W, C)
                c = first.shape[2]
                channel_first = False

            canvas_probs = np.zeros((c, full_height, full_width), dtype=np.float32)
            canvas_weights = np.zeros((full_height, full_width), dtype=np.float32)

            for mask, win in zip(tile_masks, windows):
                h, w = win.tile_h, win.tile_w
                w_mat = compute_hann_window(h, w, self.overlap // 2, self.overlap // 2)

                m_c = mask if channel_first else np.transpose(mask, (2, 0, 1))
                weighted_m = m_c[:, :h, :w] * w_mat[np.newaxis, :, :]

                canvas_probs[:, win.y_start:win.y_end, win.x_start:win.x_end] += weighted_m
                canvas_weights[win.y_start:win.y_end, win.x_start:win.x_end] += w_mat

            canvas_weights = np.maximum(canvas_weights, 1e-5)
            normalized_probs = canvas_probs / canvas_weights[np.newaxis, :, :]
            return np.argmax(normalized_probs, axis=0).astype(np.uint8)

        else:
            # 2D class label masks
            if num_classes is None:
                max_val = max(int(np.max(m)) for m in tile_masks) if tile_masks else 0
                num_classes = max(2, max_val + 1)

            canvas_votes = np.zeros((num_classes, full_height, full_width), dtype=np.float32)
            canvas_weights = np.zeros((full_height, full_width), dtype=np.float32)

            for mask, win in zip(tile_masks, windows):
                h, w = win.tile_h, win.tile_w
                w_mat = compute_hann_window(h, w, self.overlap // 2, self.overlap // 2)

                # One-hot representation of tile
                m_sub = mask[:h, :w].astype(int)
                for k in range(num_classes):
                    vote_k = (m_sub == k).astype(np.float32) * w_mat
                    canvas_votes[k, win.y_start:win.y_end, win.x_start:win.x_end] += vote_k

                canvas_weights[win.y_start:win.y_end, win.x_start:win.x_end] += w_mat

            canvas_weights = np.maximum(canvas_weights, 1e-5)
            return np.argmax(canvas_votes, axis=0).astype(np.uint8)

    def remap_detections(
        self,
        detections_by_tile: List[List[Dict[str, Any]]],
        windows: List[TileWindow],
        full_height: int,
        full_width: int,
        geo_meta: Optional[Dict[str, Any]] = None,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Re-map tile-relative detections back into full image normalized coordinates
        and perform cross-tile Non-Maximum Suppression (NMS) to eliminate duplicate detections.
        """
        global_detections: List[Dict[str, Any]] = []

        for tile_dets, win in zip(detections_by_tile, windows):
            for det in tile_dets:
                bbox = det.get("bbox_norm")
                if not bbox:
                    continue
                # Tile-relative [ymin, xmin, ymax, xmax] (normalized 0..1 in tile)
                ymin_t, xmin_t, ymax_t, xmax_t = bbox

                # Absolute pixel coords in full image
                y_min_abs = win.y_start + ymin_t * win.tile_h
                x_min_abs = win.x_start + xmin_t * win.tile_w
                y_max_abs = win.y_start + ymax_t * win.tile_h
                x_max_abs = win.x_start + xmax_t * win.tile_w

                # Global normalized coords [0..1]
                global_bbox = [
                    round(float(y_min_abs / full_height), 4),
                    round(float(x_min_abs / full_width), 4),
                    round(float(y_max_abs / full_height), 4),
                    round(float(x_max_abs / full_width), 4)
                ]

                # Compute geographic center if georeferenced
                geo_pos = None
                if geo_meta and "west" in geo_meta and "north" in geo_meta:
                    center_x_abs = (x_min_abs + x_max_abs) / 2.0
                    center_y_abs = (y_min_abs + y_max_abs) / 2.0
                    west = geo_meta["west"]
                    east = geo_meta["east"]
                    north = geo_meta["north"]
                    south = geo_meta["south"]
                    lon = west + (center_x_abs / full_width) * (east - west)
                    lat = north - (center_y_abs / full_height) * (north - south)
                    geo_pos = {"latitude": round(lat, 6), "longitude": round(lon, 6)}

                new_det = dict(det)
                new_det["bbox_norm"] = global_bbox
                new_det["tile_id"] = win.tile_id
                if geo_pos:
                    new_det["geographic_position"] = geo_pos
                global_detections.append(new_det)

        # Non-Maximum Suppression across overlapping tiles
        if not global_detections:
            return []

        return self._apply_nms(global_detections, iou_threshold=iou_threshold)

    def _apply_nms(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """Greedy Non-Maximum Suppression on normalized bounding boxes."""
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
        keep = []

        def _iou(boxA, boxB):
            yA = max(boxA[0], boxB[0])
            xA = max(boxA[1], boxB[1])
            yB = min(boxA[2], boxB[2])
            xB = min(boxA[3], boxB[3])
            interArea = max(0.0, yB - yA) * max(0.0, xB - xA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            unionArea = boxAArea + boxBArea - interArea
            return interArea / max(1e-6, unionArea)

        while sorted_dets:
            current = sorted_dets.pop(0)
            keep.append(current)
            sorted_dets = [
                d for d in sorted_dets
                if d.get("class_name") != current.get("class_name") or
                   _iou(current["bbox_norm"], d["bbox_norm"]) < iou_threshold
            ]

        return keep


# Global sliding window tiler instance
sliding_window_tiler = SlidingWindowTiler(tile_size=512, overlap=64)
