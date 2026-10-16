"""
backend/test_input_robustness.py
Unit tests for Phase 7: Input Handling, GeoTIFF Ingestion, Sliding Window Tiling,
Nodata handling, and Robustness Guardrails.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import io
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

from model_interfaces import decode_satellite_image, detect_image_format
from tiling import sliding_window_tiler, SlidingWindowTiler, TileWindow
from tool_registry import tool_registry, StructuredErrorCode


def _create_synthetic_geotiff(
    dtype=np.uint16,
    bands=3,
    height=256,
    width=256,
    nodata=None,
    fill_nodata_region=False,
    all_empty=False
) -> bytes:
    """Helper to generate in-memory GeoTIFF with real CRS and bounding box."""
    transform = from_bounds(72.8, 18.9, 73.0, 19.1, width, height)
    memfile = rasterio.MemoryFile()

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata
    }

    with memfile.open(**profile) as dst:
        for i in range(1, bands + 1):
            if all_empty:
                arr = np.zeros((height, width), dtype=dtype)
            elif dtype == np.uint16:
                arr = np.random.RandomState(42 + i).randint(500, 8000, size=(height, width), dtype=np.uint16)
            elif dtype == np.float32:
                # Realistic linear SAR backscatter values (0.001 to 2.5)
                arr = np.random.RandomState(42 + i).gamma(shape=2.0, scale=0.1, size=(height, width)).astype(np.float32)
            else:
                arr = np.random.RandomState(42 + i).randint(0, 255, size=(height, width), dtype=np.uint8)

            if fill_nodata_region and nodata is not None:
                # Put nodata values in top-left 50x50 corner
                arr[:50, :50] = nodata

            dst.write(arr, i)

    return memfile.read()


def test_16bit_geotiff_ingestion():
    """Verify 16-bit uint16 GeoTIFF decodes with georeferencing and contrast stretch."""
    tiff_bytes = _create_synthetic_geotiff(dtype=np.uint16, bands=3, height=256, width=256)
    cv_img, pil_img, geo_meta = decode_satellite_image(tiff_bytes)

    assert cv_img is not None
    assert cv_img.shape == (256, 256, 3)
    assert cv_img.dtype == np.uint8
    assert pil_img is not None
    assert geo_meta is not None
    assert geo_meta["crs"] == "EPSG:4326"
    assert "west" in geo_meta and "north" in geo_meta
    assert geo_meta["dtype"] == "uint16"


def test_float32_sar_geotiff_ingestion():
    """Verify float32 SAR backscatter GeoTIFF is converted to decibels and stretched to uint8."""
    tiff_bytes = _create_synthetic_geotiff(dtype=np.float32, bands=1, height=128, width=128)
    cv_img, pil_img, geo_meta = decode_satellite_image(tiff_bytes)

    assert cv_img is not None
    assert cv_img.shape == (128, 128, 3)
    assert cv_img.dtype == np.uint8
    assert geo_meta is not None
    assert geo_meta["band_count"] == 1
    assert geo_meta["dtype"] == "float32"


def test_multi_band_4band_geotiff():
    """Verify 4-band GeoTIFF (RGB + NIR) selects valid visual channels."""
    tiff_bytes = _create_synthetic_geotiff(dtype=np.uint8, bands=4, height=128, width=128)
    cv_img, pil_img, geo_meta = decode_satellite_image(tiff_bytes)

    assert cv_img is not None
    assert cv_img.shape == (128, 128, 3)
    assert geo_meta is not None
    assert geo_meta["band_count"] == 4


def test_nodata_handling():
    """Verify nodata values are masked out and do not distort valid pixels."""
    tiff_bytes = _create_synthetic_geotiff(
        dtype=np.uint16,
        bands=3,
        height=128,
        width=128,
        nodata=0,
        fill_nodata_region=True
    )
    cv_img, pil_img, geo_meta = decode_satellite_image(tiff_bytes)

    assert cv_img is not None
    # The nodata region (:50, :50) should be cleanly set to 0 (black background)
    assert np.all(cv_img[:50, :50] == 0)
    # The valid region should have non-zero variance
    assert np.std(cv_img[50:, 50:]) > 5.0
    assert geo_meta["nodata"] == 0.0


def test_empty_blank_geotiff_rejection():
    """Verify uniform blank / all-zero GeoTIFF raises ValueError."""
    tiff_bytes = _create_synthetic_geotiff(dtype=np.uint16, bands=1, all_empty=True)
    with pytest.raises(ValueError) as exc:
        decode_satellite_image(tiff_bytes)
    assert "blank" in str(exc.value) or "empty" in str(exc.value)


def test_corrupt_bytes_rejection():
    """Verify corrupted byte stream raises ValueError."""
    corrupt_bytes = b"NOT_A_VALID_TIFF_STREAM" * 10
    with pytest.raises(ValueError) as exc:
        decode_satellite_image(corrupt_bytes)
    assert "corrupted" in str(exc.value).lower()


def test_sliding_window_tiling_coverage():
    """Verify sliding window tiler completely covers arbitrary image dimensions without gaps."""
    tiler = SlidingWindowTiler(tile_size=256, overlap=32)
    windows = tiler.get_windows(height=600, width=800)

    assert len(windows) > 1
    # Verify bounds coverage
    min_y = min(w.y_start for w in windows)
    max_y = max(w.y_end for w in windows)
    min_x = min(w.x_start for w in windows)
    max_x = max(w.x_end for w in windows)

    assert min_y == 0
    assert max_y == 600
    assert min_x == 0
    assert max_x == 800


def test_sliding_window_stitching_accuracy():
    """Verify stitching reconstructs exact original mask dimensions."""
    tiler = SlidingWindowTiler(tile_size=128, overlap=32)
    full_h, full_w = 300, 400
    windows = tiler.get_windows(full_h, full_w)

    # Generate dummy class masks
    tile_masks = []
    for win in windows:
        # Fill tile with class 1
        m = np.ones((win.tile_h, win.tile_w), dtype=np.uint8)
        tile_masks.append(m)

    stitched = tiler.stitch_masks(tile_masks, windows, full_h, full_w, num_classes=2)
    assert stitched.shape == (full_h, full_w)
    assert np.all(stitched == 1)


def test_sliding_window_detection_remapping_and_nms():
    """Verify tile detections are remapped to global coordinates and overlapping duplicates suppressed."""
    tiler = SlidingWindowTiler(tile_size=256, overlap=64)
    full_h, full_w = 400, 400
    windows = tiler.get_windows(full_h, full_w)

    # Place identical object in overlap zone of tile 0 and tile 1
    # Global object at y=200..240, x=200..240
    conf_primary = round(9.0 / 10.0, 2)
    conf_secondary = round(8.5 / 10.0, 2)
    # In Tile 0 (y: 0..256, x: 0..256):
    det0 = [{
        "class_name": "building",
        "confidence": conf_primary,
        "bbox_norm": [200 / 256.0, 200 / 256.0, 240 / 256.0, 240 / 256.0]
    }]
    # In Tile 1 (y: 0..256, x: 144..400):
    det1 = [{
        "class_name": "building",
        "confidence": conf_secondary,
        "bbox_norm": [200 / 256.0, (200 - 144) / 256.0, 240 / 256.0, (240 - 144) / 256.0]
    }]

    tile_dets = [det0, det1] + [[] for _ in range(len(windows) - 2)]
    remapped = tiler.remap_detections(tile_dets, windows, full_h, full_w, iou_threshold=0.3)

    # Cross-tile NMS should keep the higher-confidence detection and suppress duplicate
    assert len(remapped) == 1
    assert remapped[0]["confidence"] == conf_primary
    assert len(remapped[0]["bbox_norm"]) == 4


def test_benchmark_mode_format_enforcement():
    """Verify non-benchmark mode rejects unreferenced PNG/JPEG, and benchmark mode accepts."""
    # When benchmark_mode is False, standard PNG must be rejected
    err = tool_registry.validate_inputs(
        tool_id="single_image_vqa",
        num_images=1,
        detected_modalities=["OPTICAL"],
        detected_formats=["PNG"],
        benchmark_mode=False
    )
    assert err is not None
    assert err.error_code == StructuredErrorCode.UNSUPPORTED_FORMAT

    # When benchmark_mode is True, PNG is accepted
    err_ok = tool_registry.validate_inputs(
        tool_id="single_image_vqa",
        num_images=1,
        detected_modalities=["OPTICAL"],
        detected_formats=["PNG"],
        benchmark_mode=True
    )
    assert err_ok is None
