"""
SatQuery AI — OWL-ViT Visual Grounding Integration Tests
Verifies offline loading, coordinate validity, empty detections, threshold handling,
model provenance, latency recording, and heuristic fallback behavior.
"""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR))

import pytest
import numpy as np
from PIL import Image
import io
import time

from owlvit_grounding import OwlViTGroundingEngine, find_owlvit_snapshot_path
from model_interfaces import SingleImageGrounding


def create_synthetic_test_image(width: int = 512, height: int = 512) -> bytes:
    """Creates a deterministic RGB image encoded as PNG bytes."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Background: dark oceanic water
    arr[:, :] = [18, 32, 48]
    # Draw a simulated harbor dock / quay (bright structure)
    arr[150:280, 100:350] = [180, 190, 205]
    # Draw a simulated vessel (elongated bright shape)
    arr[320:420, 200:260] = [230, 240, 255]
    
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestOwlViTOfflineLoading:
    """Verifies that OWL-ViT loads exclusively from the offline local cache."""

    def test_local_snapshot_exists(self):
        """Verify the cached snapshot directory exists locally without network."""
        snap = find_owlvit_snapshot_path()
        assert snap is not None, "OWL-ViT snapshot directory not found in local cache"
        assert (snap / "model.safetensors").exists(), "model.safetensors missing from snapshot"
        assert (snap / "config.json").exists(), "config.json missing from snapshot"
        assert (snap / "preprocessor_config.json").exists(), "preprocessor_config.json missing from snapshot"

    def test_engine_availability(self):
        """Verify is_available() reports True for existing local weights."""
        engine = OwlViTGroundingEngine()
        assert engine.is_available() is True

    def test_engine_load_offline(self):
        """Verify model loads in offline mode with local_files_only."""
        engine = OwlViTGroundingEngine()
        loaded = engine.load_model()
        assert loaded is True
        assert engine._is_loaded is True
        assert engine._model is not None
        assert engine._processor is not None


class TestOwlViTCoordinateAndDetectionValidity:
    """Verifies bounding-box clamping, scaling, and GeoJSON validity."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.engine = OwlViTGroundingEngine()
        self.engine.load_model()
        self.img_bytes = create_synthetic_test_image(512, 512)

    def test_coordinate_validity_and_bounds(self):
        """Verify all predicted bounding boxes lie strictly within image boundaries."""
        res = self.engine.ground_expression(
            image=self.img_bytes,
            query="docking berth or harbor structure",
            threshold=0.03
        )
        assert res["status"] in ("SUCCESS", "NO_DETECTION")
        assert "bounding_boxes" in res
        
        w, h = 512, 512
        for box in res["bounding_boxes"]:
            bx, by, bw, bh = box
            assert 0 <= bx < w, f"bx out of bounds: {bx}"
            assert 0 <= by < h, f"by out of bounds: {by}"
            assert bw > 0, f"bw must be positive: {bw}"
            assert bh > 0, f"bh must be positive: {bh}"
            assert bx + bw <= w, f"Box exceeds width: {bx + bw} > {w}"
            assert by + bh <= h, f"Box exceeds height: {by + bh} > {h}"

    def test_geojson_feature_structure(self):
        """Verify GeoJSON FeatureCollection format and coordinate reference system."""
        res = self.engine.ground_expression(
            image=self.img_bytes,
            query="ship",
            threshold=0.03
        )
        geojson = res.get("geojson_data")
        assert geojson is not None
        assert geojson["type"] == "FeatureCollection"
        assert geojson["metadata"]["execution_path"] == "NEURAL_OWLVIT"
        assert geojson["metadata"]["model"] == "google/owlvit-base-patch32"
        
        for feature in geojson["features"]:
            assert feature["type"] == "Feature"
            assert feature["geometry"]["type"] == "Polygon"
            coords = feature["geometry"]["coordinates"][0]
            assert len(coords) == 5  # Closed ring
            assert coords[0] == coords[-1], "GeoJSON polygon ring must be closed"
            for lon, lat in coords:
                assert -180.0 <= lon <= 180.0
                assert -90.0 <= lat <= 90.0


class TestOwlViTThresholdAndEmptyDetection:
    """Verifies threshold filtering and explicit no-detection behavior."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.engine = OwlViTGroundingEngine()
        self.engine.load_model()
        self.img_bytes = create_synthetic_test_image(256, 256)

    def test_explicit_no_detection_when_threshold_unmet(self):
        """Verify high threshold returns explicit NO_DETECTION result."""
        res = self.engine.ground_expression(
            image=self.img_bytes,
            query="airport runway",
            threshold=0.999  # Impossibly high threshold
        )
        assert res["status"] == "NO_DETECTION"
        assert res["has_detection"] is False
        assert res["confidence"] == 0.0
        assert len(res["bounding_boxes"]) == 0
        assert len(res["detections"]) == 0
        assert len(res["geojson_data"]["features"]) == 0
        assert res["inference_metadata"]["execution_path"] == "NEURAL_OWLVIT"
        assert res["inference_metadata"]["detection_count"] == 0
        assert "No object candidate exceeded" in res["text"]


class TestSingleImageGroundingIntegration:
    """Verifies SingleImageGrounding specialist dispatch, fallback, and provenance."""

    def test_single_image_grounding_neural_dispatch(self):
        """Verify default SingleImageGrounding execution routes to neural OWL-ViT."""
        grounding_model = SingleImageGrounding()
        img_bytes = create_synthetic_test_image(256, 256)
        
        res = grounding_model.execute([img_bytes], "locate the harbor dock", threshold=0.04)
        
        assert "inference_metadata" in res
        meta = res["inference_metadata"]
        assert meta["execution_path"] == "NEURAL_OWLVIT"
        assert meta["model_loaded"] is True
        assert meta["forward_latency_ms"] > 0.0
        assert "Google OWL-ViT" in res["model_provenance"]
        assert res["compatibility_status"] == "PASSED"

    def test_single_image_grounding_force_heuristic_fallback(self):
        """Verify force_heuristic=True preserves the heuristic contour delineator."""
        grounding_model = SingleImageGrounding()
        img_bytes = create_synthetic_test_image(256, 256)
        
        res = grounding_model.execute([img_bytes], "water bodies and terrestrial land", force_heuristic=True)
        
        assert "inference_metadata" in res
        meta = res["inference_metadata"]
        assert meta["execution_path"] == "HEURISTIC_FALLBACK"
        assert meta["model_loaded"] is False
        assert "fallback_heuristic" in res["model_provenance"]
        assert "Explicit heuristic" in meta["fallback_reason"]

    def test_single_image_grounding_mode_heuristic_parameter(self):
        """Verify mode='heuristic' explicitly runs heuristic path."""
        grounding_model = SingleImageGrounding()
        img_bytes = create_synthetic_test_image(256, 256)
        
        res = grounding_model.execute([img_bytes], "port berths", mode="heuristic")
        
        assert res["inference_metadata"]["execution_path"] == "HEURISTIC_FALLBACK"
        assert "fallback_heuristic" in res["model_provenance"]

    def test_agent_controller_grounding_neural_routing(self):
        """Verify agent_controller.execute_query routes locate queries to OWL-ViT."""
        from agent_controller import agent_controller
        img_bytes = create_synthetic_test_image(256, 256)
        res = agent_controller.execute_query(
            query="Locate the harbor dock",
            images=[img_bytes],
            parameters={"threshold": 0.04}
        )
        assert res["status"] == "success"
        assert res["execution_trace"]["selected_task"] == "SINGLE_IMAGE_GROUNDING"
        assert res["inference_metadata"]["execution_path"] == "NEURAL_OWLVIT"
        assert "Google OWL-ViT" in res["model_provenance"]

    def test_agent_controller_grounding_heuristic_routing(self):
        """Verify agent_controller.execute_query supports force_heuristic parameter."""
        from agent_controller import agent_controller
        img_bytes = create_synthetic_test_image(256, 256)
        res = agent_controller.execute_query(
            query="Locate water bodies and terrestrial land",
            images=[img_bytes],
            parameters={"force_heuristic": True}
        )
        assert res["status"] == "success"
        assert res["execution_trace"]["selected_task"] == "SINGLE_IMAGE_GROUNDING"
        assert res["inference_metadata"]["execution_path"] == "HEURISTIC_FALLBACK"
        assert "fallback_heuristic" in res["model_provenance"]
