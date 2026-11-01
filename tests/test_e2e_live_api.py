"""
tests/test_e2e_live_api.py

End-to-End Live API Integration Tests for SatQuery AI.
Tests real HTTP request/response lifecycles, middlewares, and ML pipelines
using authentic demo payloads from frontend/public/demo/.
"""

import sys
import os
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DEMO_DIR = ROOT_DIR / "frontend" / "public" / "demo"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_e2e_health_check():
    """Verify system health, loaded models, and runtime device telemetry."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "models" in data
    assert data["models"]["segmentation"]["status"] == "ready"
    assert data["models"]["change_detection"]["status"] == "ready"
    assert data["models"]["sar_fusion"]["status"] == "ready"

def test_e2e_reports_discovery():
    """Verify /api/reports endpoint discovers verified benchmark reports."""
    response = client.get("/api/reports")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, (list, dict))

def test_e2e_preflight_analysis():
    """Verify /api/preflight telemetry on real bi-temporal demo scenes."""
    t1_path = DEMO_DIR / "uttarakhand_pre_event_t1.png"
    t2_path = DEMO_DIR / "uttarakhand_post_event_t2.png"
    assert t1_path.exists() and t2_path.exists()

    with open(t1_path, "rb") as f1, open(t2_path, "rb") as f2:
        files = [
            ("images", ("t1.png", f1.read(), "image/png")),
            ("images", ("t2.png", f2.read(), "image/png"))
        ]
        response = client.post("/api/preflight", files=files, data={"benchmark_mode": "true"})
        assert response.status_code == 200
        data = response.json()
        assert data["num_images"] == 2
        assert data["preflight_status"] == "READY"
        assert data["pairwise_compatibility"]["status"] == "PASSED"

def test_e2e_change_detection_pipeline():
    """Verify end-to-end bi-temporal change detection execution."""
    t1_path = DEMO_DIR / "uttarakhand_pre_event_t1.png"
    t2_path = DEMO_DIR / "uttarakhand_post_event_t2.png"

    with open(t1_path, "rb") as f1, open(t2_path, "rb") as f2:
        files = [
            ("images", ("t1.png", f1.read(), "image/png")),
            ("images", ("t2.png", f2.read(), "image/png"))
        ]
        response = client.post(
            "/api/query",
            files=files,
            data={
                "query": "Detect all flood changes and surface water expansion between T1 and T2.",
                "benchmark_mode": "true"
            }
        )
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        assert "answer" in res and len(res["answer"]) > 50
        assert "visual_evidence" in res
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0

        # Verify result retrieval by ID
        result_id = res.get("result_id")
        if result_id:
            get_res = client.get(f"/api/results/{result_id}")
            assert get_res.status_code == 200

def test_e2e_cross_modal_pipeline():
    """Verify end-to-end Optical + SAR cross-modal fusion execution."""
    opt_path = DEMO_DIR / "mumbai_cartosat2s_optical.png"
    sar_path = DEMO_DIR / "risat1_cband_radar_sar.png"

    with open(opt_path, "rb") as f1, open(sar_path, "rb") as f2:
        files = [
            ("images", ("mumbai_opt.png", f1.read(), "image/png")),
            ("images", ("risat_sar.png", f2.read(), "image/png"))
        ]
        response = client.post(
            "/api/query",
            files=files,
            data={
                "query": "Perform joint optical-SAR cross-modal fusion to penetrate cloud cover and map water bodies.",
                "benchmark_mode": "true"
            }
        )
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        assert "answer" in res and len(res["answer"]) > 50
        assert "visual_evidence" in res
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0

def test_e2e_single_image_segmentation():
    """Verify single-image land cover classification on Cartosat optical scene."""
    opt_path = DEMO_DIR / "bengaluru_cartosat3.png"

    with open(opt_path, "rb") as f:
        files = [
            ("images", ("bengaluru.png", f.read(), "image/png"))
        ]
        response = client.post(
            "/api/query",
            files=files,
            data={
                "query": "Segment the land cover distribution and calculate vegetation and urban area in hectares.",
                "benchmark_mode": "true"
            }
        )
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        assert "answer" in res and len(res["answer"]) > 50
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0

def test_e2e_out_of_domain_guardrail():
    """Verify out-of-domain query is cleanly rejected by semantic guardrail."""
    opt_path = DEMO_DIR / "bengaluru_cartosat3.png"

    with open(opt_path, "rb") as f:
        files = [
            ("images", ("bengaluru.png", f.read(), "image/png"))
        ]
        response = client.post(
            "/api/query",
            files=files,
            data={
                "query": "What is the stock price of Apple?",
                "benchmark_mode": "true"
            }
        )
        assert response.status_code == 200
        res = response.json()
        assert res["status"] in ["rejected", "REJECTED_OUT_OF_DOMAIN", "FAILED"]
        assert "non_geospatial" in res.get("answer", "").lower() or "outside remote sensing" in res.get("answer", "").lower()
