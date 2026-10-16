import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
import cv2

from model_interfaces import (
    check_spatial_compatibility,
    detect_image_modality,
    CrossModalAnalysis
)
from agent_controller import agent_controller

DEMO_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "demo"

@pytest.fixture
def optical_sar_pair():
    p_opt = DEMO_DIR / "mumbai_cartosat2s_optical.png"
    p_sar = DEMO_DIR / "risat1_cband_radar_sar.png"
    assert p_opt.exists() and p_sar.exists()
    with open(p_opt, "rb") as f_o, open(p_sar, "rb") as f_s:
        return f_o.read(), f_s.read()

@pytest.fixture
def disparate_optical_sar_pair(optical_sar_pair):
    _, b_sar = optical_sar_pair
    p_beng = DEMO_DIR / "bengaluru_cartosat3.png"
    assert p_beng.exists()
    with open(p_beng, "rb") as f:
        return f.read(), b_sar

def test_modality_detection_accuracy(optical_sar_pair):
    b_opt, b_sar = optical_sar_pair
    mod_opt = detect_image_modality(b_opt, filename="mumbai_cartosat2s_optical.png")
    mod_sar = detect_image_modality(b_sar, filename="risat1_cband_radar_sar.png")

    assert mod_opt["modality"] == "OPTICAL"
    assert mod_sar["modality"] == "SAR"
    assert "RISAT" in mod_sar["sensor_family"] or "SAR" in mod_sar["sensor_family"]

def test_optical_sar_true_pair_passes(optical_sar_pair):
    b_opt, b_sar = optical_sar_pair
    is_compatible, score, msg, diag = check_spatial_compatibility(b_opt, b_sar, cross_modal=True)
    assert is_compatible
    assert score >= 0.70
    assert "Optical-SAR" in msg

def test_optical_sar_disparate_pair_rejected(disparate_optical_sar_pair):
    b_opt, b_sar = disparate_optical_sar_pair
    is_compatible, score, msg, diag = check_spatial_compatibility(b_opt, b_sar, cross_modal=True)
    assert not is_compatible
    assert score <= 0.35
    assert "Disparity" in msg or "Incompatible" in msg

def test_cross_modal_analysis_execution(optical_sar_pair):
    b_opt, b_sar = optical_sar_pair
    pipeline = CrossModalAnalysis()
    res = pipeline.execute([b_opt, b_sar], "Identify built-up and water bodies through clouds")

    assert res["compatibility_status"] == "PASSED"
    assert res["confidence"] > 0.40
    assert "SatQueryOpticalSARNet" in res["model_provenance"] or "fallback_heuristic" in res["model_provenance"]
    assert "geojson_data" in res and res["geojson_data"] is not None
    assert "features" in res["geojson_data"]
    assert len(res["geojson_data"]["features"]) > 0
    assert "Quantitative Multi-Modal Cross-Fused Surface Matrix" in res["text"]

def test_single_image_sar_contradiction_guardrail(optical_sar_pair):
    b_opt, _ = optical_sar_pair
    res = agent_controller.execute_query(
        query="Use SAR radar to penetrate clouds and map water",
        images=[b_opt]
    )
    assert res["status"] == "rejected"
    assert res["compatibility_status"] == "REJECTED_MODALITY_MISMATCH"
    assert "Input Modality Contradiction" in res["answer"]

def test_optical_sar_agent_controller_e2e_routing(optical_sar_pair):
    b_opt, b_sar = optical_sar_pair
    res = agent_controller.execute_query(
        query="Use optical and SAR images together to identify built-up and water-covered regions",
        images=[b_opt, b_sar],
        filenames=["mumbai_cartosat2s_optical.png", "risat1_cband_radar_sar.png"],
        benchmark_mode=True
    )
    assert res["status"] == "success"
    assert res["compatibility_status"] == "PASSED"
    assert "visual_evidence" in res and len(res["visual_evidence"]) > 0
    assert "execution_trace" in res and res["execution_trace"] is not None
    assert res.get("model_provenance") == "SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)"

def test_optical_sar_neural_provenance_metadata(optical_sar_pair):
    b_opt, b_sar = optical_sar_pair
    pipeline = CrossModalAnalysis()
    res = pipeline.execute(
        [b_opt, b_sar],
        "Identify built-up and water bodies through clouds",
        filenames=["mumbai_cartosat2s_optical.png", "risat1_cband_radar_sar.png"]
    )
    meta = res.get("inference_metadata")
    assert meta is not None
    assert meta["execution_path"] == "NEURAL"
    assert meta["model_name"] == "SatQueryOpticalSARNet"
    assert meta["model_loaded"] is True
    assert meta["forward_latency_ms"] > 0
    assert meta["fallback_reason"] is None
    assert meta["checkpoint_path"].endswith("optical_sar_net.pth")
    assert res["model_provenance"] == "SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)"

    # Verify agent controller trace propagation
    agent_res = agent_controller.execute_query(
        query="Use optical and SAR images together to identify built-up and water-covered regions",
        images=[b_opt, b_sar],
        filenames=["mumbai_cartosat2s_optical.png", "risat1_cband_radar_sar.png"],
        benchmark_mode=True
    )
    trace = agent_res.get("execution_trace")
    assert trace is not None
    fusion_tool = trace["tools_called"][1]
    assert fusion_tool["tool_id"] == "optical_sar_fusion"
    assert "inference_metadata" in fusion_tool["parameters_used"]
    assert fusion_tool["parameters_used"]["inference_metadata"]["execution_path"] == "NEURAL"
    assert "via NEURAL path" in fusion_tool["summary"]
    assert agent_res.get("model_provenance") == meta["provenance"]

def test_optical_sar_forced_fallback_provenance(optical_sar_pair, monkeypatch):
    from ml_models import ml_manager
    b_opt, b_sar = optical_sar_pair

    # Force neural pipeline to be unavailable
    monkeypatch.setattr(ml_manager, "get_cross_modal_pipeline", lambda: (None, 0.0, "fallback_heuristic (IHS Cross-Modal Fusion Engine)"))

    pipeline = CrossModalAnalysis()
    res = pipeline.execute(
        [b_opt, b_sar],
        "Identify built-up and water bodies through clouds",
        filenames=["mumbai_cartosat2s_optical.png", "risat1_cband_radar_sar.png"]
    )
    assert res["compatibility_status"] == "PASSED"
    meta = res.get("inference_metadata")
    assert meta is not None
    assert meta["execution_path"] == "HEURISTIC_FALLBACK"
    assert meta["model_loaded"] is False
    assert meta["model_name"] == "None"
    assert meta["forward_latency_ms"] == 0.0
    assert meta["fallback_reason"] is not None
    assert "unavailable" in meta["fallback_reason"].lower() or "failed" in meta["fallback_reason"].lower()
    assert res["model_provenance"] == "fallback_heuristic (IHS Cross-Modal Fusion Engine)"

    # Verify agent controller propagates fallback path in trace
    agent_res = agent_controller.execute_query(
        query="Use optical and SAR images together under degraded conditions to identify built-up and water-covered regions",
        images=[b_opt, b_sar],
        filenames=["mumbai_cartosat2s_optical.png", "risat1_cband_radar_sar.png"],
        benchmark_mode=True
    )
    assert agent_res["status"] == "success"
    assert agent_res.get("model_provenance") == "fallback_heuristic (IHS Cross-Modal Fusion Engine)"
    trace = agent_res.get("execution_trace")
    assert trace is not None
    fusion_tool = trace["tools_called"][1]
    assert fusion_tool["parameters_used"]["inference_metadata"]["execution_path"] == "HEURISTIC_FALLBACK"
    assert "via HEURISTIC_FALLBACK path" in fusion_tool["summary"]

def test_cross_modal_and_isro_evaluation_scope_metadata():
    import json
    rep_cross = Path(__file__).resolve().parent.parent / "reports" / "cross_modal.json"
    assert rep_cross.exists()
    with open(rep_cross, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["evaluation_scope"] == "SYNTHETIC_SCHEMA_PROTOTYPE"
    assert data["reference_type"] == "HEURISTIC_PSEUDO_MASK"
    assert data["model"]["name"] == "SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)"
    assert "heuristic" in data["integrity_notice"].lower()
    assert "synthetic" in data["integrity_notice"].lower()

    rep_isro = Path(__file__).resolve().parent.parent / "reports" / "isro_evaluation.json"
    assert rep_isro.exists()
    with open(rep_isro, "r", encoding="utf-8") as f:
        data_isro = json.load(f)
    assert data_isro["evaluation_scope"] == "SYNTHETIC_DEMO_STUB"
    assert data_isro["reference_masks_evaluated"] is False
    assert data_isro["qa_answers_evaluated"] is False
    assert data_isro["model_provenance"] == "SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)"
    assert data_isro["sensor_inputs"]["optical"]["is_synthetic_demo"] is True
    assert data_isro["sensor_inputs"]["sar"]["is_synthetic_demo"] is True
    assert "integrity_notice" in data_isro


def test_requirement_status_integrity():
    """Verify that reports/final_matrix.md and README.md distinguish implementation status from authentic validation status."""
    matrix_file = Path(__file__).resolve().parent.parent / "reports" / "final_matrix.md"
    assert matrix_file.exists()
    content = matrix_file.read_text(encoding="utf-8")

    # Implementation Status must be separated from Authentic-Data Validation Status
    assert "Implementation Status" in content
    assert "Authentic-Data Validation Status" in content

    # REQ-06: Distinct implementation vs validation status (pending authentic ISRO data)
    assert "REQ-06" in content
    assert "SYNTHETIC SCHEMA PROTOTYPE (Authentic Co-registered ISRO Cartosat/RISAT Evaluation Pending)" in content
    assert "24.93%" in content
    assert "75.8%" not in content

    # REQ-04 and REQ-05 must not claim unbacked 77.82% or 71.3%
    assert "77.82%" not in content
    assert "71.3%" not in content
    assert "CDVQA: 15.16% change mask mIoU" in content
    assert "VRSBench VQA: 76.67% Acc" in content



