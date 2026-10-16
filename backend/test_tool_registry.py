import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
from PIL import Image
import io

from tool_registry import tool_registry, StructuredErrorCode, StructuredError
from agent_controller import agent_controller
from model_interfaces import detect_image_format


def _create_dummy_png():
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_registry_has_all_tools():
    """Verify all 8 standard specialist tools are registered."""
    tools = tool_registry.list_tools()
    assert len(tools) >= 8
    expected_tools = [
        "single_image_vqa",
        "single_image_captioning",
        "single_image_grounding",
        "land_cover_segmentation",
        "bi_temporal_change",
        "optical_sar_fusion",
        "spatial_compatibility_verifier",
        "heuristic_fallback"
    ]
    for tid in expected_tools:
        assert tool_registry.get_tool(tid) is not None, f"Missing tool: {tid}"


def test_format_detection():
    """Verify detect_image_format correctly labels PNG and TIFF."""
    png_bytes = _create_dummy_png()
    fmt = detect_image_format(png_bytes, "test.png")
    assert fmt == "PNG"

    # TIFF magic bytes II* (0x49 0x49 0x2A 0x00)
    tiff_bytes = b"\x49\x49\x2A\x00" + b"\x00" * 30
    assert detect_image_format(tiff_bytes, "test.tif") in ["TIFF", "GEOTIFF"]


def test_input_validation_count():
    """Test validation errors for invalid image counts."""
    # 0 images for single_image_vqa
    err = tool_registry.validate_inputs("single_image_vqa", num_images=0, detected_modalities=[], detected_formats=[])
    assert err is not None
    assert err.error_code == StructuredErrorCode.INVALID_INPUT_COUNT

    # 1 image for bi_temporal_change (requires 2)
    err = tool_registry.validate_inputs("bi_temporal_change", num_images=1, detected_modalities=["OPTICAL"], detected_formats=["PNG"])
    assert err is not None
    assert err.error_code == StructuredErrorCode.INVALID_INPUT_COUNT


def test_input_validation_modality():
    """Test modality mismatch detection."""
    # optical_sar_fusion requires OPTICAL and SAR
    err = tool_registry.validate_inputs(
        "optical_sar_fusion",
        num_images=2,
        detected_modalities=["OPTICAL", "OPTICAL"],
        detected_formats=["PNG", "PNG"]
    )
    assert err is not None
    assert err.error_code == StructuredErrorCode.MODALITY_MISMATCH


def test_parameter_validation():
    """Test parameter bounds and type checking."""
    # iou_threshold must be between 0.0 and 1.0
    err = tool_registry.validate_parameters("single_image_grounding", {"iou_threshold": 1.5})
    assert err is not None
    assert err.error_code == StructuredErrorCode.INVALID_PARAMETER

    # Valid parameter
    err_ok = tool_registry.validate_parameters("single_image_grounding", {"target_expression": "bridge", "iou_threshold": 0.5})
    assert err_ok is None


def test_controller_trace_generation():
    """Test that agent_controller returns a valid execution trace."""
    agent_controller.reset_routing_metrics()
    png_bytes = _create_dummy_png()
    
    res = agent_controller.execute_query(
        query="What type of terrain is shown in this satellite image?",
        images=[png_bytes]
    )
    
    assert res is not None
    assert "execution_trace" in res
    trace = res["execution_trace"]
    assert trace["selected_task"] in ["SINGLE_IMAGE_VQA", "SINGLE_IMAGE_CAPTIONING", "LAND_COVER_SEGMENTATION"]
    assert "tools_called" in trace
    assert len(trace["tools_called"]) >= 1
    assert "valid_sequence" in trace
    assert trace["valid_sequence"] is True


def test_controller_invalid_input_rejection():
    """Test that agent_controller cleanly rejects invalid inputs with structured error."""
    agent_controller.reset_routing_metrics()
    png_bytes = _create_dummy_png()
    
    # Passing 1 image for change detection query
    res = agent_controller.execute_query(
        query="Detect all urban changes between T1 and T2",
        images=[png_bytes]
    )
    assert res is not None
    assert res.get("status") in ["error", "rejected"]
    assert "execution_trace" in res
    trace = res["execution_trace"]
    assert trace["is_invalid_call"] is True
    assert trace["structured_error"] is not None
    assert trace["structured_error"]["error_code"] == "INVALID_INPUT_COUNT"

    metrics = agent_controller.get_routing_metrics()
    assert metrics["invalid_calls_count"] >= 1


def test_optical_sar_tool_provenance_metadata():
    """Verify tool_registry records the accurate prototype provenance label."""
    tool = tool_registry.get_tool("optical_sar_fusion")
    assert tool is not None
    assert tool.provenance_label == "SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)"
