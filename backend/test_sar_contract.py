import io
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, os.path.dirname(__file__))

from model_interfaces import (
    resolve_sar_input_contract,
    detect_image_modality,
    CrossModalAnalysis
)

DEMO_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "demo"

def create_in_memory_geotiff(
    dtype="float32",
    values=None,
    tags=None,
    descriptions=None,
    crs="EPSG:4326",
    h=64,
    w=64
) -> bytes:
    """Helper to generate in-memory GeoTIFFs with custom metadata tags."""
    mem = io.BytesIO()
    if values is None:
        if dtype == "float32":
            arr = np.linspace(-25.0, -2.0, h * w, dtype=np.float32).reshape(h, w)
        elif dtype == "uint16":
            arr = np.linspace(1000, 25000, h * w, dtype=np.uint16).reshape(h, w)
        else:
            arr = np.linspace(0, 255, h * w, dtype=np.uint8).reshape(h, w)
    else:
        arr = values.astype(dtype)

    transform = from_bounds(72.8, 18.9, 73.0, 19.1, w, h)
    with rasterio.open(
        mem,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as ds:
        ds.write(arr, 1)
        if tags:
            ds.update_tags(**tags)
        if descriptions:
            ds.descriptions = descriptions
    return mem.getvalue()

@pytest.fixture
def optical_demo_bytes():
    p_opt = DEMO_DIR / "mumbai_cartosat2s_optical.png"
    assert p_opt.exists()
    return p_opt.read_bytes()

@pytest.fixture
def sar_demo_bytes():
    p_sar = DEMO_DIR / "risat1_cband_radar_sar.png"
    assert p_sar.exists()
    return p_sar.read_bytes()

def test_calibrated_float32_input_contract(optical_demo_bytes):
    """Verify that a georeferenced float32 GeoTIFF with explicit dB metadata qualifies as CALIBRATED_SIGMA0_DB."""
    cal_bytes = create_in_memory_geotiff(
        dtype="float32",
        tags={"unit": "dB", "radiometry": "sigma0_db", "mission": "Sentinel-1"},
        descriptions=("Sigma0_VV_db",)
    )

    # 1. Direct resolver test
    contract = resolve_sar_input_contract(cal_bytes, filename="sentinel1_s0_vv.tif")
    assert contract["contract"] == "CALIBRATED_SIGMA0_DB"
    assert contract["is_scientifically_calibrated"] is True
    assert contract["input_units"] == "Decibels (dB)"
    assert contract["has_georeference"] is True

    # 2. Modality detection integration
    mod_sar = detect_image_modality(cal_bytes, filename="sentinel1_s0_vv.tif")
    assert mod_sar["modality"] == "SAR"
    assert "sar_input_contract" in mod_sar
    assert mod_sar["sar_input_contract"]["contract"] == "CALIBRATED_SIGMA0_DB"

    # 3. CrossModalAnalysis pipeline execution
    pipeline = CrossModalAnalysis()
    res = pipeline.execute([optical_demo_bytes, cal_bytes], "Identify water and built-up through clouds")
    assert res["compatibility_status"] == "PASSED"

    # Verify metadata contract propagation
    meta = res["inference_metadata"]
    assert "sar_input_contract" in meta
    assert meta["sar_input_contract"]["contract"] == "CALIBRATED_SIGMA0_DB"
    assert meta["sar_input_contract"]["is_scientifically_calibrated"] is True

    # Verify output label
    after_label = res["pair_comparison"]["after_label"]
    assert "CALIBRATED SENTINEL-1 C-BAND SAR" in after_label
    assert "RISAT" not in after_label
    assert "SURROGATE" not in after_label

def test_visual_grayscale_surrogate_contract(optical_demo_bytes, sar_demo_bytes):
    """Verify that 8-bit visual PNG images are classified as SURROGATE_VISUAL_GRAYSCALE."""
    contract = resolve_sar_input_contract(sar_demo_bytes, filename="risat1_cband_radar_sar.png")
    assert contract["contract"] == "SURROGATE_VISUAL_GRAYSCALE"
    assert contract["is_scientifically_calibrated"] is False
    assert "surrogate" in contract["calibration_notice"].lower()

    pipeline = CrossModalAnalysis()
    res = pipeline.execute([optical_demo_bytes, sar_demo_bytes], "Identify water and built-up through clouds")
    meta = res["inference_metadata"]
    assert meta["sar_input_contract"]["contract"] == "SURROGATE_VISUAL_GRAYSCALE"
    assert meta["sar_input_contract"]["is_scientifically_calibrated"] is False

    # Never label arbitrary PNG as calibrated RISAT-1 backscatter
    after_label = res["pair_comparison"]["after_label"]
    assert "SURROGATE RADAR MAPPING, UNCALIBRATED" in after_label
    assert "CALIBRATED C-BAND SAR" not in after_label

def test_uncalibrated_dn_contract(optical_demo_bytes):
    """Verify that integer GeoTIFFs (raw DN) are identified as UNCALIBRATED_DN."""
    dn_bytes = create_in_memory_geotiff(
        dtype="uint16",
        tags={"sensor": "RISAT-1", "product": "FRS-1"}
    )
    contract = resolve_sar_input_contract(dn_bytes, filename="risat1_frs_raw_dn.tif")
    assert contract["contract"] == "UNCALIBRATED_DN"
    assert contract["is_scientifically_calibrated"] is False
    assert "LUT" in contract["calibration_notice"]

    pipeline = CrossModalAnalysis()
    res = pipeline.execute([optical_demo_bytes, dn_bytes], "Identify water and built-up through clouds")
    assert res["inference_metadata"]["sar_input_contract"]["contract"] == "UNCALIBRATED_DN"
    assert "RAW SAR DIGITAL NUMBERS (UNCALIBRATED DN, SURROGATE MAPPING)" in res["pair_comparison"]["after_label"]

def test_unknown_calibration_float_raster(optical_demo_bytes):
    """Verify that a float32 GeoTIFF without verified radar calibration tags returns UNKNOWN."""
    # A float32 raster with no unit or radar backscatter metadata (e.g. elevation or heat surface)
    uncal_bytes = create_in_memory_geotiff(
        dtype="float32",
        values=np.linspace(100.0, 500.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    )
    contract = resolve_sar_input_contract(uncal_bytes, filename="unverified_float.tif")
    assert contract["contract"] == "UNKNOWN"
    assert contract["is_scientifically_calibrated"] is False
    assert "could not be verified" in contract["calibration_notice"]

    pipeline = CrossModalAnalysis()
    res = pipeline.execute([optical_demo_bytes, uncal_bytes], "Identify water and built-up")
    assert res["inference_metadata"]["sar_input_contract"]["contract"] == "UNKNOWN"
    assert "SYNTHETIC APERTURE RADAR (UNCALIBRATED / UNKNOWN RADIOMETRY)" in res["pair_comparison"]["after_label"]

def test_override_alone_rejected_without_metadata(optical_demo_bytes):
    """Verify that passing sar_calibration='sigma0_db' ALONE does NOT establish calibration without trustworthy metadata (Blocker A)."""
    raw_float_db = create_in_memory_geotiff(
        dtype="float32",
        values=np.linspace(-22.0, -5.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    )
    # Attempting override on uncalibrated float raster must fail closed to UNKNOWN
    c_override = resolve_sar_input_contract(raw_float_db, filename="radar.tif", explicit_calibration="sigma0_db")
    assert c_override["contract"] == "UNKNOWN"
    assert c_override["is_scientifically_calibrated"] is False
    assert "rejected" in c_override["details"].lower()

    pipeline = CrossModalAnalysis()
    res = pipeline.execute([optical_demo_bytes, raw_float_db], "Identify features", sar_calibration="sigma0_db")
    assert res["inference_metadata"]["sar_input_contract"]["contract"] == "UNKNOWN"
    assert res["inference_metadata"]["sar_input_contract"]["is_scientifically_calibrated"] is False
    assert "SYNTHETIC APERTURE RADAR (UNCALIBRATED / UNKNOWN RADIOMETRY)" in res["pair_comparison"]["after_label"]

def test_misleading_substring_metadata_rejected():
    """Verify that generic substrings like 'project_db' or 'db_version' do NOT establish calibration (Blocker A)."""
    misleading_bytes = create_in_memory_geotiff(
        dtype="float32",
        values=np.linspace(-20.0, -5.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        descriptions=("project_db_version_2",)
    )
    contract = resolve_sar_input_contract(misleading_bytes, filename="project_db.tif")
    assert contract["contract"] == "UNKNOWN"
    assert contract["is_scientifically_calibrated"] is False
    assert "lacks verified" in contract["details"].lower() or "lacking" in contract["calibration_notice"].lower()

def test_contradictory_units_rejected(optical_demo_bytes):
    """Verify that contradictory metadata (e.g. unit: meters with backscatter tags) fails closed to UNKNOWN (Blocker A)."""
    contra_bytes = create_in_memory_geotiff(
        dtype="float32",
        values=np.linspace(-20.0, -5.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        tags={"unit": "meters", "radiometry": "sigma0"},
        descriptions=("sigma0_vv",)
    )
    # Even if client attempts override, contradiction must cause failure
    contract = resolve_sar_input_contract(contra_bytes, filename="contradictory.tif", explicit_calibration="sigma0_db")
    assert contract["contract"] == "UNKNOWN"
    assert contract["is_scientifically_calibrated"] is False
    assert "contradictory" in contract["details"].lower()

def test_generic_calibrated_sar_does_not_assert_risat(optical_demo_bytes):
    """Verify that generic calibrated SAR does NOT assert RISAT-1 identity (Blocker B)."""
    generic_cal_bytes = create_in_memory_geotiff(
        dtype="float32",
        values=np.linspace(-25.0, -3.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        tags={"unit": "dB", "radiometry": "sigma0"}
    )
    contract = resolve_sar_input_contract(generic_cal_bytes, filename="calibrated_patch.tif")
    assert contract["contract"] == "CALIBRATED_SIGMA0_DB"
    assert contract["is_scientifically_calibrated"] is True
    assert contract["sensor_identity"] == "UNVERIFIED"
    assert "CALIBRATED SAR — SENSOR UNVERIFIED" in contract["sensor_family"]
    assert "RISAT" not in contract["sensor_family"]
    assert "Sentinel" not in contract["sensor_family"]

    mod = detect_image_modality(generic_cal_bytes, filename="calibrated_patch.tif")
    assert "CALIBRATED SAR — SENSOR UNVERIFIED" in mod["sensor_family"]
    assert "RISAT" not in mod["sensor_family"]

    pipeline = CrossModalAnalysis()
    res = pipeline.execute([optical_demo_bytes, generic_cal_bytes], "Identify water and built-up")
    after_label = res["pair_comparison"]["after_label"]
    assert "CALIBRATED SAR — SENSOR UNVERIFIED" in after_label
    assert "RISAT" not in after_label

def test_verified_sentinel1_calibration(optical_demo_bytes):
    """Verify that authentic Sentinel-1 metadata correctly identifies Sentinel-1 C-Band SAR (Blocker B)."""
    s1_bytes = create_in_memory_geotiff(
        dtype="float32",
        values=np.linspace(-24.0, -4.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        tags={"unit": "dB", "radiometry": "sigma0", "mission": "Sentinel-1"}
    )
    contract = resolve_sar_input_contract(s1_bytes, filename="s1_iw_grd.tif")
    assert contract["contract"] == "CALIBRATED_SIGMA0_DB"
    assert contract["is_scientifically_calibrated"] is True
    assert contract["sensor_identity"] == "SENTINEL-1"
    assert contract["sensor_family"] == "Sentinel-1 C-Band SAR"

    pipeline = CrossModalAnalysis()
    res = pipeline.execute([optical_demo_bytes, s1_bytes], "Identify water and built-up")
    assert "CALIBRATED SENTINEL-1 C-BAND SAR" in res["pair_comparison"]["after_label"]
    assert "RISAT" not in res["pair_comparison"]["after_label"]

def test_verified_risat1_calibration(optical_demo_bytes):
    """Verify that authentic RISAT-1 metadata correctly identifies RISAT-1 C-Band SAR (Blocker B)."""
    risat_bytes = create_in_memory_geotiff(
        dtype="float32",
        values=np.linspace(-23.0, -3.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        tags={"unit": "dB", "radiometry": "sigma0", "sensor": "RISAT-1"}
    )
    contract = resolve_sar_input_contract(risat_bytes, filename="risat1_mrs_calibrated.tif")
    assert contract["contract"] == "CALIBRATED_SIGMA0_DB"
    assert contract["is_scientifically_calibrated"] is True
    assert contract["sensor_identity"] == "RISAT-1"
    assert contract["sensor_family"] == "RISAT-1 C-Band SAR"

    pipeline = CrossModalAnalysis()
    res = pipeline.execute([optical_demo_bytes, risat_bytes], "Identify water and built-up")
    assert "CALIBRATED RISAT-1 C-BAND SAR" in res["pair_comparison"]["after_label"]

def test_preprocessing_consistency_and_finite_outputs(optical_demo_bytes):
    """Verify that both calibrated float32 and surrogate grayscale produce finite masks and non-empty GeoJSON."""
    pipeline = CrossModalAnalysis()

    # Case A: Calibrated Float32
    cal_bytes = create_in_memory_geotiff(
        dtype="float32",
        values=np.linspace(-25.0, -1.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        tags={"unit": "dB", "radiometry": "sigma0_db"}
    )
    res_cal = pipeline.execute([optical_demo_bytes, cal_bytes], "Map features")
    assert res_cal["compatibility_status"] == "PASSED"
    assert res_cal["geojson_data"] is not None
    assert len(res_cal["geojson_data"]["features"]) > 0
    assert 0.0 <= res_cal["confidence"] <= 1.0

    # Case B: Surrogate 8-bit Visual
    p_sar = DEMO_DIR / "risat1_cband_radar_sar.png"
    res_surr = pipeline.execute([optical_demo_bytes, p_sar.read_bytes()], "Map features")
    assert res_surr["compatibility_status"] == "PASSED"
    assert res_surr["geojson_data"] is not None
    assert len(res_surr["geojson_data"]["features"]) > 0
    assert 0.0 <= res_surr["confidence"] <= 1.0
