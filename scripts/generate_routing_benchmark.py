"""
scripts/generate_routing_benchmark.py
Generates tests/routing_cases.jsonl containing >= 150 labelled test cases for
SatQuery AI's agentic router, covering:
- Single-image VQA (RSVQA / VRSBench style)
- Single-image Captioning
- Single-image Grounding / Localization
- Land-Cover Classification & Surface Area Quantification
- Bi-Temporal Change Detection (CDVQA style)
- Optical-SAR Cross-Modal Joint Analysis (BigEarthNet / ISRO style)
- Adversarial edge cases (OOD trivia, input count mismatch, modality mismatch,
  unsupported format in non-benchmark mode, parameter out of bounds)
"""
import os
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT_DIR / "tests"
TESTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = TESTS_DIR / "routing_cases.jsonl"

CASES = []

def add_case(cid, query, num_images, modalities, formats, has_geo, benchmark_mode, params, exp_task, exp_tool, validity, err_code, category):
    CASES.append({
        "id": cid,
        "query": query,
        "num_images": num_images,
        "modalities": modalities,
        "formats": formats,
        "has_georeferencing": has_geo,
        "benchmark_mode": benchmark_mode,
        "parameters": params,
        "expected_task": exp_task,
        "expected_tool": exp_tool,
        "expected_validity": validity,
        "expected_error_code": err_code,
        "category": category
    })

# ─────────────────────────────────────────────────────────────────────────────
# 1. Single Image VQA (30 cases)
# ─────────────────────────────────────────────────────────────────────────────
vqa_queries = [
    "Is there an airport runway visible in this scene?",
    "Are there any water bodies present in this image?",
    "How many storage tanks can be identified in this industrial area?",
    "Is there a bridge crossing the river?",
    "How many aircraft are stationed on the tarmac?",
    "Does the image contain residential buildings?",
    "What is the predominant land cover in the southern quadrant?",
    "Are there commercial vessels docked at the harbor?",
    "Is any railway infrastructure visible?",
    "Are there solar photovoltaic panels installed in this area?",
    "How many distinct building complexes can be seen?",
    "Is there active agricultural cultivation in these fields?",
    "Are there shipping containers stacked in the depot?",
    "Does this satellite tile show dense forest canopy?",
    "Is there an electrical transmission substation present?",
    "Are any vehicles visible on the primary highway?",
    "Is there a coastline or shoreline visible?",
    "How many industrial sheds are located in this sector?",
    "Is there evidence of terrain erosion or quarry excavation?",
    "Are there circular pivot irrigation fields in this scene?",
    "What type of vegetation dominates this scene?",
    "Is there a major highway interchange in the image?",
    "Are there boats or barges anchored in the waterway?",
    "Does this image depict an urban downtown or rural countryside?",
    "Is there a sports stadium or recreational field visible?",
    "How many water reservoirs are visible in this rural zone?",
    "Is there a construction site with earthmoving activity?",
    "Are there communication towers or tall masts visible?",
    "Is the waterway an open river or an engineered canal?",
    "Are there greenhouses or covered agricultural tunnels?"
]
for i, q in enumerate(vqa_queries, 1):
    add_case(
        cid=f"vqa_{i:03d}",
        query=q,
        num_images=1,
        modalities=["OPTICAL"],
        formats=["PNG"],
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="SINGLE_IMAGE_VQA",
        exp_tool="single_image_vqa",
        validity="VALID",
        err_code=None,
        category="single_image_vqa"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 2. Single Image Captioning (20 cases)
# ─────────────────────────────────────────────────────────────────────────────
caption_queries = [
    "Describe this satellite image in detail.",
    "Provide a comprehensive scene description of this observation.",
    "Summarize the key terrain and infrastructure features visible.",
    "What can you see in this remote sensing scene?",
    "Give an overview of the geographic context and land use.",
    "Describe the landscape and settlements visible in this tile.",
    "Provide a detailed caption describing this aerial observation.",
    "What is visible in this satellite imagery?",
    "Describe the coastline, waterways, and urban development.",
    "Provide a structured overview of the terrain and human settlements.",
    "Explain the distribution of natural and built features in the image.",
    "Describe the agricultural layout and surrounding woodlands.",
    "Summarize all prominent topographical and man-made structures.",
    "Give a concise description of the airport and surrounding facilities.",
    "Describe the port facility, docks, and maritime traffic.",
    "Provide a visual summary of the industrial complex and transport links.",
    "Tell me about the geographic features present in this image.",
    "Describe the river basin and surrounding floodplains.",
    "What are the major environmental and surface characteristics shown?",
    "Provide an executive summary of the observation."
]
for i, q in enumerate(caption_queries, 1):
    add_case(
        cid=f"caption_{i:03d}",
        query=q,
        num_images=1,
        modalities=["OPTICAL"],
        formats=["PNG"],
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="SINGLE_IMAGE_CAPTIONING",
        exp_tool="single_image_captioning",
        validity="VALID",
        err_code=None,
        category="single_image_captioning"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3. Single Image Grounding / Localization (25 cases)
# ─────────────────────────────────────────────────────────────────────────────
grounding_queries = [
    "Locate the airport runway and taxiways in this image.",
    "Find and delineate all circular fuel storage tanks.",
    "Where is the suspension bridge crossing the river?",
    "Pinpoint the maritime shipping vessels in the harbor.",
    "Locate the industrial warehouse complex in the eastern sector.",
    "Show me where the railway junction and tracks are located.",
    "Locate the sports stadium within the urban district.",
    "Find the primary water treatment plant and settling basins.",
    "Spot the electrical power substation in the northern area.",
    "Locate the highway overpass and junction ramps.",
    "Find the cluster of commercial cargo ships.",
    "Where are the solar panel arrays located?",
    "Locate the hospital or institutional building complex.",
    "Find and mark the quarry excavation site.",
    "Pinpoint the container terminal cranes and storage yards.",
    "Locate the telecommunication tower facility.",
    "Where is the central commercial shopping mall?",
    "Locate all parking lots with visible vehicle clusters.",
    "Find the dam structure and spillway along the river.",
    "Locate the coastal lighthouse on the promontory.",
    "Show me where the aircraft hangars are situated.",
    "Pinpoint the agricultural grain elevators and silos.",
    "Locate the water tower in the residential colony.",
    "Where are the toll plazas along the expressway?",
    "Locate the active mining pits in the terrain."
]
for i, q in enumerate(grounding_queries, 1):
    add_case(
        cid=f"ground_{i:03d}",
        query=q,
        num_images=1,
        modalities=["OPTICAL"],
        formats=["PNG"],
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="SINGLE_IMAGE_GROUNDING",
        exp_tool="single_image_grounding",
        validity="VALID",
        err_code=None,
        category="single_image_grounding"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Land Cover Classification & Quantification (25 cases)
# ─────────────────────────────────────────────────────────────────────────────
landcover_queries = [
    "Perform land cover classification and compute surface areas in hectares.",
    "Calculate the total vegetation coverage percentage across the image.",
    "Measure the surface area of water bodies in square kilometers.",
    "Provide a full LULC breakdown with built-up, vegetation, water, and bare soil.",
    "What is the percentage of urban built-up area in this tile?",
    "Calculate the total forest area in hectares.",
    "Quantify the agricultural cropland coverage percentage.",
    "Compute the surface area of bare rocky soil in sq km.",
    "Provide the land-cover distribution statistics for this scene.",
    "What proportion of the image is occupied by wetlands and inland waters?",
    "Calculate the built-up area in square kilometers.",
    "Estimate the percentage of tree canopy versus open grassland.",
    "Measure the total surface water coverage in hectares.",
    "Provide the surface area quantification for all land cover classes.",
    "What is the ratio of agricultural land to urban settlements?",
    "Calculate the coastal beach and sand dune surface area in ha.",
    "Classify land-cover and summarize surface areas in km².",
    "Measure the area occupied by industrial and commercial zones.",
    "Quantify the percentage of cloud and shadow contamination.",
    "Provide an automated land-cover map and area tally.",
    "Calculate the vegetative index and green space coverage.",
    "Measure the open water reservoir surface in square kilometers.",
    "Provide the hectare breakdown for cropland and forest.",
    "What percentage of this coastal scene consists of marine water?",
    "Compute the total impervious surface area in hectares."
]
for i, q in enumerate(landcover_queries, 1):
    add_case(
        cid=f"lulc_{i:03d}",
        query=q,
        num_images=1,
        modalities=["OPTICAL"],
        formats=["PNG"],
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="LAND_COVER_SEGMENTATION",
        exp_tool="land_cover_segmentation",
        validity="VALID",
        err_code=None,
        category="land_cover"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5. Bi-Temporal Change Detection (25 cases)
# ─────────────────────────────────────────────────────────────────────────────
change_queries = [
    "What changed between the two dates?",
    "Detect newly constructed buildings between pre-event and post-event observations.",
    "Identify floodwater expansion between the two dates.",
    "Analyze deforestation and loss of tree cover between the observations.",
    "Where did urban sprawl expand between baseline and follow-up?",
    "Detect surface water changes following the monsoon event.",
    "What infrastructure has been added between the two observations?",
    "Measure the area of new construction in hectares between T1 and T2.",
    "Detect changes in agricultural fields between pre and post harvest.",
    "Identify shoreline erosion and coastal changes between the two dates.",
    "What vegetation loss occurred following the wildfire?",
    "Detect landslide scars between pre-disaster and post-disaster images.",
    "Did the reservoir water level increase or decrease between the observations?",
    "Identify newly excavated mining pits between the two dates.",
    "Quantify the built-up surface area change between baseline and current tile.",
    "Detect new road segments constructed between the two satellite passes.",
    "Analyze the flood damage and inundated buildings between the two images.",
    "What changes occurred in the wetland boundaries over this period?",
    "Identify new industrial installations added between the two dates.",
    "Did the forest canopy recover or decline between the observations?",
    "Detect expanding quarry boundaries between T1 and T2.",
    "Identify agricultural expansion into previously barren land.",
    "Measure urban encroachment into protected reserve forest.",
    "Detect water body shrinkage and drought impact between the observations.",
    "What surface transitions occurred between the pre-event and post-event dates?"
]
for i, q in enumerate(change_queries, 1):
    add_case(
        cid=f"change_{i:03d}",
        query=q,
        num_images=2,
        modalities=["OPTICAL", "OPTICAL"],
        formats=["PNG", "PNG"],
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="CHANGE_ANALYSIS",
        exp_tool="bi_temporal_change",
        validity="VALID",
        err_code=None,
        category="change"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. Optical-SAR Cross-Modal Joint Analysis (20 cases)
# ─────────────────────────────────────────────────────────────────────────────
cross_modal_queries = [
    "Use SAR radar backscatter to penetrate monsoon cloud cover and map water bodies.",
    "Perform optical and SAR joint fusion for all-weather flood delineation.",
    "Fuse Sentinel-1 SAR and Sentinel-2 optical imagery to extract built-up areas.",
    "Analyze microwave dielectric backscatter and optical reflectance for flood mapping.",
    "Combine Cartosat optical resolution with RISAT SAR backscatter for coastal analysis.",
    "Use synthetic aperture radar to delineate flooded fields obscured by dense cloud.",
    "Perform cross-modal land cover extraction using co-registered Optical and SAR.",
    "Fuse microwave radar and optical multi-spectral data to map surface inundation.",
    "Leverage SAR C-band penetration with optical spectral bands for disaster assessment.",
    "Map urban settlement boundaries using fused optical reflectance and SAR roughness.",
    "Extract water body contours by combining optical NIR reflectance and SAR backscatter.",
    "Assess coastal flood impact using complementary optical and radar sensors.",
    "Use SAR backscatter to verify suspected flood zones beneath persistent cloud cover.",
    "Perform joint optical-SAR extraction for built-up and water body delineation.",
    "Fuse Sentinel-1 VV/VH polarizations with Sentinel-2 RGB bands.",
    "Analyze microwave surface scattering alongside optical multispectral signatures.",
    "Delineate water bodies under overcast skies using optical-SAR joint reasoning.",
    "Combine RISAT radar imagery with optical imagery for all-weather monitoring.",
    "Use dual-encoder optical-SAR fusion to classify terrain beneath haze.",
    "Map river network expansion during monsoons using optical and SAR fusion."
]
for i, q in enumerate(cross_modal_queries, 1):
    add_case(
        cid=f"cross_{i:03d}",
        query=q,
        num_images=2,
        modalities=["OPTICAL", "SAR"],
        formats=["PNG", "PNG"],
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="CROSS_MODAL_EXTRACTION",
        exp_tool="optical_sar_fusion",
        validity="VALID",
        err_code=None,
        category="cross_modal"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7. Adversarial & Edge Cases (22 cases)
# ─────────────────────────────────────────────────────────────────────────────
# 7a. Out of domain trivia / non-geospatial
ood_cases = [
    ("What is the capital of France?", "France capital trivia inquiry"),
    ("Write a poem about love and planetary orbits.", "Creative text generation request"),
    ("Solve the equation 2x + 5 = 15.", "Algebraic equation solving"),
    ("Who won the 2022 FIFA World Cup in Qatar?", "Sports trivia question"),
    ("Explain the principle of quantum mechanics and wave-particle duality.", "Theoretical physics inquiry"),
    ("What is the current stock price and dividend yield of Apple?", "Financial market inquiry"),
    ("Show me the interior layout of the master bedroom furniture.", "Interior building query unresolvable from nadir satellite"),
    ("Detect the nuclear submarine submerged 100 meters underwater.", "Subsurface marine vessel invisible to optical sensors")
]
for i, (q, desc) in enumerate(ood_cases, 1):
    add_case(
        cid=f"adv_ood_{i:03d}",
        query=q,
        num_images=1,
        modalities=["OPTICAL"],
        formats=["PNG"],
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="OUT_OF_DOMAIN_REJECTION",
        exp_tool="semantic_guardrail",
        validity="INVALID",
        err_code="NON_GEOSPATIAL_QUERY",
        category="adversarial_ood"
    )

# 7b. Image count mismatch
count_mismatch_cases = [
    ("What changed between pre-event and post-event observations?", 1, ["OPTICAL"], "Bi-temporal change requested with only 1 image"),
    ("Detect land-cover change and urban growth between the two dates.", 1, ["OPTICAL"], "Change query with single image upload"),
    ("Analyze bi-temporal flood expansion over time.", 1, ["OPTICAL"], "Flood change query with single image upload"),
    ("Compare surface transitions across multi-temporal observations.", 3, ["OPTICAL", "OPTICAL", "OPTICAL"], "3 images uploaded for pairwise change tool"),
    ("Fuse optical and SAR imagery to penetrate cloud cover.", 3, ["OPTICAL", "SAR", "OPTICAL"], "3 images uploaded for pairwise fusion tool"),
    ("Perform change analysis across these observations.", 0, [], "0 images uploaded for change analysis")
]
for i, (q, count, mods, desc) in enumerate(count_mismatch_cases, 1):
    add_case(
        cid=f"adv_cnt_{i:03d}",
        query=q,
        num_images=count,
        modalities=mods,
        formats=["PNG"] * count,
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="CHANGE_ANALYSIS" if "change" in q or "transitions" in q else "CROSS_MODAL_EXTRACTION",
        exp_tool="bi_temporal_change" if "change" in q or "transitions" in q else "optical_sar_fusion",
        validity="INVALID",
        err_code="INVALID_INPUT_COUNT",
        category="adversarial_count"
    )

# 7c. Modality mismatches
modality_mismatch_cases = [
    ("Use SAR radar backscatter to analyze microwave dielectric roughness.", 1, ["OPTICAL"], "Single optical image for SAR query"),
    ("Use synthetic aperture radar to penetrate cloud cover.", 1, ["OPTICAL"], "Single optical image when SAR radar is requested"),
    ("Fuse optical and SAR radar imagery to penetrate monsoon clouds.", 2, ["OPTICAL", "OPTICAL"], "Two optical images for cross-modal Optical-SAR query"),
    ("Perform optical and SAR joint flood extraction.", 2, ["OPTICAL", "OPTICAL"], "Both images optical when SAR is mandatory")
]
for i, (q, count, mods, desc) in enumerate(modality_mismatch_cases, 1):
    add_case(
        cid=f"adv_mod_{i:03d}",
        query=q,
        num_images=count,
        modalities=mods,
        formats=["PNG"] * count,
        has_geo=True,
        benchmark_mode=True,
        params=None,
        exp_task="MODALITY_GUARDRAIL" if count == 1 else "CROSS_MODAL_EXTRACTION",
        exp_tool="optical_sar_fusion",
        validity="INVALID",
        err_code="MODALITY_MISMATCH",
        category="adversarial_modality"
    )

# 7d. Format & parameter errors
param_and_format_cases = [
    ("Detect newly constructed buildings between pre and post events.", 2, ["OPTICAL", "OPTICAL"], ["PNG", "PNG"], True, False, None,
     "CHANGE_ANALYSIS", "bi_temporal_change", "INVALID", "UNSUPPORTED_FORMAT", "adversarial_format"),
    ("Detect newly constructed buildings between pre and post events.", 2, ["OPTICAL", "OPTICAL"], ["PNG", "PNG"], True, True, {"change_threshold": 5.0},
     "CHANGE_ANALYSIS", "bi_temporal_change", "INVALID", "PARAMETER_OUT_OF_RANGE", "adversarial_param"),
    ("Detect newly constructed buildings between pre and post events.", 2, ["OPTICAL", "OPTICAL"], ["PNG", "PNG"], True, True, {"unsupported_param_xyz": 123},
     "CHANGE_ANALYSIS", "bi_temporal_change", "INVALID", "INVALID_PARAMETER", "adversarial_param"),
    ("Locate the airport runway and taxiways.", 1, ["OPTICAL"], ["PNG"], True, True, {"iou_threshold": -0.5},
     "SINGLE_IMAGE_GROUNDING", "single_image_grounding", "INVALID", "PARAMETER_OUT_OF_RANGE", "adversarial_param")
]
for i, (q, count, mods, fmts, geo, bm, p, task, tool, val, code, cat) in enumerate(param_and_format_cases, 1):
    add_case(
        cid=f"adv_param_{i:03d}",
        query=q,
        num_images=count,
        modalities=mods,
        formats=fmts,
        has_geo=geo,
        benchmark_mode=bm,
        params=p,
        exp_task=task,
        exp_tool=tool,
        validity=val,
        err_code=code,
        category=cat
    )

def main():
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for case in CASES:
            f.write(json.dumps(case) + "\n")
    print(f"[OK] Generated {len(CASES)} routing cases -> {OUT_FILE}")

if __name__ == "__main__":
    main()
