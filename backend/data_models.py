"""
SatQuery AI — Unified Data Models
Structured dataclasses for the entire query → analysis → answer pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# INTENT TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────

INTENT_OBJECT_DETECTION          = "OBJECT_DETECTION"
INTENT_OBJECT_LOCALIZATION       = "OBJECT_LOCALIZATION"
INTENT_OBJECT_COUNTING           = "OBJECT_COUNTING"
INTENT_OBJECT_CLASSIFICATION     = "OBJECT_CLASSIFICATION"
INTENT_OBJECT_ATTRIBUTE          = "OBJECT_ATTRIBUTE_ANALYSIS"
INTENT_LAND_COVER                = "LAND_COVER_CLASSIFICATION"
INTENT_SEMANTIC_SEGMENTATION     = "SEMANTIC_SEGMENTATION"
INTENT_INSTANCE_SEGMENTATION     = "INSTANCE_SEGMENTATION"
INTENT_AREA_QUANTIFICATION       = "AREA_QUANTIFICATION"
INTENT_DISTANCE_PROXIMITY        = "DISTANCE_PROXIMITY_ANALYSIS"
INTENT_SPATIAL_RELATIONSHIP      = "SPATIAL_RELATIONSHIP_ANALYSIS"
INTENT_REGION_DIRECTION          = "REGION_DIRECTION_ANALYSIS"
INTENT_IMAGE_DESCRIPTION         = "IMAGE_DESCRIPTION"
INTENT_SCENE_UNDERSTANDING       = "SCENE_UNDERSTANDING"
INTENT_CHANGE_DETECTION          = "CHANGE_DETECTION"
INTENT_COMPARISON                = "COMPARISON"
INTENT_TEMPORAL                  = "TEMPORAL_ANALYSIS"
INTENT_ANOMALY_DETECTION         = "ANOMALY_DETECTION"
INTENT_INFRASTRUCTURE            = "INFRASTRUCTURE_ANALYSIS"
INTENT_AGRICULTURAL              = "AGRICULTURAL_ANALYSIS"
INTENT_WATER_BODY                = "WATER_BODY_ANALYSIS"
INTENT_VEGETATION                = "VEGETATION_ANALYSIS"
INTENT_ROAD_NETWORK              = "ROAD_NETWORK_ANALYSIS"
INTENT_COORDINATE_REQUEST        = "COORDINATE_REQUEST"
INTENT_MULTI_TASK                = "MULTI_TASK_QUERY"


@dataclass
class QueryIntent:
    """Structured representation of a parsed user query."""
    raw_query: str
    intent: str                                   # Primary INTENT_* constant
    sub_intents: List[str] = field(default_factory=list)   # For multi-task
    targets: List[str] = field(default_factory=list)       # e.g. ["building","road"]
    operation: str = "analyze"                    # locate, count, calculate, describe
    spatial_constraint: Optional[str] = None      # "northern part", "near roads"
    temporal_constraint: Optional[str] = None     # "between two dates"
    quantitative: bool = False                    # does user want numbers?
    requires_coordinates: bool = False
    requires_visual_overlay: bool = True
    requires_area_calculation: bool = False
    requires_count: bool = False
    requires_comparison: bool = False
    confidence: float = 0.8
    required_vision_capability: str = "general"  # general|detection|segmentation|vlm|change


@dataclass
class AnalysisStep:
    """One step in the analysis plan."""
    step_type: str                  # "object_detection", "land_cover", "scene_description" …
    targets: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1               # lower = earlier


@dataclass
class AnalysisPlan:
    """Ordered set of analysis steps derived from a QueryIntent."""
    intent: QueryIntent
    steps: List[AnalysisStep] = field(default_factory=list)
    is_multi_step: bool = False


@dataclass
class Detection:
    """A single detected object with spatial metadata."""
    id: str                        # e.g. "building_01"
    class_name: str                # canonical class: "building"
    display_name: str              # human-readable: "Building #01"
    confidence: float              # 0.0 – 1.0
    confidence_label: str          # "High" | "Moderate" | "Low" | "Uncertain"
    bbox_norm: Dict[str, float]    # {x_min, y_min, x_max, y_max} in [0,1]
    bbox_px: Dict[str, int]        # pixel coordinates
    centroid_norm: Dict[str, float]
    image_position: str            # "central-north", "lower-left" …
    geographic_position: Optional[Dict[str, float]] = None  # lat/lon if georeferenced
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    """A land-cover or semantic segment."""
    class_name: str
    pixel_count: int
    percentage: float
    area_km2: Optional[float] = None
    area_ha: Optional[float] = None
    mask_available: bool = True


@dataclass
class Measurement:
    """A single quantitative metric with provenance."""
    metric: str                    # "built_up_percentage", "building_count" …
    value: Any
    unit: str                      # "%", "km²", "count", "pixels"
    source: str                    # "pixel_analysis", "contour_detection" …
    confidence: float = 0.8
    note: Optional[str] = None     # uncertainty notes


@dataclass
class SpatialRelationship:
    """Spatial relationship between two sets of objects."""
    subject_class: str
    relation: str              # "near", "inside", "north_of", "overlaps"
    object_class: str
    instances: List[str] = field(default_factory=list)  # subject ids


@dataclass
class AnalysisResult:
    """Unified result container output by any analysis engine."""
    query_intent: QueryIntent
    analysis_methods: List[str] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    measurements: List[Measurement] = field(default_factory=list)
    spatial_relationships: List[SpatialRelationship] = field(default_factory=list)
    visual_evidence_b64: Optional[str] = None
    visual_description: str = ""
    scene_description: str = ""
    geo_metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.8
    limitations: List[str] = field(default_factory=list)
    model_provenance: str = "SatQuery CV Engine"
    raw_land_cover_stats: Optional[Dict[str, Any]] = None  # kept for land-cover queries


@dataclass
class ValidationResult:
    """Outcome of the result validation layer."""
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    downgraded_claims: List[str] = field(default_factory=list)
    overall_confidence: float = 0.8
