"""
SatQuery AI — Typed Tool Registry (Pydantic v2)
Problem Statement Alignment: SIH26167 (ISRO / Space Applications Centre)

Defines strict schemas for all specialist remote sensing tools:
- Input specifications (min/max image count, allowed sensor modalities, file formats)
- Parameter specifications with type checks, numeric ranges, and allowed enums
- Output schemas (text answer, visual evidence, GeoJSON, measurements, confidence)
- Structured error representations with actionable remediation guidance
- Comprehensive execution trace schemas
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums for Modality, Format, and Error Codes
# ─────────────────────────────────────────────────────────────────────────────

class SensorModality(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    MULTISPECTRAL = "MULTISPECTRAL"
    HYPERSPECTRAL = "HYPERSPECTRAL"
    UNKNOWN = "UNKNOWN"


class ImageFormat(str, Enum):
    GEOTIFF = "GEOTIFF"
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"


class StructuredErrorCode(str, Enum):
    INVALID_INPUT_COUNT = "INVALID_INPUT_COUNT"
    MODALITY_MISMATCH = "MODALITY_MISMATCH"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    NON_GEOSPATIAL_QUERY = "NON_GEOSPATIAL_QUERY"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    MISSING_REQUIRED_PARAMETER = "MISSING_REQUIRED_PARAMETER"
    PARAMETER_OUT_OF_RANGE = "PARAMETER_OUT_OF_RANGE"
    SPATIAL_DISPARITY = "SPATIAL_DISPARITY"
    IDENTICAL_IMAGES = "IDENTICAL_IMAGES"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"


# ─────────────────────────────────────────────────────────────────────────────
# Schema Definitions
# ─────────────────────────────────────────────────────────────────────────────

class InputRequirements(BaseModel):
    """Specification of input imagery requirements for a specialist tool."""
    min_images: int = Field(ge=0, le=10, description="Minimum number of input images required")
    max_images: int = Field(ge=0, le=10, description="Maximum number of input images accepted")
    allowed_modalities: List[SensorModality] = Field(
        default_factory=lambda: [SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
        description="List of permitted sensor modalities"
    )
    allowed_formats: List[ImageFormat] = Field(
        default_factory=lambda: [ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG],
        description="Supported raster file formats"
    )
    requires_georeferencing: bool = Field(
        default=False,
        description="Whether georeferenced raster metadata (CRS, Affine transform) is mandatory"
    )
    requires_cross_modal_pair: bool = Field(
        default=False,
        description="Whether tool strictly requires complementary modalities (e.g., 1 Optical + 1 SAR)"
    )


class ParameterSpec(BaseModel):
    """Specification and validation constraints for an individual tool parameter."""
    name: str
    param_type: str = Field(description="float | int | str | bool | list[str]")
    description: str
    default: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = False

    def validate_value(self, value: Any) -> Optional[str]:
        """Validate a parameter value against this specification. Returns error message or None."""
        if value is None:
            if self.required:
                return f"Missing required parameter '{self.name}'"
            return None

        # Type validation
        if self.param_type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return f"Parameter '{self.name}' must be a float, got {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return f"Parameter '{self.name}' value {value} is below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return f"Parameter '{self.name}' value {value} exceeds maximum {self.max_value}"

        elif self.param_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"Parameter '{self.name}' must be an integer, got {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return f"Parameter '{self.name}' value {value} is below minimum {int(self.min_value)}"
            if self.max_value is not None and value > self.max_value:
                return f"Parameter '{self.name}' value {value} exceeds maximum {int(self.max_value)}"

        elif self.param_type == "str":
            if not isinstance(value, str):
                return f"Parameter '{self.name}' must be a string, got {type(value).__name__}"
            if self.allowed_values and value not in self.allowed_values:
                return f"Parameter '{self.name}' value '{value}' is not permitted. Allowed: {self.allowed_values}"

        elif self.param_type == "bool":
            if not isinstance(value, bool):
                return f"Parameter '{self.name}' must be a boolean, got {type(value).__name__}"

        elif self.param_type == "list[str]":
            if not isinstance(value, (list, tuple)):
                return f"Parameter '{self.name}' must be a list of strings, got {type(value).__name__}"
            if not all(isinstance(v, str) for v in value):
                return f"Parameter '{self.name}' elements must all be strings"
            if self.allowed_values:
                for v in value:
                    if v not in self.allowed_values:
                        return f"Parameter '{self.name}' contains invalid element '{v}'. Allowed: {self.allowed_values}"

        return None


class OutputSchema(BaseModel):
    """Specification of outputs produced by a specialist tool."""
    has_text_answer: bool = True
    has_visual_evidence: bool = True
    has_geojson: bool = False
    has_mask: bool = False
    has_measurements: bool = False
    has_derived_confidence: bool = True


class ToolSpec(BaseModel):
    """Complete typed specification for a specialist remote sensing tool."""
    tool_id: str
    task_name: str
    version: str = "1.0.0"
    description: str
    inputs: InputRequirements
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    output_schema: OutputSchema = Field(default_factory=OutputSchema)
    is_heuristic_fallback: bool = False
    provenance_label: str


class StructuredError(BaseModel):
    """Standardized structured error returned when validation or guardrails fail."""
    status: str = "error"
    error_code: StructuredErrorCode
    error_type: str
    message: str
    reason: str
    suggested_fix: str
    invalid_parameter: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """Record of an individual tool execution in the agentic trace."""
    tool_id: str
    task_name: str
    version: str
    status: str  # "SUCCESS" | "FAILED" | "REJECTED"
    latency_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class ExecutionTrace(BaseModel):
    """Full execution trace recording the 6-stage lifecycle of an agentic query."""
    trace_id: str
    timestamp: str
    query: str
    selected_task: str
    task_confidence: float
    input_checks: Dict[str, Any]
    tools_called: List[ToolCallRecord] = Field(default_factory=list)
    per_step_latency_ms: Dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float
    outputs_summary: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    confidence_derivation: str
    valid_sequence: bool
    is_invalid_call: bool
    structured_error: Optional[StructuredError] = None


# ─────────────────────────────────────────────────────────────────────────────
# Master Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central repository of typed specifications for all SatQuery AI tools.
    Provides validation of inputs, formats, parameters, and modalities.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._register_default_tools()

    def register(self, spec: ToolSpec):
        """Register a new tool specification."""
        self._tools[spec.tool_id] = spec

    def get_tool(self, tool_id: str) -> Optional[ToolSpec]:
        """Retrieve tool specification by tool_id or task_name."""
        if tool_id in self._tools:
            return self._tools[tool_id]
        for tool in self._tools.values():
            if tool.task_name == tool_id:
                return tool
        return None

    def list_tools(self) -> List[ToolSpec]:
        """Return all registered tool specifications."""
        return list(self._tools.values())

    def _register_default_tools(self):
        # 1. Single Image VQA
        self.register(ToolSpec(
            tool_id="single_image_vqa",
            task_name="SINGLE_IMAGE_VQA",
            version="1.0.0",
            description="Visual Question Answering on single remote sensing images using RS-adapted VLM (RSVQA / VRSBench).",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG],
                requires_georeferencing=False
            ),
            parameters={
                "question": ParameterSpec(
                    name="question",
                    param_type="str",
                    description="Natural language question about remote sensing imagery",
                    required=True
                ),
                "confidence_threshold": ParameterSpec(
                    name="confidence_threshold",
                    param_type="float",
                    description="Minimum confidence threshold for positive answers",
                    default=0.25,
                    min_value=0.0,
                    max_value=1.0
                ),
                "max_length": ParameterSpec(
                    name="max_length",
                    param_type="int",
                    description="Maximum token length for generated response",
                    default=128,
                    min_value=16,
                    max_value=256
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=False,
                has_mask=False,
                has_measurements=False,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery-RS-VLM-v1.0 (BLIP RS-Adapted ViT-B/16)"
        ))

        # 2. Single Image Captioning
        self.register(ToolSpec(
            tool_id="single_image_captioning",
            task_name="SINGLE_IMAGE_CAPTIONING",
            version="1.0.0",
            description="Comprehensive scene description and semantic captioning for single remote sensing scenes.",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "detail_level": ParameterSpec(
                    name="detail_level",
                    param_type="str",
                    description="Level of detail in generated scene caption",
                    default="standard",
                    allowed_values=["short", "standard", "comprehensive"]
                ),
                "max_length": ParameterSpec(
                    name="max_length",
                    param_type="int",
                    description="Maximum token length for caption",
                    default=180,
                    min_value=32,
                    max_value=512
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=False,
                has_mask=False,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery-RS-VLM-v1.0 (VRSBench Captioning)"
        ))

        # 3. Single Image Grounding / Sector Delineation
        self.register(ToolSpec(
            tool_id="single_image_grounding",
            task_name="SINGLE_IMAGE_GROUNDING",
            version="1.0.0",
            description="Heuristic land-cover and sector delineation (water, land, port, runway, ship) with bounding box and GeoJSON polygon extraction.",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "target_expression": ParameterSpec(
                    name="target_expression",
                    param_type="str",
                    description="Geospatial target or sector expression (e.g., port, runway, water, land, ship)",
                    required=True
                ),
                "iou_threshold": ParameterSpec(
                    name="iou_threshold",
                    param_type="float",
                    description="IoU threshold for overlapping sector boundaries",
                    default=0.45,
                    min_value=0.1,
                    max_value=0.95
                ),
                "max_detections": ParameterSpec(
                    name="max_detections",
                    param_type="int",
                    description="Maximum number of sectors to delineate",
                    default=50,
                    min_value=1,
                    max_value=200
                ),
                "threshold": ParameterSpec(
                    name="threshold",
                    param_type="float",
                    description="Confidence threshold for neural visual grounding",
                    default=0.08,
                    min_value=0.001,
                    max_value=1.0,
                    required=False
                ),
                "force_heuristic": ParameterSpec(
                    name="force_heuristic",
                    param_type="bool",
                    description="Force execution of classical heuristic contour delineation",
                    default=False,
                    required=False
                ),
                "mode": ParameterSpec(
                    name="mode",
                    param_type="str",
                    description="Grounding execution mode ('auto', 'neural', or 'heuristic')",
                    default="auto",
                    required=False
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=True,
                has_mask=False,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=True,
            provenance_label="SatQuery Heuristic Delineator (Spectral/Morphological Sector Grounding)"
        ))

        # 4. Land Cover Segmentation & Surface Quantification
        self.register(ToolSpec(
            tool_id="land_cover_segmentation",
            task_name="LAND_COVER_SEGMENTATION",
            version="1.0.0",
            description="Multi-class land-cover classification and surface area quantification in hectares and km².",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "classes": ParameterSpec(
                    name="classes",
                    param_type="list[str]",
                    description="Target land-cover classes to quantify",
                    default=["water", "forest", "crop", "built_up", "bare_soil"],
                    allowed_values=["water", "forest", "crop", "built_up", "bare_soil", "cloud", "vegetation"]
                ),
                "calculate_area": ParameterSpec(
                    name="calculate_area",
                    param_type="bool",
                    description="Whether to calculate surface areas in km² and ha",
                    default=True
                ),
                "gsd_meters": ParameterSpec(
                    name="gsd_meters",
                    param_type="float",
                    description="Ground Sampling Distance in meters per pixel if unreferenced",
                    default=10.0,
                    min_value=0.1,
                    max_value=100.0
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=True,
                has_mask=True,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery Land-Cover Engine"
        ))

        # 5. Bi-Temporal Change Analysis
        self.register(ToolSpec(
            tool_id="bi_temporal_change",
            task_name="CHANGE_ANALYSIS",
            version="1.0.0",
            description="Learned bi-temporal change detection and CDVQA difference question answering using SatQueryChangeNet.",
            inputs=InputRequirements(
                min_images=2,
                max_images=2,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG],
                requires_coregistration=True
            ),
            parameters={
                "change_threshold": ParameterSpec(
                    name="change_threshold",
                    param_type="float",
                    description="Confidence threshold for pixel difference segmentation",
                    default=0.35,
                    min_value=0.05,
                    max_value=0.95
                ),
                "min_contour_area_px": ParameterSpec(
                    name="min_contour_area_px",
                    param_type="int",
                    description="Minimum connected component pixel area to retain as a change polygon",
                    default=25,
                    min_value=5,
                    max_value=2000
                ),
                "quadrant_analysis": ParameterSpec(
                    name="quadrant_analysis",
                    param_type="bool",
                    description="Whether to segment changes into geographic/image quadrants",
                    default=True
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=True,
                has_mask=True,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQueryChangeNet (Siamese Feature Diff U-Net, CDVQA trained)"
        ))

        # 6. Optical-SAR Joint Analysis
        self.register(ToolSpec(
            tool_id="optical_sar_fusion",
            task_name="CROSS_MODAL_EXTRACTION",
            version="1.0.0",
            description="Deep cross-modal joint reasoning combining Optical imagery with SAR microwave backscatter (SatQueryOpticalSARNet).",
            inputs=InputRequirements(
                min_images=2,
                max_images=2,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.SAR],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG],
                requires_cross_modal_pair=True
            ),
            parameters={
                "target_classes": ParameterSpec(
                    name="target_classes",
                    param_type="list[str]",
                    description="Target classes for cross-modal extraction",
                    default=["water", "vegetation", "built_up"],
                    allowed_values=["water", "vegetation", "built_up"]
                ),
                "min_nmi": ParameterSpec(
                    name="min_nmi",
                    param_type="float",
                    description="Minimum Normalized Mutual Information for unreferenced co-registration",
                    default=0.010,
                    min_value=0.001,
                    max_value=0.1
                ),
                "confidence_threshold": ParameterSpec(
                    name="confidence_threshold",
                    param_type="float",
                    description="Threshold for multi-modal class probability activation",
                    default=0.30,
                    min_value=0.05,
                    max_value=0.95
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=True,
                has_mask=True,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)"
        ))

        # 7. Spatial Compatibility Verifier
        self.register(ToolSpec(
            tool_id="spatial_compatibility_verifier",
            task_name="COMPATIBILITY_CHECK",
            version="1.0.0",
            description="Validates spatial co-registration across multi-temporal or cross-modal imagery pairs.",
            inputs=InputRequirements(
                min_images=2,
                max_images=2,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.SAR, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "verification_mode": ParameterSpec(
                    name="verification_mode",
                    param_type="str",
                    description="Verification algorithm to execute",
                    default="auto",
                    allowed_values=["auto", "optical_optical_sift", "optical_sar_nmi", "rasterio_crs_bounds"]
                ),
                "ransac_threshold": ParameterSpec(
                    name="ransac_threshold",
                    param_type="float",
                    description="RANSAC reprojection error threshold in pixels for SIFT",
                    default=5.0,
                    min_value=1.0,
                    max_value=20.0
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=False,
                has_mask=False,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery Spatial Coherence Engine (Rasterio + SIFT/RANSAC + NMI)"
        ))

        # 8. Classical Heuristic Fallback (Strictly labeled)
        self.register(ToolSpec(
            tool_id="heuristic_fallback",
            task_name="FALLBACK_HEURISTIC",
            version="1.0.0",
            description="Classical rule-based computer vision routines (color thresholds, edge contours). Strictly designated as fallback.",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "method": ParameterSpec(
                    name="method",
                    param_type="str",
                    description="Classical CV algorithm",
                    default="color_threshold",
                    allowed_values=["color_threshold", "edge_contour", "otsu"]
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=False,
                has_mask=True,
                has_measurements=False,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=True,
            provenance_label="SatQuery Classical CV Engine (fallback_heuristic)"
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # Input & Parameter Validation Methods
    # ─────────────────────────────────────────────────────────────────────────

    def validate_inputs(
        self,
        tool_id: str,
        num_images: int,
        detected_modalities: List[str],
        detected_formats: List[str],
        benchmark_mode: bool = True
    ) -> Optional[StructuredError]:
        """Validate input count, modalities, and formats against tool requirements."""
        tool = self.get_tool(tool_id)
        if not tool:
            return StructuredError(
                error_code=StructuredErrorCode.INVALID_PARAMETER,
                error_type="UnknownToolError",
                message=f"Specialist tool '{tool_id}' is not registered.",
                reason="The controller attempted to dispatch an unregistered tool ID.",
                suggested_fix=f"Register tool '{tool_id}' or route to an existing registered tool."
            )

        reqs = tool.inputs

        # 1. Count validation
        if num_images < reqs.min_images or num_images > reqs.max_images:
            reason = (
                f"Tool '{tool.task_name}' requires between {reqs.min_images} and {reqs.max_images} images, "
                f"but {num_images} image(s) were supplied."
            )
            suggested = f"Provide exactly {reqs.min_images} image(s) for {tool.task_name}." if reqs.min_images == reqs.max_images else f"Provide {reqs.min_images} to {reqs.max_images} images."
            return StructuredError(
                error_code=StructuredErrorCode.INVALID_INPUT_COUNT,
                error_type="InputCountError",
                message=f"Invalid image count for {tool.task_name}: received {num_images}, expected {reqs.min_images}-{reqs.max_images}.",
                reason=reason,
                suggested_fix=suggested,
                details={"received_count": num_images, "expected_min": reqs.min_images, "expected_max": reqs.max_images}
            )

        # 2. Format validation (when benchmark mode is off, PNG/JPEG without georeferencing is rejected)
        if not benchmark_mode:
            for fmt in detected_formats:
                fmt_upper = fmt.upper()
                if fmt_upper in ("PNG", "JPEG", "JPG"):
                    return StructuredError(
                        error_code=StructuredErrorCode.UNSUPPORTED_FORMAT,
                        error_type="UnsupportedFormatError",
                        message=f"Standard raster format '{fmt}' is not permitted in production georeferenced mode.",
                        reason="Operational ISRO queries require calibrated 16-bit GeoTIFF imagery with CRS/transform metadata.",
                        suggested_fix="Upload a georeferenced GeoTIFF (.tif/.geotiff) or enable benchmark mode toggle for public PNG/JPEG datasets.",
                        details={"detected_format": fmt, "benchmark_mode": benchmark_mode}
                    )

        # 3. Cross-modal complementary pair check (Optical + SAR)
        if reqs.requires_cross_modal_pair:
            mod_set = {m.upper() for m in detected_modalities}
            has_optical = any(m in ("OPTICAL", "MULTISPECTRAL") for m in mod_set)
            has_sar = "SAR" in mod_set
            if not (has_optical and has_sar):
                return StructuredError(
                    error_code=StructuredErrorCode.MODALITY_MISMATCH,
                    error_type="CrossModalMismatchError",
                    message="Cross-modal analysis requires complementary Optical and SAR microwave imagery.",
                    reason=f"Detected modalities {detected_modalities} do not contain both Optical and SAR inputs.",
                    suggested_fix="Upload one Optical multispectral image and one Synthetic Aperture Radar (SAR) tile (e.g. RISAT-1 or Sentinel-1).",
                    details={"detected_modalities": detected_modalities}
                )

        # 4. Modality check against allowed modalities
        allowed_str = [m.value for m in reqs.allowed_modalities]
        for m in detected_modalities:
            if m.upper() not in allowed_str and m.upper() != "UNKNOWN":
                return StructuredError(
                    error_code=StructuredErrorCode.MODALITY_MISMATCH,
                    error_type="ModalityMismatchError",
                    message=f"Modality '{m}' is not supported by {tool.task_name}.",
                    reason=f"Tool '{tool.task_name}' accepts only {allowed_str}, but received '{m}'.",
                    suggested_fix=f"Provide imagery matching one of: {allowed_str}.",
                    details={"received_modality": m, "allowed_modalities": allowed_str}
                )

        return None

    def validate_parameters(
        self,
        tool_id: str,
        parameters: Dict[str, Any]
    ) -> Optional[StructuredError]:
        """Validate parameter dictionary against tool specification."""
        tool = self.get_tool(tool_id)
        if not tool:
            return StructuredError(
                error_code=StructuredErrorCode.INVALID_PARAMETER,
                error_type="UnknownToolError",
                message=f"Tool '{tool_id}' not found.",
                reason="Cannot validate parameters for an unknown tool.",
                suggested_fix="Verify the tool ID."
            )

        # Check for unknown parameters
        for p_name in parameters.keys():
            if p_name not in tool.parameters:
                return StructuredError(
                    error_code=StructuredErrorCode.INVALID_PARAMETER,
                    error_type="UnknownParameterError",
                    message=f"Unknown parameter '{p_name}' for tool '{tool.task_name}'.",
                    reason=f"Tool '{tool.task_name}' does not permit parameter '{p_name}'.",
                    suggested_fix=f"Permitted parameters: {list(tool.parameters.keys())}",
                    invalid_parameter=p_name,
                    details={"permitted_parameters": list(tool.parameters.keys())}
                )

        # Validate values of known parameters
        for p_name, p_spec in tool.parameters.items():
            val = parameters.get(p_name, p_spec.default)
            err = p_spec.validate_value(val)
            if err:
                code = StructuredErrorCode.PARAMETER_OUT_OF_RANGE if ("below minimum" in err or "exceeds maximum" in err) else StructuredErrorCode.INVALID_PARAMETER
                return StructuredError(
                    error_code=code,
                    error_type="ParameterValidationError",
                    message=err,
                    reason=f"Validation failed for parameter '{p_name}'.",
                    suggested_fix=f"Adjust '{p_name}' to meet constraints: {p_spec.description}",
                    invalid_parameter=p_name,
                    details={"parameter": p_name, "expected_type": p_spec.param_type}
                )

        return None


# Global singleton instance of ToolRegistry
tool_registry = ToolRegistry()
