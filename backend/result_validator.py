"""
SatQuery AI — Result Validator
Validates that the analysis result actually addresses the user's query.

Checks:
1. Answer type matches query intent
2. Requested targets are present
3. Land-cover % not used as answer to localization query
4. Coordinates not fabricated
5. Numeric values came from computation
6. Confidence values are valid

On failure: downgrades claims and adds warnings rather than failing silently.
"""
import logging
from typing import List

from data_models import (
    QueryIntent, AnalysisResult, ValidationResult, Detection,
    INTENT_OBJECT_DETECTION, INTENT_OBJECT_LOCALIZATION, INTENT_OBJECT_COUNTING,
    INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION, INTENT_AREA_QUANTIFICATION,
    INTENT_SCENE_UNDERSTANDING, INTENT_IMAGE_DESCRIPTION, INTENT_SPATIAL_RELATIONSHIP,
    INTENT_WATER_BODY, INTENT_VEGETATION, INTENT_AGRICULTURAL, INTENT_ROAD_NETWORK,
    INTENT_REGION_DIRECTION, INTENT_COMPARISON, INTENT_CHANGE_DETECTION, INTENT_MULTI_TASK,
)

logger = logging.getLogger(__name__)

# Intents that MUST produce detections, not land-cover stats
DETECTION_INTENTS = {
    INTENT_OBJECT_DETECTION,
    INTENT_OBJECT_LOCALIZATION,
    INTENT_OBJECT_COUNTING,
    INTENT_SPATIAL_RELATIONSHIP,
}

# Intents where land-cover percentages ARE valid
LANDCOVER_VALID_INTENTS = {
    INTENT_LAND_COVER,
    INTENT_SEMANTIC_SEGMENTATION,
    INTENT_AREA_QUANTIFICATION,
    INTENT_SCENE_UNDERSTANDING,
    INTENT_IMAGE_DESCRIPTION,
    INTENT_REGION_DIRECTION,
    INTENT_COMPARISON,
    INTENT_CHANGE_DETECTION,
    INTENT_WATER_BODY,
    INTENT_VEGETATION,
    INTENT_AGRICULTURAL,
    INTENT_ROAD_NETWORK,
    INTENT_MULTI_TASK,
}


class ResultValidator:
    """
    Validates analysis results against the original query intent.
    Returns a ValidationResult with issues/warnings and a final confidence score.
    """

    def validate(self, qi: QueryIntent, result: AnalysisResult) -> ValidationResult:
        issues = []
        warnings = []
        downgraded = []

        # ── Check 1: Detection intents must produce detections ──────────────
        effective_intent = qi.intent
        is_detection_query = effective_intent in DETECTION_INTENTS
        if effective_intent == INTENT_MULTI_TASK:
            is_detection_query = any(s in DETECTION_INTENTS for s in qi.sub_intents)

        if is_detection_query:
            if len(result.detections) == 0:
                warnings.append(
                    "Query requests object localization but no detections were produced. "
                    "The scene may lack detectable structures at this resolution, "
                    "or the target class is not supported by the current CV engine."
                )
                # Not a hard failure — the system should report "none found"

        # ── Check 2: Land-cover % should not be primary answer for pure detection queries ──
        user_wanted_landcover = (
            effective_intent in (INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION, INTENT_SCENE_UNDERSTANDING, INTENT_IMAGE_DESCRIPTION) or
            (effective_intent == INTENT_MULTI_TASK and any(s in (INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION, INTENT_SCENE_UNDERSTANDING) for s in qi.sub_intents))
        )
        if is_detection_query and not user_wanted_landcover and result.segments:
            # Segments are fine to include as supplementary context,
            # but they must not be the ONLY result when pure detection was requested
            if len(result.detections) == 0 and len(result.segments) > 0:
                issues.append(
                    "ANSWER TYPE MISMATCH: Query requested object localization/detection, "
                    "but only land-cover percentages were produced. "
                    "Land-cover statistics are NOT a substitute for object detection."
                )
                downgraded.append("Land-cover percentages downgraded from primary to supplementary context.")

        # ── Check 3: Verify requested targets are addressed ─────────────────
        if qi.targets:
            detected_classes = {d.class_name for d in result.detections}
            for target in qi.targets:
                if target not in detected_classes and target not in (result.limitations or []):
                    warnings.append(
                        f"Requested target '{target}' is not represented in the results. "
                        f"This may indicate the object is absent, undetectable at this resolution, "
                        f"or not supported by the current detection engine."
                    )

        # ── Check 4: Coordinate fabrication guard ───────────────────────────
        has_geo = bool(result.geo_metadata and "west" in result.geo_metadata)
        for det in result.detections:
            if det.geographic_position and not has_geo:
                issues.append(
                    f"Detection '{det.id}' has geographic coordinates but no georeferencing metadata "
                    "was found. Geographic coordinates removed."
                )
                det.geographic_position = None  # Remove fabricated coords
                downgraded.append(f"Geographic coordinates removed from {det.id} (no georeferencing).")

        # ── Check 5: Count queries need actual count ─────────────────────────
        if qi.requires_count and effective_intent == INTENT_OBJECT_COUNTING:
            count_metric = next(
                (m for m in result.measurements if "count" in m.metric.lower()), None
            )
            if count_metric is None:
                warnings.append("Count query detected but no count metric was produced.")

        # ── Check 6: Confidence sanity ───────────────────────────────────────
        for det in result.detections:
            if not 0.0 <= det.confidence <= 1.0:
                det.confidence = max(0.0, min(1.0, det.confidence))
                warnings.append(f"Confidence for {det.id} was out of range; clamped to [0,1].")

        # ── Compute overall confidence ───────────────────────────────────────
        base_conf = result.confidence
        if issues:
            base_conf = max(0.20, base_conf - 0.25 * len(issues))
        if warnings:
            base_conf = max(0.35, base_conf - 0.08 * len(warnings))

        passed = len(issues) == 0
        if not passed:
            logger.warning(f"Validation FAILED for intent={qi.intent}: {issues}")
        else:
            logger.info(f"Validation PASSED for intent={qi.intent}")

        return ValidationResult(
            passed=passed,
            issues=issues,
            warnings=warnings,
            downgraded_claims=downgraded,
            overall_confidence=round(base_conf, 2)
        )


# Singleton
result_validator = ResultValidator()
