"""
SatQuery AI — Analysis Planner
Maps a QueryIntent to an ordered list of analysis steps (AnalysisPlan).

This module decides WHAT to analyze — NOT how. Execution is handled by
the QueryRouter which dispatches each step to the appropriate engine.
"""
import logging
from typing import List

from data_models import (
    QueryIntent, AnalysisPlan, AnalysisStep,
    INTENT_OBJECT_DETECTION, INTENT_OBJECT_LOCALIZATION, INTENT_OBJECT_COUNTING,
    INTENT_OBJECT_CLASSIFICATION, INTENT_OBJECT_ATTRIBUTE,
    INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION, INTENT_INSTANCE_SEGMENTATION,
    INTENT_AREA_QUANTIFICATION, INTENT_DISTANCE_PROXIMITY, INTENT_SPATIAL_RELATIONSHIP,
    INTENT_REGION_DIRECTION, INTENT_IMAGE_DESCRIPTION, INTENT_SCENE_UNDERSTANDING,
    INTENT_CHANGE_DETECTION, INTENT_COMPARISON, INTENT_TEMPORAL, INTENT_ANOMALY_DETECTION,
    INTENT_INFRASTRUCTURE, INTENT_AGRICULTURAL, INTENT_WATER_BODY, INTENT_VEGETATION,
    INTENT_ROAD_NETWORK, INTENT_COORDINATE_REQUEST, INTENT_MULTI_TASK,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Step type constants (consumed by the QueryRouter executor)
# ─────────────────────────────────────────────────────────────────────────────
STEP_OBJECT_DETECTION    = "object_detection"
STEP_LAND_COVER          = "land_cover_segmentation"
STEP_WATER_DETECTION     = "water_body_detection"
STEP_VEGETATION_ANALYSIS = "vegetation_analysis"
STEP_ROAD_DETECTION      = "road_detection"
STEP_SCENE_DESCRIPTION   = "scene_description"
STEP_CHANGE_DETECTION    = "change_detection"
STEP_REGION_ANALYSIS     = "region_direction_analysis"
STEP_AREA_CALCULATION    = "area_calculation"
STEP_COUNT_OBJECTS       = "count_objects"
STEP_SPATIAL_RELATION    = "spatial_relationship"
STEP_COORDINATE_EXTRACT  = "coordinate_extraction"


def _steps_for_intent(intent: str, qi: QueryIntent) -> List[AnalysisStep]:
    """
    Return the ordered list of AnalysisStep objects for a single intent.
    """
    targets = qi.targets

    if intent == INTENT_OBJECT_DETECTION:
        return [AnalysisStep(STEP_OBJECT_DETECTION, targets=targets, priority=1)]

    if intent == INTENT_OBJECT_LOCALIZATION:
        steps = [AnalysisStep(STEP_OBJECT_DETECTION, targets=targets, priority=1)]
        if qi.requires_coordinates:
            steps.append(AnalysisStep(STEP_COORDINATE_EXTRACT, targets=targets, priority=2))
        return steps

    if intent == INTENT_OBJECT_COUNTING:
        return [
            AnalysisStep(STEP_OBJECT_DETECTION, targets=targets, priority=1),
            AnalysisStep(STEP_COUNT_OBJECTS,    targets=targets, priority=2),
        ]

    if intent == INTENT_OBJECT_CLASSIFICATION:
        return [AnalysisStep(STEP_OBJECT_DETECTION, targets=targets, priority=1)]

    if intent == INTENT_LAND_COVER:
        return [AnalysisStep(STEP_LAND_COVER, priority=1)]

    if intent == INTENT_SEMANTIC_SEGMENTATION:
        return [AnalysisStep(STEP_LAND_COVER, targets=targets, priority=1)]

    if intent == INTENT_AREA_QUANTIFICATION:
        # Determine what to segment: specific targets or full land-cover
        seg_targets = targets or []
        steps = []
        # Check which specific segmentation is needed
        water_related = any(t in ["water_body","coastline"] for t in seg_targets)
        veg_related   = any(t in ["vegetation","forest","agricultural_field"] for t in seg_targets)
        built_related = any(t in ["building","urban_area","road","infrastructure"] for t in seg_targets)

        if water_related:
            steps.append(AnalysisStep(STEP_WATER_DETECTION, targets=seg_targets, priority=1))
        if veg_related:
            steps.append(AnalysisStep(STEP_VEGETATION_ANALYSIS, targets=seg_targets, priority=1))
        if not steps or built_related:
            steps.append(AnalysisStep(STEP_LAND_COVER, targets=seg_targets, priority=1))
        steps.append(AnalysisStep(STEP_AREA_CALCULATION, targets=seg_targets, priority=2))
        return steps

    if intent == INTENT_SPATIAL_RELATIONSHIP:
        return [
            AnalysisStep(STEP_OBJECT_DETECTION, targets=targets, priority=1),
            AnalysisStep(STEP_SPATIAL_RELATION, targets=targets, priority=2),
        ]

    if intent == INTENT_REGION_DIRECTION:
        return [
            AnalysisStep(STEP_REGION_ANALYSIS, parameters={"constraint": qi.spatial_constraint}, priority=1),
            AnalysisStep(STEP_LAND_COVER, priority=2),  # provide land-cover for the region
        ]

    if intent == INTENT_SCENE_UNDERSTANDING:
        return [
            AnalysisStep(STEP_SCENE_DESCRIPTION, priority=1),
            AnalysisStep(STEP_LAND_COVER, priority=2),
        ]

    if intent == INTENT_IMAGE_DESCRIPTION:
        return [
            AnalysisStep(STEP_SCENE_DESCRIPTION, priority=1),
            AnalysisStep(STEP_LAND_COVER, priority=2),
        ]

    if intent == INTENT_CHANGE_DETECTION:
        return [AnalysisStep(STEP_CHANGE_DETECTION, priority=1)]

    if intent == INTENT_COMPARISON:
        return [
            AnalysisStep(STEP_REGION_ANALYSIS, priority=1),
            AnalysisStep(STEP_LAND_COVER, priority=2),
        ]

    if intent == INTENT_WATER_BODY:
        steps = [AnalysisStep(STEP_WATER_DETECTION, targets=targets, priority=1)]
        if qi.requires_area_calculation:
            steps.append(AnalysisStep(STEP_AREA_CALCULATION, targets=["water_body"], priority=2))
        return steps

    if intent == INTENT_VEGETATION:
        steps = [AnalysisStep(STEP_VEGETATION_ANALYSIS, targets=targets, priority=1)]
        if qi.requires_area_calculation:
            steps.append(AnalysisStep(STEP_AREA_CALCULATION, targets=["vegetation"], priority=2))
        return steps

    if intent == INTENT_AGRICULTURAL:
        steps = [
            AnalysisStep(STEP_VEGETATION_ANALYSIS, targets=targets, priority=1),
            AnalysisStep(STEP_LAND_COVER, priority=2),
        ]
        if qi.requires_area_calculation:
            steps.append(AnalysisStep(STEP_AREA_CALCULATION, targets=["agricultural_field"], priority=3))
        return steps

    if intent == INTENT_ROAD_NETWORK:
        steps = [AnalysisStep(STEP_ROAD_DETECTION, targets=targets, priority=1)]
        if qi.requires_area_calculation:
            steps.append(AnalysisStep(STEP_AREA_CALCULATION, targets=["road"], priority=2))
        return steps

    if intent == INTENT_INFRASTRUCTURE:
        return [
            AnalysisStep(STEP_OBJECT_DETECTION, targets=targets or ["building","road","bridge"], priority=1),
            AnalysisStep(STEP_LAND_COVER, priority=2),
        ]

    if intent == INTENT_COORDINATE_REQUEST:
        return [
            AnalysisStep(STEP_OBJECT_DETECTION, targets=targets, priority=1),
            AnalysisStep(STEP_COORDINATE_EXTRACT, priority=2),
        ]

    if intent == INTENT_ANOMALY_DETECTION:
        return [
            AnalysisStep(STEP_SCENE_DESCRIPTION, priority=1),
            AnalysisStep(STEP_OBJECT_DETECTION, targets=targets, priority=2),
        ]

    # Default fallback
    return [
        AnalysisStep(STEP_SCENE_DESCRIPTION, priority=1),
        AnalysisStep(STEP_LAND_COVER, priority=2),
    ]


class AnalysisPlanner:
    """
    Maps a QueryIntent to an executable AnalysisPlan.
    Handles both single-intent and multi-task queries.
    """

    def build_plan(self, qi: QueryIntent) -> AnalysisPlan:
        """
        Returns an AnalysisPlan with ordered steps.
        For MULTI_TASK queries, merges steps from all sub-intents.
        """
        try:
            return self._do_build(qi)
        except Exception as e:
            logger.error(f"AnalysisPlanner error: {e}")
            # Safe fallback
            return AnalysisPlan(
                intent=qi,
                steps=[
                    AnalysisStep(STEP_SCENE_DESCRIPTION, priority=1),
                    AnalysisStep(STEP_LAND_COVER, priority=2),
                ],
                is_multi_step=False,
            )

    def _do_build(self, qi: QueryIntent) -> AnalysisPlan:
        if qi.intent == INTENT_MULTI_TASK and qi.sub_intents:
            # Build steps for each sub-intent and merge, removing duplicates
            all_steps = []
            seen_types = set()
            for sub in qi.sub_intents:
                sub_steps = _steps_for_intent(sub, qi)
                for step in sub_steps:
                    if step.step_type not in seen_types:
                        all_steps.append(step)
                        seen_types.add(step.step_type)
            all_steps.sort(key=lambda s: s.priority)
            plan = AnalysisPlan(intent=qi, steps=all_steps, is_multi_step=len(all_steps) > 1)
        else:
            steps = _steps_for_intent(qi.intent, qi)
            steps.sort(key=lambda s: s.priority)
            plan = AnalysisPlan(intent=qi, steps=steps, is_multi_step=len(steps) > 1)

        logger.info(
            f"AnalysisPlan: intent={qi.intent}, "
            f"steps={[s.step_type for s in plan.steps]}"
        )
        return plan


# Singleton
analysis_planner = AnalysisPlanner()
