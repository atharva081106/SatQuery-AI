"""
SatQuery AI — Query Routing Test Suite
Tests that every query type routes to the correct analysis pipeline.
Run: python -m pytest test_query_routing.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from query_understanding import query_understanding
from analysis_planner import analysis_planner
from data_models import (
    INTENT_OBJECT_LOCALIZATION, INTENT_OBJECT_DETECTION, INTENT_OBJECT_COUNTING,
    INTENT_LAND_COVER, INTENT_AREA_QUANTIFICATION, INTENT_SEMANTIC_SEGMENTATION,
    INTENT_WATER_BODY, INTENT_VEGETATION, INTENT_AGRICULTURAL, INTENT_ROAD_NETWORK,
    INTENT_SPATIAL_RELATIONSHIP, INTENT_REGION_DIRECTION, INTENT_SCENE_UNDERSTANDING,
    INTENT_CHANGE_DETECTION, INTENT_COMPARISON, INTENT_IMAGE_DESCRIPTION, INTENT_MULTI_TASK,
    INTENT_TEMPORAL,
)
from analysis_planner import STEP_OBJECT_DETECTION, STEP_LAND_COVER, STEP_WATER_DETECTION


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 28 REGRESSION TEST — this must never regress
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionBuildingLocalization:
    """Critical regression: 'Locate buildings' must NEVER return land-cover stats as primary answer."""

    def test_locate_buildings_not_land_cover(self):
        qi = query_understanding.analyze("Locate buildings and storage facilities")
        # Must NOT be land cover
        assert qi.intent != INTENT_LAND_COVER, (
            f"REGRESSION: 'Locate buildings' mapped to LAND_COVER (was {qi.intent}). "
            "This is the core bug that must be fixed."
        )
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK), (
            f"Expected OBJECT_LOCALIZATION or MULTI_TASK, got {qi.intent}"
        )

    def test_locate_buildings_targets_extracted(self):
        qi = query_understanding.analyze("Locate buildings and storage facilities")
        assert "building" in qi.targets, f"'building' not in targets: {qi.targets}"
        assert "storage_facility" in qi.targets, f"'storage_facility' not in targets: {qi.targets}"

    def test_locate_buildings_plan_uses_detection(self):
        qi = query_understanding.analyze("Locate buildings and storage facilities")
        plan = analysis_planner.build_plan(qi)
        step_types = [s.step_type for s in plan.steps]
        assert STEP_OBJECT_DETECTION in step_types, (
            f"Plan must include object_detection step. Steps: {step_types}"
        )
        # Land-cover MUST NOT be the only or primary step for this query
        if step_types[0] == STEP_LAND_COVER:
            pytest.fail(
                f"REGRESSION: First plan step is land_cover for object localization query. "
                f"Steps: {step_types}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT LOCALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectLocalization:

    def test_locate_buildings(self):
        qi = query_understanding.analyze("Locate buildings")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "building" in qi.targets

    def test_find_storage_facilities(self):
        qi = query_understanding.analyze("Find all storage facilities")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "storage_facility" in qi.targets

    def test_find_bridges(self):
        qi = query_understanding.analyze("Find bridges")
        # Bridges are part of road network — both ROAD_NETWORK and OBJECT_LOCALIZATION are correct
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK, INTENT_ROAD_NETWORK)
        assert "bridge" in qi.targets

    def test_detect_airports(self):
        qi = query_understanding.analyze("Detect airports")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_OBJECT_DETECTION, INTENT_MULTI_TASK)
        assert "airport" in qi.targets

    def test_military_infrastructure(self):
        qi = query_understanding.analyze("Locate potential military infrastructure")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_OBJECT_DETECTION, INTENT_MULTI_TASK)

    def test_find_warehouses(self):
        qi = query_understanding.analyze("Find warehouses")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "storage_facility" in qi.targets or "building" in qi.targets

    def test_locate_vehicles(self):
        qi = query_understanding.analyze("Locate vehicles")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_OBJECT_DETECTION, INTENT_MULTI_TASK)
        assert "vehicle" in qi.targets

    def test_find_construction_sites(self):
        qi = query_understanding.analyze("Find construction sites")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "construction_site" in qi.targets

    def test_identify_residential(self):
        qi = query_understanding.analyze("Where are the residential areas?")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "residential_area" in qi.targets


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT COUNTING
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectCounting:

    def test_count_buildings(self):
        qi = query_understanding.analyze("How many buildings are visible?")
        assert qi.intent == INTENT_OBJECT_COUNTING
        assert qi.requires_count

    def test_count_buildings_variant(self):
        qi = query_understanding.analyze("Count buildings")
        assert qi.intent == INTENT_OBJECT_COUNTING

    def test_number_of_ships(self):
        qi = query_understanding.analyze("What is the number of ships?")
        assert qi.intent == INTENT_OBJECT_COUNTING
        assert "ship" in qi.targets


# ─────────────────────────────────────────────────────────────────────────────
# LAND COVER / AREA
# ─────────────────────────────────────────────────────────────────────────────

class TestLandCover:

    def test_land_cover_classification(self):
        qi = query_understanding.analyze("What are the dominant land-cover classes?")
        assert qi.intent in (INTENT_LAND_COVER, INTENT_SCENE_UNDERSTANDING)

    def test_agricultural_percentage(self):
        qi = query_understanding.analyze("What percentage is agricultural land?")
        assert qi.intent in (INTENT_AREA_QUANTIFICATION, INTENT_LAND_COVER, INTENT_MULTI_TASK)
        assert qi.quantitative

    def test_built_up_area(self):
        qi = query_understanding.analyze("Calculate the built-up area")
        assert qi.intent in (INTENT_AREA_QUANTIFICATION, INTENT_LAND_COVER, INTENT_MULTI_TASK)
        assert qi.requires_area_calculation

    def test_distribution_query(self):
        qi = query_understanding.analyze("What is the land cover breakdown?")
        assert qi.intent in (INTENT_LAND_COVER, INTENT_SCENE_UNDERSTANDING)

    def test_built_area_plan_uses_land_cover(self):
        qi = query_understanding.analyze("Calculate the built-up area")
        plan = analysis_planner.build_plan(qi)
        step_types = [s.step_type for s in plan.steps]
        assert STEP_LAND_COVER in step_types, (
            f"Area calculation must use land_cover step. Steps: {step_types}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# WATER BODY
# ─────────────────────────────────────────────────────────────────────────────

class TestWaterBody:

    def test_find_water_bodies(self):
        qi = query_understanding.analyze("Find water bodies")
        assert qi.intent in (INTENT_WATER_BODY, INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "water_body" in qi.targets

    def test_identify_rivers(self):
        qi = query_understanding.analyze("Identify rivers and water bodies")
        assert qi.intent in (INTENT_WATER_BODY, INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "water_body" in qi.targets

    def test_locate_rivers(self):
        qi = query_understanding.analyze("Locate rivers")
        assert qi.intent in (INTENT_WATER_BODY, INTENT_OBJECT_LOCALIZATION)


# ─────────────────────────────────────────────────────────────────────────────
# VEGETATION
# ─────────────────────────────────────────────────────────────────────────────

class TestVegetation:

    def test_dense_vegetation(self):
        qi = query_understanding.analyze("Identify dense vegetation")
        assert qi.intent in (INTENT_VEGETATION, INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "vegetation" in qi.targets or "forest" in qi.targets

    def test_agricultural_fields(self):
        qi = query_understanding.analyze("Locate agricultural fields")
        assert qi.intent in (INTENT_AGRICULTURAL, INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "agricultural_field" in qi.targets


# ─────────────────────────────────────────────────────────────────────────────
# ROAD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class TestRoads:

    def test_find_roads(self):
        qi = query_understanding.analyze("Find all roads")
        assert qi.intent in (INTENT_ROAD_NETWORK, INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "road" in qi.targets


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-OBJECT
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiObject:

    def test_multi_object_query(self):
        qi = query_understanding.analyze("Locate buildings, roads and water")
        assert "building" in qi.targets
        assert "road" in qi.targets
        assert "water_body" in qi.targets

    def test_multi_object_not_land_cover(self):
        qi = query_understanding.analyze("Locate buildings, roads and water bodies")
        assert qi.intent != INTENT_LAND_COVER, (
            "Multi-object localization must not be routed to land-cover classification"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL RELATIONSHIP
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialRelationship:

    def test_buildings_near_roads(self):
        qi = query_understanding.analyze("Find buildings near roads")
        assert qi.intent in (INTENT_SPATIAL_RELATIONSHIP, INTENT_MULTI_TASK)
        assert "building" in qi.targets
        assert "road" in qi.targets


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTION / REGION
# ─────────────────────────────────────────────────────────────────────────────

class TestDirection:

    def test_northern_query(self):
        qi = query_understanding.analyze("What is in the northern part?")
        assert qi.intent in (INTENT_REGION_DIRECTION, INTENT_SCENE_UNDERSTANDING)

    def test_eastern_buildings(self):
        qi = query_understanding.analyze("Find buildings in the eastern region")
        assert qi.intent in (INTENT_OBJECT_LOCALIZATION, INTENT_REGION_DIRECTION, INTENT_MULTI_TASK)

    def test_upper_right(self):
        qi = query_understanding.analyze("What is in the upper-right region?")
        assert qi.intent in (INTENT_REGION_DIRECTION, INTENT_SCENE_UNDERSTANDING)

    def test_spatial_constraint_extracted(self):
        qi = query_understanding.analyze("Find buildings in the northern area")
        assert qi.spatial_constraint is not None, "Spatial constraint should be extracted"


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

class TestComparison:

    def test_east_west_comparison(self):
        qi = query_understanding.analyze("Compare the eastern and western areas")
        assert qi.intent in (INTENT_COMPARISON, INTENT_REGION_DIRECTION)

    def test_two_image_change(self):
        qi = query_understanding.analyze("Find changes between these two images", num_images=2)
        assert qi.intent == INTENT_CHANGE_DETECTION

    def test_urban_expansion(self):
        qi = query_understanding.analyze("Is there urban expansion?")
        assert qi.intent in (INTENT_CHANGE_DETECTION, INTENT_SCENE_UNDERSTANDING, INTENT_MULTI_TASK)


# ─────────────────────────────────────────────────────────────────────────────
# OPEN-ENDED / SCENE
# ─────────────────────────────────────────────────────────────────────────────

class TestSceneUnderstanding:

    def test_what_is_visible(self):
        qi = query_understanding.analyze("What objects are visible in this image?")
        assert qi.intent in (INTENT_SCENE_UNDERSTANDING, INTENT_IMAGE_DESCRIPTION, INTENT_OBJECT_DETECTION)

    def test_describe_image(self):
        qi = query_understanding.analyze("Describe this satellite image.")
        assert qi.intent in (INTENT_SCENE_UNDERSTANDING, INTENT_IMAGE_DESCRIPTION)

    def test_general_overview(self):
        qi = query_understanding.analyze("Give me an overview of this scene")
        assert qi.intent in (INTENT_SCENE_UNDERSTANDING, INTENT_IMAGE_DESCRIPTION)

    def test_ambiguous(self):
        qi = query_understanding.analyze("What important features are present?")
        assert qi.intent in (INTENT_SCENE_UNDERSTANDING, INTENT_OBJECT_DETECTION)
        assert qi.confidence < 0.95  # Should not be overconfident on ambiguous query


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidence:

    def test_clear_localization_high_confidence(self):
        qi = query_understanding.analyze("Locate buildings")
        assert qi.confidence >= 0.85

    def test_ambiguous_lower_confidence(self):
        qi = query_understanding.analyze("interesting things here")
        assert qi.confidence <= 0.80

    def test_confidence_in_range(self):
        queries = [
            "Locate buildings", "Find roads", "How many buildings?",
            "What is this?", "Calculate built area", "Find water bodies"
        ]
        for q in queries:
            qi = query_understanding.analyze(q)
            assert 0.0 <= qi.confidence <= 1.0, (
                f"Confidence out of range for '{q}': {qi.confidence}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# PLAN CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanCorrectness:

    def test_count_plan_has_count_step(self):
        from analysis_planner import STEP_COUNT_OBJECTS
        qi = query_understanding.analyze("How many buildings are there?")
        plan = analysis_planner.build_plan(qi)
        step_types = [s.step_type for s in plan.steps]
        assert STEP_COUNT_OBJECTS in step_types, f"Count query missing count step. Steps: {step_types}"

    def test_spatial_relation_plan(self):
        from analysis_planner import STEP_SPATIAL_RELATION
        qi = query_understanding.analyze("Find buildings near roads")
        plan = analysis_planner.build_plan(qi)
        step_types = [s.step_type for s in plan.steps]
        assert STEP_OBJECT_DETECTION in step_types, f"Steps: {step_types}"

    def test_water_query_not_object_detection_primary(self):
        qi = query_understanding.analyze("How much water is there?")
        plan = analysis_planner.build_plan(qi)
        # For area queries about water, first step should be water or land-cover
        step_types = [s.step_type for s in plan.steps]
        assert step_types[0] in (STEP_WATER_DETECTION, STEP_LAND_COVER), (
            f"Water area query first step should be water/land-cover, got {step_types}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# TARGET 5 USER QUERIES SPECIFIC SUITE
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetFiveQuestions:
    """Specific tests for the 5 target user queries to ensure 100% routing & planning precision."""

    def test_query_1_describe_landcover_and_objects(self):
        # 'Describe the land-cover and major objects visible in this image.'
        q = "Describe the land-cover and major objects visible in this image."
        qi = query_understanding.analyze(q)
        assert qi.intent in (INTENT_MULTI_TASK, INTENT_SCENE_UNDERSTANDING, INTENT_LAND_COVER)
        plan = analysis_planner.build_plan(qi)
        step_types = [s.step_type for s in plan.steps]
        assert STEP_LAND_COVER in step_types or STEP_SCENE_DESCRIPTION in step_types
        assert STEP_OBJECT_DETECTION in step_types

    def test_query_2_highlight_water_body(self):
        # 'Highlight the water body referred to in the query.'
        q = "Highlight the water body referred to in the query."
        qi = query_understanding.analyze(q)
        assert qi.intent in (INTENT_WATER_BODY, INTENT_OBJECT_LOCALIZATION, INTENT_MULTI_TASK)
        assert "water_body" in qi.targets
        plan = analysis_planner.build_plan(qi)
        step_types = [s.step_type for s in plan.steps]
        assert STEP_WATER_DETECTION in step_types or STEP_OBJECT_DETECTION in step_types

    def test_query_3_what_changed_and_where(self):
        # 'What changed between these two dates, and where did the change occur?'
        q = "What changed between these two dates, and where did the change occur?"
        qi = query_understanding.analyze(q, num_images=2)
        assert qi.intent in (INTENT_CHANGE_DETECTION, INTENT_COMPARISON)
        assert qi.required_vision_capability == "change"

    def test_query_4_optical_and_sar_fusion(self):
        # 'Use the optical and SAR images together to identify built-up and water-covered regions.'
        q = "Use the optical and SAR images together to identify built-up and water-covered regions."
        qi = query_understanding.analyze(q, num_images=2)
        assert "water_body" in qi.targets
        assert any(t in ("urban_area", "building") for t in qi.targets)
        assert qi.required_vision_capability == "change"

    def test_query_5_built_up_area_trend(self):
        # 'Has the built-up area increased, decreased, or remained unchanged?'
        q = "Has the built-up area increased, decreased, or remained unchanged?"
        qi = query_understanding.analyze(q, num_images=2)
        assert qi.intent in (INTENT_CHANGE_DETECTION, INTENT_COMPARISON, INTENT_TEMPORAL)
        assert any(t in ("urban_area", "building") for t in qi.targets)


if __name__ == "__main__":
    # Quick smoke test without pytest
    print("Running quick smoke test...")
    tests = [
        ("Locate buildings and storage facilities", INTENT_OBJECT_LOCALIZATION),
        ("How many buildings are visible?", INTENT_OBJECT_COUNTING),
        ("Calculate the built-up area", INTENT_AREA_QUANTIFICATION),
        ("Find water bodies", INTENT_WATER_BODY),
        ("Locate agricultural fields", INTENT_AGRICULTURAL),
        ("Find roads", INTENT_ROAD_NETWORK),
        ("What is visible in this image?", INTENT_SCENE_UNDERSTANDING),
        ("Describe this satellite image", INTENT_SCENE_UNDERSTANDING),
    ]
    passed = 0
    failed = 0
    for query, expected_intent in tests:
        qi = query_understanding.analyze(query)
        ok = qi.intent == expected_intent or (
            expected_intent == INTENT_OBJECT_LOCALIZATION and qi.intent == INTENT_MULTI_TASK
        ) or (
            expected_intent == INTENT_AREA_QUANTIFICATION and qi.intent in (
                INTENT_AREA_QUANTIFICATION, INTENT_LAND_COVER, INTENT_MULTI_TASK
            )
        )
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} '{query[:45]}' -> {qi.intent} (expected {expected_intent})")
    print(f"\n{passed} passed, {failed} failed")
