"""
SatQuery AI — Query Understanding Layer
Parses a natural-language query into a structured QueryIntent.

This module uses semantic pattern matching (NOT keyword lookup) to:
  1. Classify the primary intent (one of 25 task types)
  2. Extract target objects/features
  3. Detect spatial, temporal and quantitative constraints
  4. Determine which vision capability is required
  5. Handle multi-task queries

Design principles:
  - Never silently discard a target
  - Multi-task queries become INTENT_MULTI_TASK with sub_intents list
  - Ambiguous queries receive low confidence + SCENE_UNDERSTANDING fallback
  - "Locate buildings" → OBJECT_LOCALIZATION  (NOT land-cover)
  - "Calculate built-up area" → AREA_QUANTIFICATION + SEMANTIC_SEGMENTATION
"""
import re
import logging
from typing import List, Tuple, Optional

from data_models import (
    QueryIntent,
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
# TARGET VOCABULARY
# Maps canonical names → regex patterns to extract from query
# ─────────────────────────────────────────────────────────────────────────────

TARGET_PATTERNS: List[Tuple[str, str]] = [
    # Infrastructure / built environment
    ("building",          r'\b(building|buildings|structure|structures|construction|constructions)\b'),
    ("storage_facility",  r'\b(storage|warehouse|warehouses|shed|sheds|depot|depots|silo|silos|tank|tanks|facility|facilities)\b'),
    ("industrial_area",   r'\b(industrial|industry|factory|factories|plant|plants|mill|mills)\b'),
    ("residential_area",  r'\b(residential|housing|homes|houses|neighborhood|neighbourhood|colony|colonies)\b'),
    ("military",          r'\b(military|defence|defense|barracks|installation|base|bunker)\b'),
    ("airport",           r'\b(airport|airports|runway|runways|airstrip|airstrips|airfield|airfields|hangar|hangars|aviation)\b'),
    ("bridge",            r'\b(bridge|bridges|overpass|overpasses|viaduct|viaducts)\b'),
    ("port",              r'\b(port|ports|harbor|harbour|dock|docks|pier|piers|jetty|berth|wharf)\b'),
    ("construction_site", r'\b(construction site|construction sites|excavation|development site)\b'),
    # Roads and networks
    ("road",              r'\b(road|roads|highway|highways|motorway|motorways|street|streets|path|paths|transport network|network)\b'),
    ("vehicle",           r'\b(vehicle|vehicles|car|cars|truck|trucks|bus|buses)\b'),
    ("railway",           r'\b(railway|railways|railroad|track|tracks|train)\b'),
    # Water
    ("water_body",        r'\b(water|water bod(?:y|ies)|river|rivers|lake|lakes|reservoir|reservoirs|pond|ponds|ocean|sea|bay|creek|drainage|canal|canals|wetland|wetlands)\b'),
    ("coastline",         r'\b(coast(?:line)?|shore|shoreline|beach)\b'),
    # Vegetation / agriculture
    ("vegetation",        r'\b(vegetation|green(?:ery)?|canopy|plant|plants|trees?)\b'),
    ("forest",            r'\b(forest|forests|woodland|woodlands|jungle|mangrove|mangroves)\b'),
    ("agricultural_field",r'\b(agricultural?|agriculture|farm|farms|field|fields|crop|crops|cropland|paddy|plantation|orchard)\b'),
    # Land cover classes
    ("urban_area",        r'\b(urban|city|cities|town|towns|settlement|settlements|built.?up)\b'),
    ("bare_soil",         r'\b(bare\s+soil|rocky|rocky\s+terrain|desert|arid|mud|mudflat)\b'),
    ("land",              r'\b(land(?:mass)?|terrain|ground|earth|soil|peninsula|continent|mainland|island)\b'),
    # Special
    ("cloud",             r'\b(cloud|clouds|cloudy|haze|fog|overcast)\b'),
    ("ship",              r'\b(ship|ships|vessel|vessels|boat|boats|tanker|tankers|cargo|barge|ferry)\b'),
]

# ─────────────────────────────────────────────────────────────────────────────
# INTENT SIGNAL PATTERNS
# Each entry: (intent_constant, [action patterns], [object/topic patterns])
# An intent fires if ANY action AND ANY topic matches (or topic list is empty)
# ─────────────────────────────────────────────────────────────────────────────

# Localization verbs — strongly imply position-finding
_LOCATE_VERBS = r'\b(locate|find|where|spot|pinpoint|identify|show me|mark|highlight|detect|map out|show|look for|search for)\b'

# Counting verbs
_COUNT_VERBS   = r'\b(count|how many|number of|enumerate|tally|quantify)\b'

# Area/calculation verbs
_AREA_VERBS    = r'\b(calculate|compute|measure|estimate|how much|what percentage|what area|what proportion|coverage of)\b'

# Description verbs
_DESCRIBE_VERBS = r'\b(describe|what is|what are|explain|overview|tell me about|summarize|what can you see|what do you see|what is present|what objects|what features|what is visible|give.*overview|scene)\b'

# Comparison verbs
_COMPARE_VERBS  = r'\b(compare|contrast|difference between|vs|versus|east.*west|north.*south|left.*right|upper.*lower|before.*after|changed|change between)\b'

# Direction triggers
_DIRECTION_WORDS = r'\b(north(?:ern)?|south(?:ern)?|east(?:ern)?|west(?:ern)?|upper|lower|upper.?left|upper.?right|lower.?left|lower.?right|center|central|middle|northwest|northeast|southwest|southeast)\b'

# Land-cover specific
_LANDCOVER_WORDS = r'\b(land\s*cover|lulc|classification|breakdown|classes|percentage|distribution|proportion|statistics|what\s+type|dominant|cover\s+type)\b'


def _extract_targets(q: str) -> List[str]:
    """Extract all target entities from query, preserving order of first mention."""
    found = []
    for canonical, pattern in TARGET_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            if canonical not in found:
                found.append(canonical)
    return found


def _has_localization_intent(q: str) -> bool:
    """True if query asks WHERE something IS (position/localization)."""
    return bool(re.search(_LOCATE_VERBS, q, re.IGNORECASE))


def _has_counting_intent(q: str) -> bool:
    return bool(re.search(_COUNT_VERBS, q, re.IGNORECASE))


def _has_area_intent(q: str) -> bool:
    return bool(re.search(_AREA_VERBS, q, re.IGNORECASE))


def _has_description_intent(q: str) -> bool:
    return bool(re.search(_DESCRIBE_VERBS, q, re.IGNORECASE))


def _has_comparison_intent(q: str) -> bool:
    return bool(re.search(_COMPARE_VERBS, q, re.IGNORECASE))


def _has_direction_intent(q: str) -> bool:
    return bool(re.search(_DIRECTION_WORDS, q, re.IGNORECASE))


def _has_landcover_intent(q: str) -> bool:
    return bool(re.search(_LANDCOVER_WORDS, q, re.IGNORECASE))


def _has_spatial_relation(q: str) -> bool:
    """True if query expresses a spatial relationship between objects."""
    return bool(re.search(
        r'\b(near|close to|adjacent|next to|beside|around|within|inside|outside|along|between)\b',
        q, re.IGNORECASE
    ))


def _has_change_intent(q: str) -> bool:
    return bool(re.search(
        r'\b(change|changed|changes|before.*after|temporal|different dates|expansion|growth|new|loss|deforestation|urban.*expansion)\b',
        q, re.IGNORECASE
    ))


def _extract_spatial_constraint(q: str) -> Optional[str]:
    """Extract directional or proximity spatial constraint from query."""
    m = re.search(
        r'\b(?:in|at|near|within|around|along)?\s*(north(?:ern)?|south(?:ern)?|east(?:ern)?|west(?:ern)?|'
        r'upper|lower|upper.?left|upper.?right|lower.?left|lower.?right|center|central|middle|'
        r'northwest|northeast|southwest|southeast)\s*(?:part|region|area|portion|section|half)?\b',
        q, re.IGNORECASE
    )
    if m:
        return m.group(0).strip()

    m2 = re.search(
        r'\b(near|close to|adjacent to|next to|beside|within \d+ (?:km|meters?|pixels?))\b.*?(\w+)',
        q, re.IGNORECASE
    )
    if m2:
        return m2.group(0).strip()
    return None


def _classify_primary_intent(
    q: str,
    targets: List[str],
    has_locate: bool,
    has_count: bool,
    has_area: bool,
    has_describe: bool,
    has_compare: bool,
    has_direction: bool,
    has_landcover: bool,
    has_relation: bool,
    has_change: bool,
) -> Tuple[str, float, List[str]]:
    """
    Returns (primary_intent, confidence, sub_intents).
    Priority order: explicit counting > explicit area > localization > description > land-cover > scene.
    """
    sub_intents = []

    # Change detection (high priority — usually multi-image)
    if has_change and not has_locate:
        return INTENT_CHANGE_DETECTION, 0.92, []

    # Counting
    if has_count:
        sub_intents = []
        if has_area:
            sub_intents.append(INTENT_AREA_QUANTIFICATION)
        return INTENT_OBJECT_COUNTING, 0.95, sub_intents

    # Area/quantification without localization
    if has_area and not has_locate:
        if targets and any(t in ["water_body","agricultural_field","forest","vegetation","urban_area","building"] for t in targets):
            sub_intents = [INTENT_SEMANTIC_SEGMENTATION]
            return INTENT_AREA_QUANTIFICATION, 0.93, sub_intents
        if has_landcover:
            return INTENT_LAND_COVER, 0.92, []
        sub_intents = [INTENT_SEMANTIC_SEGMENTATION]
        return INTENT_AREA_QUANTIFICATION, 0.88, sub_intents

    # Explicit land-cover classification request
    if has_landcover and not has_locate:
        return INTENT_LAND_COVER, 0.93, []

    # Comparison
    if has_compare:
        return INTENT_COMPARISON, 0.90, []

    # Spatial relationship (near, adjacent, inside)
    if has_relation and has_locate:
        return INTENT_SPATIAL_RELATIONSHIP, 0.91, []

    # Direction/region query
    if has_direction and not has_locate and not targets:
        return INTENT_REGION_DIRECTION, 0.87, []

    # Object localization — the core fix
    if has_locate and targets:
        sub_intents = []
        if has_area:
            sub_intents.append(INTENT_AREA_QUANTIFICATION)
        if has_direction:
            sub_intents.append(INTENT_REGION_DIRECTION)

        # Water body specific
        if all(t in ["water_body","coastline","river","lake"] for t in targets):
            return INTENT_WATER_BODY, 0.94, sub_intents
        # Road network specific
        if all(t in ["road","railway","bridge"] for t in targets):
            return INTENT_ROAD_NETWORK, 0.93, sub_intents
        # Vegetation specific
        if all(t in ["vegetation","forest","agricultural_field"] for t in targets):
            if "agricultural_field" in targets:
                return INTENT_AGRICULTURAL, 0.92, sub_intents
            return INTENT_VEGETATION, 0.91, sub_intents
        # General object localization
        return INTENT_OBJECT_LOCALIZATION, 0.94, sub_intents

    # Localization without explicit targets — direction query
    if has_locate and has_direction:
        return INTENT_REGION_DIRECTION, 0.87, []

    # Localization with no known targets — general detection
    if has_locate:
        return INTENT_OBJECT_DETECTION, 0.82, []

    # Water body queries
    if targets and all(t in ["water_body","coastline"] for t in targets):
        if has_area:
            return INTENT_AREA_QUANTIFICATION, 0.91, [INTENT_WATER_BODY]
        return INTENT_WATER_BODY, 0.90, []

    # Vegetation queries
    if targets and all(t in ["vegetation","forest","agricultural_field"] for t in targets):
        if has_area:
            return INTENT_AREA_QUANTIFICATION, 0.91, [INTENT_VEGETATION]
        return INTENT_VEGETATION, 0.89, []

    # Description / open-ended scene
    if has_describe and not targets:
        return INTENT_SCENE_UNDERSTANDING, 0.88, []
    if has_describe:
        return INTENT_IMAGE_DESCRIPTION, 0.85, []

    # Multiple targets with localization verbs absent → land cover more likely
    if targets and has_landcover:
        return INTENT_LAND_COVER, 0.88, []

    # Fallback: if targets known but no clear intent signal, treat as scene understanding
    if targets:
        return INTENT_SCENE_UNDERSTANDING, 0.72, []

    return INTENT_SCENE_UNDERSTANDING, 0.65, []


def _determine_vision_capability(intent: str, targets: List[str]) -> str:
    """Returns the required vision capability string."""
    localization_intents = {
        INTENT_OBJECT_DETECTION, INTENT_OBJECT_LOCALIZATION,
        INTENT_OBJECT_COUNTING, INTENT_SPATIAL_RELATIONSHIP,
        INTENT_INFRASTRUCTURE, INTENT_ROAD_NETWORK,
    }
    segmentation_intents = {
        INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION, INTENT_AREA_QUANTIFICATION,
        INTENT_WATER_BODY, INTENT_VEGETATION, INTENT_AGRICULTURAL,
    }
    if intent in localization_intents:
        return "detection"
    if intent in segmentation_intents:
        return "segmentation"
    if intent in {INTENT_CHANGE_DETECTION, INTENT_TEMPORAL}:
        return "change"
    if intent in {INTENT_IMAGE_DESCRIPTION, INTENT_SCENE_UNDERSTANDING, INTENT_COMPARISON}:
        return "vlm"
    return "general"


class QueryUnderstanding:
    """
    Parses a natural-language remote-sensing query into a structured QueryIntent.
    Uses semantic pattern matching across 25 intent categories.
    """

    def analyze(self, query: str, num_images: int = 1) -> QueryIntent:
        """
        Main entry point. Returns a QueryIntent for any query string.
        Never raises — returns SCENE_UNDERSTANDING with low confidence on failure.
        """
        try:
            return self._do_analyze(query, num_images)
        except Exception as e:
            logger.error(f"QueryUnderstanding error: {e}")
            return QueryIntent(
                raw_query=query,
                intent=INTENT_SCENE_UNDERSTANDING,
                confidence=0.40,
                limitations=["Query understanding encountered an error; defaulting to scene description."]
            )

    def _do_analyze(self, query: str, num_images: int) -> QueryIntent:
        q = query.strip()
        q_lower = q.lower()

        # ── Target extraction ──────────────────────────────────────────────
        targets = _extract_targets(q_lower)

        # ── Signal detection ───────────────────────────────────────────────
        has_locate   = _has_localization_intent(q_lower)
        has_count    = _has_counting_intent(q_lower)
        has_area     = _has_area_intent(q_lower)
        has_describe = _has_description_intent(q_lower)
        has_compare  = _has_comparison_intent(q_lower)
        has_direction= _has_direction_intent(q_lower)
        has_landcover= _has_landcover_intent(q_lower)
        has_relation = _has_spatial_relation(q_lower)
        has_change   = _has_change_intent(q_lower)

        # Multi-image always → change or cross-modal
        if num_images >= 2:
            if has_change:
                return QueryIntent(
                    raw_query=q,
                    intent=INTENT_CHANGE_DETECTION,
                    targets=targets,
                    operation="compare",
                    requires_visual_overlay=True,
                    confidence=0.95,
                    required_vision_capability="change"
                )
            return QueryIntent(
                raw_query=q,
                intent=INTENT_COMPARISON,
                targets=targets,
                operation="compare",
                requires_comparison=True,
                requires_visual_overlay=True,
                confidence=0.88,
                required_vision_capability="change"
            )

        # ── Primary intent classification ──────────────────────────────────
        primary_intent, conf, sub_intents = _classify_primary_intent(
            q_lower, targets, has_locate, has_count, has_area, has_describe,
            has_compare, has_direction, has_landcover, has_relation, has_change
        )

        # ── Multi-task promotion ───────────────────────────────────────────
        if sub_intents:
            primary_intent = INTENT_MULTI_TASK
            sub_intents = [_classify_primary_intent(
                q_lower, targets, has_locate, has_count, has_area, has_describe,
                has_compare, has_direction, has_landcover, has_relation, has_change
            )[0]] + sub_intents
            conf = min(conf, 0.88)

        # ── Determine operation verb ───────────────────────────────────────
        if has_count:
            operation = "count"
        elif has_area:
            operation = "calculate"
        elif has_locate:
            operation = "locate"
        elif has_describe:
            operation = "describe"
        elif has_compare:
            operation = "compare"
        else:
            operation = "analyze"

        # ── Spatial constraint ─────────────────────────────────────────────
        spatial_constraint = _extract_spatial_constraint(q_lower)

        # ── Output requirements ────────────────────────────────────────────
        requires_coords = bool(re.search(
            r'\b(coordinate|coordinates|lat|lon|latitude|longitude|gps|georef|geoloc|where exactly)\b',
            q_lower
        ))
        requires_count = has_count
        requires_area  = has_area
        requires_overlay = True  # almost always useful

        vision_cap = _determine_vision_capability(primary_intent, targets)

        intent = QueryIntent(
            raw_query=q,
            intent=primary_intent,
            sub_intents=sub_intents,
            targets=targets,
            operation=operation,
            spatial_constraint=spatial_constraint,
            temporal_constraint=None,
            quantitative=has_count or has_area,
            requires_coordinates=requires_coords,
            requires_visual_overlay=requires_overlay,
            requires_area_calculation=requires_area,
            requires_count=requires_count,
            requires_comparison=has_compare,
            confidence=conf,
            required_vision_capability=vision_cap,
        )

        logger.info(
            f"QueryUnderstanding: '{q[:60]}' → intent={primary_intent}, "
            f"targets={targets}, confidence={conf:.2f}"
        )
        return intent


# Singleton
query_understanding = QueryUnderstanding()
