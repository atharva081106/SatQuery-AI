"""
SatQuery AI — Query Router (AgenticController)
Central orchestration brain:

USER QUERY
     ↓
QueryUnderstanding (intent + targets + constraints)
     ↓
AnalysisPlanner (ordered steps)
     ↓
Step Executor (dispatches to specialist engines)
     ↓
ResultValidator (answer type check)
     ↓
AnswerGenerator (format-adaptive response)
     ↓
USER-FACING RESULT

Preserves all existing CHANGE_ANALYSIS and CROSS_MODAL_EXTRACTION routing.
"""
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

from data_models import (
    QueryIntent, AnalysisResult, AnalysisPlan, AnalysisStep,
    Detection, Segment, Measurement, SpatialRelationship,
    INTENT_OBJECT_DETECTION, INTENT_OBJECT_LOCALIZATION, INTENT_OBJECT_COUNTING,
    INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION, INTENT_AREA_QUANTIFICATION,
    INTENT_SCENE_UNDERSTANDING, INTENT_IMAGE_DESCRIPTION, INTENT_SPATIAL_RELATIONSHIP,
    INTENT_WATER_BODY, INTENT_VEGETATION, INTENT_AGRICULTURAL, INTENT_ROAD_NETWORK,
    INTENT_REGION_DIRECTION, INTENT_CHANGE_DETECTION, INTENT_COMPARISON,
    INTENT_COORDINATE_REQUEST, INTENT_MULTI_TASK,
)
from query_understanding import query_understanding
from analysis_planner import (
    analysis_planner,
    STEP_OBJECT_DETECTION, STEP_LAND_COVER, STEP_WATER_DETECTION,
    STEP_VEGETATION_ANALYSIS, STEP_ROAD_DETECTION, STEP_SCENE_DESCRIPTION,
    STEP_CHANGE_DETECTION, STEP_REGION_ANALYSIS, STEP_AREA_CALCULATION,
    STEP_COUNT_OBJECTS, STEP_SPATIAL_RELATION, STEP_COORDINATE_EXTRACT,
)
from object_detection import detect_objects, generate_detection_overlay, get_detection_limitations
from spatial_reasoning import analyze_proximity, partition_image_regions, describe_distribution
from result_validator import result_validator
from answer_generator import answer_generator
from model_interfaces import (
    decode_satellite_image, _analyze_land_cover, _generate_visual_evidence,
    _calculate_land_cover_percentages
)
from model_registry import registry

logger = logging.getLogger(__name__)


def _land_cover_to_segments(stats: Dict[str, Any]) -> List[Segment]:
    """Convert _analyze_land_cover() stats dict to Segment list."""
    segs = []
    mapping = [
        ("Water Bodies",   stats.get("water_pct", 0), stats.get("water_km2"), stats.get("water_ha")),
        ("Forest",         stats.get("forest_pct", 0), stats.get("forest_km2"), stats.get("forest_ha")),
        ("Cropland",       stats.get("crop_pct", 0), stats.get("crop_km2"), stats.get("crop_ha")),
        ("Built-up Area",  stats.get("built_pct", 0), stats.get("built_km2"), stats.get("built_ha")),
        ("Bare Soil",      stats.get("bare_pct", 0), stats.get("bare_km2"), stats.get("bare_ha")),
        ("Cloud / Haze",   stats.get("cloud_pct", 0), stats.get("cloud_km2"), stats.get("cloud_ha")),
    ]
    for name, pct, km2, ha in mapping:
        if pct is not None and pct > 0.1:
            segs.append(Segment(
                class_name=name,
                pixel_count=0,
                percentage=round(pct, 2),
                area_km2=round(km2, 2) if km2 else None,
                area_ha=round(ha, 0) if ha else None,
            ))
    return segs


def _make_scene_description(stats: Dict[str, Any], model_answer: str = "") -> str:
    """Generate a scene description from land-cover stats."""
    parts = []
    if model_answer and len(model_answer) > 10:
        parts.append(model_answer)

    # Identify dominant features
    ordered = sorted([
        ("built-up area and settlements", stats.get("built_pct", 0)),
        ("water bodies", stats.get("water_pct", 0)),
        ("forest and vegetation", stats.get("veg_total_pct", 0)),
        ("bare / open terrain", stats.get("bare_pct", 0)),
        ("agricultural land", stats.get("crop_pct", 0)),
    ], key=lambda x: -x[1])

    dominant = [f"**{name}** ({pct:.1f}%)" for name, pct in ordered if pct > 5.0]
    if dominant:
        if not parts:
            parts.append(f"The scene shows a mix of {', '.join(dominant)}.")
        else:
            parts.append(f"Key features visible: {', '.join(dominant)}.")

    return " ".join(parts) or "A satellite image has been analysed."


class QueryRouter:
    """
    Central orchestrator. Routes each query through the full pipeline:
    understanding → planning → execution → validation → answer generation.
    """

    def execute_query(
        self,
        query: str,
        images: List[bytes],
        history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point. Returns the full result dict consumed by main.py.
        """
        import hashlib, json, os

        # ── Disk Cache (preserves existing behaviour) ────────────────────────
        CACHE_VERSION = "v5_query_router"
        m = hashlib.md5()
        m.update(CACHE_VERSION.encode())
        m.update(query.encode())
        for img in images:
            m.update(img)
        cache_key = m.hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    logger.info(f"Cache hit for: {query[:50]}")
                    return json.load(f)
            except Exception:
                pass

        try:
            result = self._route(query, images, history or [])
        except Exception as e:
            logger.error(f"QueryRouter.execute_query error: {e}", exc_info=True)
            result = {"status": "error", "message": str(e)}

        # Save to cache
        if result.get("status") == "success":
            try:
                with open(cache_path, "w") as f:
                    json.dump(result, f)
            except Exception as e:
                logger.error(f"Cache write error: {e}")

        return result

    def _route(self, query: str, images: List[bytes], history: list) -> Dict[str, Any]:
        """Core routing logic."""
        num_images = len(images)

        # ── Phase 1: Query Understanding ────────────────────────────────────
        qi: QueryIntent = query_understanding.analyze(query, num_images)
        logger.info(f"Intent: {qi.intent}, targets={qi.targets}, confidence={qi.confidence:.2f}")

        # ── Phase 2: Multi-image → existing specialist models ────────────────
        if num_images >= 2:
            return self._handle_multi_image(query, images, qi, history)

        # ── Phase 3: Build Analysis Plan ─────────────────────────────────────
        plan: AnalysisPlan = analysis_planner.build_plan(qi)
        logger.info(f"Plan steps: {[s.step_type for s in plan.steps]}")

        # ── Phase 4: Decode image ─────────────────────────────────────────────
        cv_img, pil_img, geo_meta = decode_satellite_image(images[0])

        # ── Phase 5: Execute plan steps ───────────────────────────────────────
        analysis_result = self._execute_plan(plan, qi, cv_img, pil_img, geo_meta, images[0])

        # ── Phase 6: Validate ─────────────────────────────────────────────────
        validation = result_validator.validate(qi, analysis_result)

        # ── Phase 7: Generate answer ──────────────────────────────────────────
        final_text = answer_generator.generate(qi, analysis_result, validation)

        # ── Phase 8: Build response dict ──────────────────────────────────────
        return self._build_response(query, qi, analysis_result, validation, final_text, geo_meta)

    # ─────────────────────────────────────────────────────────────────────────
    # Plan Executor
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_plan(
        self,
        plan: AnalysisPlan,
        qi: QueryIntent,
        cv_img,
        pil_img,
        geo_meta,
        image_bytes: bytes
    ) -> AnalysisResult:
        """Execute each analysis step in the plan and merge results."""
        from ml_models import ml_manager

        detections: List[Detection] = []
        segments: List[Segment] = []
        measurements: List[Measurement] = []
        spatial_rels: List[SpatialRelationship] = []
        visual_b64: Optional[str] = None
        visual_desc: str = ""
        scene_desc: str = ""
        raw_stats: Optional[Dict] = None
        limitations: List[str] = []
        methods_used: List[str] = []
        confidence = qi.confidence

        for step in plan.steps:
            try:
                if step.step_type == STEP_OBJECT_DETECTION:
                    methods_used.append("object_detection")
                    targets = step.targets or qi.targets or []
                    if targets:
                        new_dets = detect_objects(cv_img, targets, geo_meta)
                        detections.extend(new_dets)
                        lims = get_detection_limitations(targets, new_dets)
                        limitations.extend(l for l in lims if l not in limitations)
                    else:
                        # No specific targets — run general scene analysis
                        step_result = self._general_detection(cv_img, geo_meta)
                        detections.extend(step_result)
                        limitations.append(
                            "No specific targets specified; general structure detection was applied."
                        )

                elif step.step_type in (STEP_LAND_COVER, STEP_WATER_DETECTION, STEP_VEGETATION_ANALYSIS):
                    methods_used.append("land_cover_segmentation")
                    if raw_stats is None:
                        raw_stats = _analyze_land_cover(cv_img, geo_meta)
                    segs = _land_cover_to_segments(raw_stats)
                    # Merge (avoid duplicates)
                    existing_names = {s.class_name for s in segments}
                    segments.extend(s for s in segs if s.class_name not in existing_names)

                elif step.step_type == STEP_ROAD_DETECTION:
                    methods_used.append("road_detection")
                    road_dets = detect_objects(cv_img, ["road"], geo_meta)
                    detections.extend(road_dets)
                    limitations.extend(get_detection_limitations(["road"], road_dets))

                elif step.step_type == STEP_SCENE_DESCRIPTION:
                    methods_used.append("scene_description")
                    if raw_stats is None:
                        raw_stats = _analyze_land_cover(cv_img, geo_meta)
                    # Try VQA model first
                    model_answer = ""
                    try:
                        processor, model, device, model_name = ml_manager.get_vqa_pipeline()
                        if model and processor:
                            import torch
                            prompt = "Describe this satellite image in plain English."
                            inputs = processor(pil_img, prompt, return_tensors="pt").to(device)
                            out = model.generate(**inputs, max_new_tokens=120)
                            model_answer = processor.decode(out[0], skip_special_tokens=True).strip()
                    except Exception:
                        pass
                    scene_desc = _make_scene_description(raw_stats, model_answer)
                    segs = _land_cover_to_segments(raw_stats)
                    existing_names = {s.class_name for s in segments}
                    segments.extend(s for s in segs if s.class_name not in existing_names)

                elif step.step_type == STEP_AREA_CALCULATION:
                    methods_used.append("area_calculation")
                    if raw_stats is None:
                        raw_stats = _analyze_land_cover(cv_img, geo_meta)
                    self._add_area_measurements(measurements, raw_stats, step.targets, geo_meta)

                elif step.step_type == STEP_COUNT_OBJECTS:
                    methods_used.append("count_objects")
                    by_class: Dict[str, int] = {}
                    for d in detections:
                        by_class[d.class_name] = by_class.get(d.class_name, 0) + 1
                    for cls, cnt in by_class.items():
                        measurements.append(Measurement(
                            metric=f"{cls}_count",
                            value=cnt,
                            unit="count",
                            source="object_detection",
                            confidence=0.80,
                        ))

                elif step.step_type == STEP_SPATIAL_RELATION:
                    methods_used.append("spatial_relationship_analysis")
                    if len(qi.targets) >= 2:
                        group_a = [d for d in detections if d.class_name == qi.targets[0]]
                        group_b = [d for d in detections if d.class_name == qi.targets[1]]
                        rels = analyze_proximity(group_a, group_b, max_dist_norm=0.15)
                        spatial_rels.extend(rels)

                elif step.step_type == STEP_REGION_ANALYSIS:
                    methods_used.append("region_analysis")
                    if raw_stats is None:
                        raw_stats = _analyze_land_cover(cv_img, geo_meta)
                    constraint = step.parameters.get("constraint") or qi.spatial_constraint
                    region_scene = self._analyze_region(cv_img, constraint, raw_stats, geo_meta)
                    if region_scene:
                        scene_desc = region_scene
                    segs = _land_cover_to_segments(raw_stats)
                    existing_names = {s.class_name for s in segments}
                    segments.extend(s for s in segs if s.class_name not in existing_names)

                elif step.step_type == STEP_COORDINATE_EXTRACT:
                    methods_used.append("coordinate_extraction")
                    if not geo_meta:
                        limitations.append(
                            "Geographic coordinates (lat/lon) cannot be provided: "
                            "the uploaded image does not contain georeferencing metadata (GeoTIFF/CRS). "
                            "Normalized image coordinates [0–1] are used instead."
                        )

            except Exception as step_err:
                logger.error(f"Step '{step.step_type}' failed: {step_err}", exc_info=True)
                limitations.append(f"Analysis step '{step.step_type}' encountered an error: {step_err}")

        # ── Generate visual evidence ──────────────────────────────────────────
        if detections:
            target_str = " & ".join(t.replace("_", " ").title() for t in (qi.targets or ["Objects"]))
            visual_b64 = generate_detection_overlay(cv_img, detections, title=target_str.upper())
            visual_desc = f"{len(detections)} object(s) detected — annotated with bounding boxes"
        elif raw_stats is not None:
            # Fall back to land-cover overlay
            feature_type = self._pick_feature_type(qi.intent, qi.targets)
            visual_b64, visual_desc = _generate_visual_evidence(cv_img, feature_type, raw_stats)
        else:
            # Saliency heatmap fallback
            if raw_stats is None:
                raw_stats = _analyze_land_cover(cv_img, geo_meta)
            visual_b64, visual_desc = _generate_visual_evidence(cv_img, "saliency", raw_stats)

        return AnalysisResult(
            query_intent=qi,
            analysis_methods=list(dict.fromkeys(methods_used)),
            detections=detections,
            segments=segments,
            measurements=measurements,
            spatial_relationships=spatial_rels,
            visual_evidence_b64=visual_b64,
            visual_description=visual_desc,
            scene_description=scene_desc,
            geo_metadata=geo_meta,
            confidence=confidence,
            limitations=limitations,
            model_provenance="SatQuery CV Engine (OpenCV + ML Segmentation)",
            raw_land_cover_stats=raw_stats,
        )

    def _general_detection(self, cv_img, geo_meta) -> List[Detection]:
        """Detect generic structures when no specific targets are requested."""
        general_targets = ["building", "water_body", "vegetation", "road"]
        return detect_objects(cv_img, general_targets, geo_meta)

    def _add_area_measurements(
        self,
        measurements: List[Measurement],
        stats: Dict,
        targets: List[str],
        geo_meta
    ):
        """Add area measurements from land-cover stats for the requested targets."""
        has_real_geo = bool(geo_meta and "west" in geo_meta)
        geo_note = None if has_real_geo else "Area estimated using default 10m/pixel GSD"

        target_map = {
            "water_body":        ("water_area",   "water_pct",  "water_km2",  "water_ha"),
            "vegetation":        ("veg_area",      "veg_total_pct", "veg_km2", "veg_ha"),
            "forest":            ("forest_area",   "forest_pct",  "forest_km2","forest_ha"),
            "agricultural_field":("crop_area",     "crop_pct",   "crop_km2",  "crop_ha"),
            "building":          ("built_area",    "built_pct",  "built_km2", "built_ha"),
            "urban_area":        ("built_area",    "built_pct",  "built_km2", "built_ha"),
            "road":              ("built_area",    "built_pct",  "built_km2", "built_ha"),
            "bare_soil":         ("bare_area",     "bare_pct",   "bare_km2",  "bare_ha"),
        }

        used = set()
        resolved_targets = targets or []
        if not resolved_targets:
            resolved_targets = ["building", "water_body", "vegetation"]

        for t in resolved_targets:
            keys = target_map.get(t)
            if keys and keys[0] not in used:
                used.add(keys[0])
                pct_key, pct = keys[1], stats.get(keys[1], 0)
                km2 = stats.get(keys[2])
                ha  = stats.get(keys[3])
                if pct is not None and pct > 0:
                    if km2 is not None:
                        measurements.append(Measurement(
                            metric=f"{t}_coverage_km2",
                            value=round(km2, 2),
                            unit="km²",
                            source="pixel_analysis",
                            confidence=0.82 if has_real_geo else 0.65,
                            note=geo_note
                        ))
                    measurements.append(Measurement(
                        metric=f"{t}_coverage_pct",
                        value=round(pct, 1),
                        unit="%",
                        source="pixel_analysis",
                        confidence=0.88
                    ))

    def _pick_feature_type(self, intent: str, targets: List[str]) -> str:
        """Map intent to existing feature_type string for _generate_visual_evidence."""
        if "water_body" in targets or intent == INTENT_WATER_BODY:
            return "water"
        if any(t in ["vegetation","forest"] for t in targets) or intent == INTENT_VEGETATION:
            return "vegetation"
        if any(t in ["building","urban_area","road"] for t in targets):
            return "built"
        if intent in (INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION):
            return "land_cover"
        if intent in (INTENT_SCENE_UNDERSTANDING, INTENT_IMAGE_DESCRIPTION):
            return "land_cover"
        return "saliency"

    def _analyze_region(
        self,
        cv_img,
        constraint: Optional[str],
        stats: Dict,
        geo_meta
    ) -> str:
        """Analyze a specific region of the image."""
        h, w = cv_img.shape[:2]
        has_geo = bool(geo_meta and "west" in geo_meta)

        # Parse direction from constraint
        if not constraint:
            return ""

        c = constraint.lower()
        if any(k in c for k in ["north","upper","top"]):
            region_img = cv_img[:h//3, :, :]
            region_name = "northern" if has_geo else "upper"
        elif any(k in c for k in ["south","lower","bottom"]):
            region_img = cv_img[2*h//3:, :, :]
            region_name = "southern" if has_geo else "lower"
        elif any(k in c for k in ["east","right"]):
            region_img = cv_img[:, 2*w//3:, :]
            region_name = "eastern" if has_geo else "right"
        elif any(k in c for k in ["west","left"]):
            region_img = cv_img[:, :w//3, :]
            region_name = "western" if has_geo else "left"
        elif any(k in c for k in ["center","central","middle"]):
            region_img = cv_img[h//3:2*h//3, w//3:2*w//3, :]
            region_name = "central"
        else:
            return ""

        if region_img.size == 0:
            return ""

        region_stats = _analyze_land_cover(region_img)
        dominant = max(
            [("built-up", region_stats.get("built_pct",0)),
             ("water",    region_stats.get("water_pct",0)),
             ("vegetation",region_stats.get("veg_total_pct",0)),
             ("bare soil", region_stats.get("bare_pct",0))],
            key=lambda x: x[1]
        )
        return (
            f"The **{region_name} portion** of the image is predominantly "
            f"**{dominant[0]}** ({dominant[1]:.1f}%). "
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-image routing (preserves existing models)
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_multi_image(
        self,
        query: str,
        images: List[bytes],
        qi: QueryIntent,
        history: list
    ) -> Dict[str, Any]:
        """Route multi-image queries to existing specialist models."""
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["cloud","sar","radar","penetrate","fusion"]):
            task = "CROSS_MODAL_EXTRACTION"
        else:
            task = "CHANGE_ANALYSIS"

        tool = registry.get_model(task)
        tool_output = tool.execute(images, query)
        reasoning = f"Multi-image query ({len(images)} images). Routed to {task}."

        return {
            "status": "success",
            "answer": tool_output["text"],
            "visual_evidence": tool_output["visual_evidence"],
            "confidence": tool_output["confidence"],
            "compatibility_status": tool_output.get("compatibility_status", "PASSED"),
            "spatial_coherence_score": tool_output.get("spatial_coherence_score", 1.0),
            "execution_summary": {
                "selected_task": task,
                "tool_used": tool.__class__.__name__,
                "input_scope": f"Multi-image ({len(images)} images)",
                "num_images_processed": len(images),
                "agent_reasoning": reasoning,
            },
            "geo_metadata": tool_output.get("geo_metadata"),
            "geojson_data": tool_output.get("geojson_data"),
            "pair_comparison": tool_output.get("pair_comparison"),
            "query_understanding": {
                "intent": qi.intent,
                "targets": qi.targets,
                "confidence": qi.confidence,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Response builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_response(
        self,
        query: str,
        qi: QueryIntent,
        analysis_result: AnalysisResult,
        validation,
        final_text: str,
        geo_meta,
    ) -> Dict[str, Any]:
        """Assemble the final response dict."""
        from analysis_planner import (
            STEP_OBJECT_DETECTION, STEP_LAND_COVER, STEP_SCENE_DESCRIPTION,
        )

        # Visual evidence
        visual_evidence = []
        if analysis_result.visual_evidence_b64:
            visual_evidence.append({
                "image_base64": analysis_result.visual_evidence_b64,
                "description": analysis_result.visual_description,
            })

        # Human-readable task description
        intent_labels = {
            INTENT_OBJECT_LOCALIZATION: "Object Detection + Spatial Localization",
            INTENT_OBJECT_DETECTION:    "Object Detection",
            INTENT_OBJECT_COUNTING:     "Object Counting",
            INTENT_LAND_COVER:          "Land-Cover Classification",
            INTENT_AREA_QUANTIFICATION: "Area / Surface Quantification",
            INTENT_SEMANTIC_SEGMENTATION: "Semantic Segmentation",
            INTENT_SCENE_UNDERSTANDING: "Scene Understanding",
            INTENT_IMAGE_DESCRIPTION:   "Scene Description",
            INTENT_SPATIAL_RELATIONSHIP: "Spatial Relationship Analysis",
            INTENT_WATER_BODY:          "Water Body Detection",
            INTENT_VEGETATION:          "Vegetation Analysis",
            INTENT_AGRICULTURAL:        "Agricultural Analysis",
            INTENT_ROAD_NETWORK:        "Road Network Analysis",
            INTENT_REGION_DIRECTION:    "Region / Direction Analysis",
            INTENT_COMPARISON:          "Comparative Analysis",
            INTENT_CHANGE_DETECTION:    "Change Detection",
            INTENT_MULTI_TASK:          "Multi-Task Analysis",
            INTENT_COORDINATE_REQUEST:  "Coordinate / Geolocation",
        }
        interpreted_task = intent_labels.get(qi.intent, qi.intent.replace("_", " ").title())

        # Detections as serializable list
        det_list = []
        for d in analysis_result.detections:
            det_list.append({
                "id": d.id,
                "class": d.class_name,
                "display_name": d.display_name,
                "confidence": d.confidence,
                "confidence_label": d.confidence_label,
                "position": d.image_position,
                "bbox_norm": d.bbox_norm,
                "geographic_position": d.geographic_position,
            })

        # GeoJSON from detections (bounding boxes)
        geojson_data = None
        if analysis_result.detections and geo_meta and "west" in geo_meta:
            geojson_data = self._detections_to_geojson(analysis_result.detections, geo_meta)
        elif analysis_result.raw_land_cover_stats and geo_meta:
            # Fall back to existing land-cover GeoJSON if available
            pass

        return {
            "status": "success",
            "answer": final_text,
            "visual_evidence": visual_evidence,
            "confidence": validation.overall_confidence,
            "compatibility_status": "PASSED" if validation.passed else "WARNINGS",
            "spatial_coherence_score": 1.0,
            "execution_summary": {
                "selected_task": qi.intent,
                "input_scope": "Single Image",
                "num_images_processed": 1,
                "agent_reasoning": (
                    f"Query intent: {qi.intent} (confidence {qi.confidence:.0%}). "
                    f"Targets: {qi.targets}. "
                    f"Methods: {analysis_result.analysis_methods}."
                ),
                "compatibility_status": "PASSED",
                "spatial_coherence_score": 1.0,
                "model_provenance": analysis_result.model_provenance,
            },
            "geo_metadata": geo_meta,
            "geojson_data": geojson_data,
            "pair_comparison": None,
            # New structured fields
            "query_understanding": {
                "intent": qi.intent,
                "sub_intents": qi.sub_intents,
                "targets": qi.targets,
                "operation": qi.operation,
                "spatial_constraint": qi.spatial_constraint,
                "requires_count": qi.requires_count,
                "requires_area": qi.requires_area_calculation,
                "confidence": qi.confidence,
            },
            "interpreted_task": interpreted_task,
            "detections": det_list,
            "detection_count": len(det_list),
            "measurements": [
                {"metric": m.metric, "value": m.value, "unit": m.unit,
                 "confidence": m.confidence, "note": m.note}
                for m in analysis_result.measurements
            ],
            "limitations": analysis_result.limitations,
            "validation": {
                "passed": validation.passed,
                "issues": validation.issues,
                "warnings": validation.warnings,
            },
        }

    def _detections_to_geojson(
        self,
        detections: List[Detection],
        geo_meta: Dict[str, Any]
    ) -> Dict:
        """Convert detections to RFC 7946 GeoJSON."""
        features = []
        lon_min, lon_max = geo_meta["west"], geo_meta["east"]
        lat_min, lat_max = geo_meta["south"], geo_meta["north"]

        def norm_to_geo(nx, ny):
            lon = lon_min + nx * (lon_max - lon_min)
            lat = lat_max - ny * (lat_max - lat_min)
            return [round(lon, 6), round(lat, 6)]

        for det in detections:
            b = det.bbox_norm
            p1 = norm_to_geo(b["x_min"], b["y_min"])
            p2 = norm_to_geo(b["x_max"], b["y_min"])
            p3 = norm_to_geo(b["x_max"], b["y_max"])
            p4 = norm_to_geo(b["x_min"], b["y_max"])
            features.append({
                "type": "Feature",
                "id": det.id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[p1, p2, p3, p4, p1]]
                },
                "properties": {
                    "id": det.id,
                    "class": det.class_name,
                    "label": det.display_name,
                    "confidence": det.confidence,
                    "position": det.image_position,
                }
            })

        return {
            "type": "FeatureCollection",
            "metadata": {
                "mission": "SatQuery AI Object Detection",
                "crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
                "total_detections": len(detections),
            },
            "features": features
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy classify_task (kept for compatibility — not used by new pipeline)
    # ─────────────────────────────────────────────────────────────────────────

    def classify_task(self, query: str, num_images: int, history=None) -> Dict[str, Any]:
        """Legacy method — preserved for compatibility. New code uses execute_query directly."""
        qi = query_understanding.analyze(query, num_images)
        if num_images >= 2:
            query_lower = query.lower()
            if any(kw in query_lower for kw in ["cloud","sar","radar","penetrate","fusion"]):
                task = "CROSS_MODAL_EXTRACTION"
            else:
                task = "CHANGE_ANALYSIS"
        else:
            task = "SINGLE_IMAGE_VQA"  # Not used by new pipeline but returned for compatibility
        return {"task": task, "reasoning": f"Intent: {qi.intent}"}


# Singleton — same name as before so main.py import works
agent_controller = QueryRouter()
