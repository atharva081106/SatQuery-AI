"""
SatQuery AI — Answer Generator
Produces the final user-facing text response, adapting format to query type.

Design principles:
- Output format matches the query type (not always a giant report)
- Direct answer first, evidence second
- Quantitative metrics only when justified
- Explicit uncertainty for low-confidence results
- No fake precision, no invented coordinates
"""
import logging
from typing import List, Optional

from data_models import (
    QueryIntent, AnalysisResult, ValidationResult, Detection, Measurement,
    INTENT_OBJECT_DETECTION, INTENT_OBJECT_LOCALIZATION, INTENT_OBJECT_COUNTING,
    INTENT_OBJECT_CLASSIFICATION, INTENT_OBJECT_ATTRIBUTE,
    INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION, INTENT_AREA_QUANTIFICATION,
    INTENT_DISTANCE_PROXIMITY, INTENT_SPATIAL_RELATIONSHIP,
    INTENT_REGION_DIRECTION, INTENT_IMAGE_DESCRIPTION, INTENT_SCENE_UNDERSTANDING,
    INTENT_CHANGE_DETECTION, INTENT_COMPARISON, INTENT_TEMPORAL,
    INTENT_ANOMALY_DETECTION, INTENT_INFRASTRUCTURE, INTENT_AGRICULTURAL,
    INTENT_WATER_BODY, INTENT_VEGETATION, INTENT_ROAD_NETWORK,
    INTENT_COORDINATE_REQUEST, INTENT_MULTI_TASK,
)
from spatial_reasoning import describe_distribution

logger = logging.getLogger(__name__)


def _format_detection_list(detections: List[Detection], class_name: str = None) -> str:
    """Format a list of detections as a clean numbered list."""
    if not detections:
        return "  No detections found."

    lines = []
    for det in detections:
        pos = det.image_position.replace("-", " ")
        conf_label = det.confidence_label
        geo = ""
        if det.geographic_position:
            geo = f" — ({det.geographic_position['lat']:.4f}°N, {det.geographic_position['lon']:.4f}°E)"
        lines.append(
            f"  • **{det.display_name}** — {pos}{geo} [{conf_label} confidence: {det.confidence:.0%}]"
        )
    return "\n".join(lines)


def _format_measurement(m: Measurement) -> str:
    if m.unit in ["%", "percentage"]:
        return f"**{m.value:.1f}%**"
    if m.unit in ["km²", "km2"]:
        return f"**{m.value:.2f} km²**"
    if m.unit in ["ha", "hectares"]:
        return f"**{m.value:,.0f} ha**"
    if m.unit == "count":
        return f"**{m.value}**"
    return f"**{m.value} {m.unit}**"


def _format_limitations(limitations: List[str], validation: ValidationResult) -> str:
    all_lims = list(limitations or [])
    if validation and validation.warnings:
        all_lims.extend(validation.warnings)
    if validation and not validation.passed:
        all_lims.extend(validation.issues)
    if not all_lims:
        return ""
    bullets = "\n".join(f"  • {l}" for l in all_lims)
    return f"\n\n---\n\n**⚠ Limitations & Uncertainty**\n{bullets}"


def _format_coord_note(result: AnalysisResult) -> str:
    """Note about coordinate availability."""
    if result.geo_metadata and "west" in result.geo_metadata:
        geo = result.geo_metadata
        return (
            f"\n\n**📍 Georeferencing Available**\n"
            f"  • CRS: `{geo.get('crs', 'EPSG:4326')}`\n"
            f"  • Bounds: {geo['south']:.4f}°N to {geo['north']:.4f}°N, "
            f"{geo['west']:.4f}°E to {geo['east']:.4f}°E"
        )
    return (
        "\n\n**📍 No Georeferencing Metadata**\n"
        "  • Geographic coordinates (lat/lon) are not available for this image.\n"
        "  • Object positions are reported as relative image coordinates (normalized 0–1)."
    )


class AnswerGenerator:
    """Generates the final user-facing answer string from analysis results."""

    def generate(
        self,
        qi: QueryIntent,
        result: AnalysisResult,
        validation: ValidationResult
    ) -> str:
        try:
            return self._route(qi, result, validation)
        except Exception as e:
            logger.error(f"AnswerGenerator error: {e}")
            return (
                f"Analysis completed with {len(result.detections)} detection(s) detected.\n\n"
                f"{result.scene_description or 'Scene analysis performed.'}"
            )

    def _route(self, qi: QueryIntent, result: AnalysisResult, validation: ValidationResult) -> str:
        intent = qi.intent

        # Multi-task: build composite answer
        if intent == INTENT_MULTI_TASK and qi.sub_intents:
            return self._multi_task(qi, result, validation)

        if intent == INTENT_OBJECT_LOCALIZATION:
            return self._object_localization(qi, result, validation)
        if intent == INTENT_OBJECT_DETECTION:
            return self._object_detection(qi, result, validation)
        if intent == INTENT_OBJECT_COUNTING:
            return self._object_counting(qi, result, validation)
        if intent == INTENT_OBJECT_CLASSIFICATION:
            return self._object_localization(qi, result, validation)  # same format
        if intent == INTENT_SPATIAL_RELATIONSHIP:
            return self._spatial_relationship(qi, result, validation)
        if intent in (INTENT_LAND_COVER, INTENT_SEMANTIC_SEGMENTATION):
            return self._land_cover(qi, result, validation)
        if intent == INTENT_AREA_QUANTIFICATION:
            return self._area_quantification(qi, result, validation)
        if intent == INTENT_WATER_BODY:
            return self._water_body(qi, result, validation)
        if intent == INTENT_VEGETATION:
            return self._vegetation(qi, result, validation)
        if intent == INTENT_AGRICULTURAL:
            return self._agricultural(qi, result, validation)
        if intent == INTENT_ROAD_NETWORK:
            return self._road_network(qi, result, validation)
        if intent == INTENT_REGION_DIRECTION:
            return self._region_direction(qi, result, validation)
        if intent == INTENT_SCENE_UNDERSTANDING:
            return self._scene_understanding(qi, result, validation)
        if intent == INTENT_IMAGE_DESCRIPTION:
            return self._scene_understanding(qi, result, validation)
        if intent == INTENT_COMPARISON:
            return self._comparison(qi, result, validation)
        if intent == INTENT_INFRASTRUCTURE:
            return self._object_localization(qi, result, validation)
        if intent == INTENT_COORDINATE_REQUEST:
            return self._coordinate_request(qi, result, validation)

        # Fallback
        return self._scene_understanding(qi, result, validation)

    # ── Object Localization ─────────────────────────────────────────────────

    def _object_localization(self, qi, result, validation) -> str:
        targets = [t.replace("_", " ").title() for t in qi.targets] if qi.targets else ["Objects"]
        target_str = " & ".join(targets)
        dets = result.detections

        # Group detections by class
        by_class: dict = {}
        for d in dets:
            by_class.setdefault(d.class_name, []).append(d)

        sections = []
        for cls, cls_dets in by_class.items():
            cls_label = cls.replace("_", " ").title()
            distribution = describe_distribution(cls_dets, bool(result.geo_metadata))
            section = (
                f"**{cls_label.upper()} ({len(cls_dets)} detected)**\n"
                f"{_format_detection_list(cls_dets)}\n\n"
                f"  Distribution: _{distribution}_"
            )
            sections.append(section)

        if not sections:
            no_det_msg = (
                f"**No {target_str} detected** in this image using the current CV engine.\n\n"
                f"This may be because:\n"
                f"  • The objects are absent or very small at this resolution\n"
                f"  • The image quality or lighting conditions limit detection\n"
                f"  • A trained neural detection model would provide better results"
            )
            return no_det_msg + _format_limitations(result.limitations, validation)

        header = f"**DETECTED: {target_str}**\n\n"
        body = "\n\n---\n\n".join(sections)

        coord_note = ""
        if qi.requires_coordinates:
            coord_note = _format_coord_note(result)

        return header + body + coord_note + _format_limitations(result.limitations, validation)

    # ── Object Detection (general) ──────────────────────────────────────────

    def _object_detection(self, qi, result, validation) -> str:
        dets = result.detections
        if not dets:
            return (
                "**No objects detected** matching the query criteria.\n\n"
                f"{result.scene_description or ''}"
                + _format_limitations(result.limitations, validation)
            )
        return self._object_localization(qi, result, validation)

    # ── Object Counting ─────────────────────────────────────────────────────

    def _object_counting(self, qi, result, validation) -> str:
        dets = result.detections
        targets = [t.replace("_", " ").title() for t in qi.targets] if qi.targets else ["Objects"]
        target_str = " & ".join(targets)

        # Group by class
        by_class: dict = {}
        for d in dets:
            by_class.setdefault(d.class_name, []).append(d)

        count_lines = []
        for cls, cls_dets in by_class.items():
            count_lines.append(f"  • {cls.replace('_',' ').title()}: **{len(cls_dets)}**")

        if count_lines:
            count_summary = "\n".join(count_lines)
            body = (
                f"**DETECTION COUNT: {target_str}**\n\n"
                f"  Total detected: **{len(dets)}**\n"
                f"{count_summary}\n\n"
            )
        else:
            body = f"**DETECTION COUNT: {target_str}**\n\n  Total detected: **0**\n\n"

        # Include confidence note
        if dets:
            avg_conf = sum(d.confidence for d in dets) / len(dets)
            body += f"  Average detection confidence: **{avg_conf:.0%}**\n"

        body += _format_limitations(result.limitations, validation)
        return body

    # ── Spatial Relationship ────────────────────────────────────────────────

    def _spatial_relationship(self, qi, result, validation) -> str:
        rels = result.spatial_relationships
        dets = result.detections

        if not dets:
            return (
                "**SPATIAL RELATIONSHIP ANALYSIS**\n\n"
                "No objects were detected for spatial relationship analysis.\n"
                + _format_limitations(result.limitations, validation)
            )

        lines = ["**SPATIAL RELATIONSHIP ANALYSIS**\n"]

        # Show all detections with positions
        by_class: dict = {}
        for d in dets:
            by_class.setdefault(d.class_name, []).append(d)

        for cls, cls_dets in by_class.items():
            lines.append(f"\n**{cls.replace('_',' ').title()} ({len(cls_dets)}):**")
            lines.append(_format_detection_list(cls_dets))

        if rels:
            lines.append("\n\n**Spatial Relationships:**")
            for r in rels:
                subj = r.subject_class.replace("_", " ").title()
                obj  = r.object_class.replace("_", " ").title()
                count = len(r.instances)
                lines.append(
                    f"  • **{count} {subj}(s)** detected within proximity of {obj}s "
                    f"(IDs: {', '.join(r.instances[:5])})"
                )
        else:
            lines.append("\n\nNo close proximity relationships found between requested object classes.")

        return "\n".join(lines) + _format_limitations(result.limitations, validation)

    # ── Land Cover ──────────────────────────────────────────────────────────

    def _land_cover(self, qi, result, validation) -> str:
        segs = result.segments
        if not segs:
            return (
                "**LAND-COVER ANALYSIS**\n\n"
                "Land-cover classification could not be completed for this image.\n"
                + _format_limitations(result.limitations, validation)
            )

        lines = ["**LAND-COVER CLASSIFICATION**\n"]
        for seg in sorted(segs, key=lambda s: -s.percentage):
            km2_str = f" ({seg.area_km2:.2f} km²)" if seg.area_km2 else ""
            ha_str  = f" / {seg.area_ha:,.0f} ha" if seg.area_ha else ""
            lines.append(
                f"  • **{seg.class_name.replace('_',' ').title()}**: "
                f"**{seg.percentage:.1f}%**{km2_str}{ha_str}"
            )

        note = ""
        if not result.geo_metadata:
            note = (
                "\n\n  ⚠ Area values use a default 10m/pixel GSD assumption. "
                "Actual physical area may differ. Provide a GeoTIFF for accurate measurements."
            )

        return "\n".join(lines) + note + _format_limitations(result.limitations, validation)

    # ── Area Quantification ─────────────────────────────────────────────────

    def _area_quantification(self, qi, result, validation) -> str:
        targets = [t.replace("_", " ").title() for t in qi.targets] if qi.targets else ["Surface"]
        target_str = " & ".join(targets)
        measurements = result.measurements

        lines = [f"**AREA / SURFACE QUANTIFICATION: {target_str}**\n"]

        for m in measurements:
            note = f" _{m.note}_" if m.note else ""
            lines.append(f"  • **{m.metric.replace('_',' ').title()}**: {_format_measurement(m)}{note}")

        if not measurements:
            # Fall back to segments
            if result.segments:
                for seg in result.segments:
                    if any(t.lower() in seg.class_name.lower() for t in (qi.targets or [])):
                        km2_str = f" ({seg.area_km2:.2f} km²)" if seg.area_km2 else ""
                        lines.append(
                            f"  • **{seg.class_name.replace('_',' ').title()}**: "
                            f"**{seg.percentage:.1f}%**{km2_str}"
                        )
            else:
                lines.append("  Area quantification data not available.")

        geo_note = ""
        if not result.geo_metadata:
            geo_note = (
                "\n\n  ⚠ **No georeferencing metadata found.** Area estimates use a default "
                "10m/pixel GSD (Sentinel-2 equivalent). Actual physical area may differ significantly. "
                "Provide a GeoTIFF for calibrated measurements."
            )

        return "\n".join(lines) + geo_note + _format_limitations(result.limitations, validation)

    # ── Water Body ──────────────────────────────────────────────────────────

    def _water_body(self, qi, result, validation) -> str:
        dets = [d for d in result.detections if d.class_name in ("water_body", "coastline")]
        water_segs = [s for s in result.segments if "water" in s.class_name.lower()]

        lines = ["**WATER BODY DETECTION**\n"]

        if dets:
            distribution = describe_distribution(dets)
            lines.append(f"  **{len(dets)} water bod{'ies' if len(dets)!=1 else 'y'} detected**")
            lines.append(_format_detection_list(dets))
            lines.append(f"\n  Distribution: _{distribution}_")

        if water_segs:
            for s in water_segs:
                km2_str = f" ({s.area_km2:.2f} km²)" if s.area_km2 else ""
                lines.append(f"\n  • Total water coverage: **{s.percentage:.1f}%**{km2_str}")

        if not dets and not water_segs:
            lines.append("  No significant water bodies detected in this image.")

        return "\n".join(lines) + _format_limitations(result.limitations, validation)

    # ── Vegetation ──────────────────────────────────────────────────────────

    def _vegetation(self, qi, result, validation) -> str:
        dets = [d for d in result.detections if d.class_name in ("vegetation","forest","agricultural_field")]
        veg_segs = [s for s in result.segments if any(
            k in s.class_name.lower() for k in ("veg","forest","crop","green")
        )]

        lines = ["**VEGETATION ANALYSIS**\n"]
        if dets:
            lines.append(f"  **{len(dets)} vegetation region(s) detected**")
            lines.append(_format_detection_list(dets))
        if veg_segs:
            for s in veg_segs:
                km2_str = f" ({s.area_km2:.2f} km²)" if s.area_km2 else ""
                lines.append(f"\n  • {s.class_name.replace('_',' ').title()}: **{s.percentage:.1f}%**{km2_str}")

        if not dets and not veg_segs:
            lines.append("  No significant vegetation detected in this image.")

        return "\n".join(lines) + _format_limitations(result.limitations, validation)

    # ── Agricultural ────────────────────────────────────────────────────────

    def _agricultural(self, qi, result, validation) -> str:
        dets = [d for d in result.detections if "agri" in d.class_name or "field" in d.class_name or "crop" in d.class_name]
        lines = ["**AGRICULTURAL ANALYSIS**\n"]
        if dets:
            lines.append(f"  **{len(dets)} agricultural field region(s) detected**")
            lines.append(_format_detection_list(dets))
        else:
            lines.append("  No distinct agricultural fields detected at this resolution.")

        if result.segments:
            for s in result.segments:
                if any(k in s.class_name.lower() for k in ("agri","crop","farm")):
                    km2_str = f" ({s.area_km2:.2f} km²)" if s.area_km2 else ""
                    lines.append(f"\n  • Coverage: **{s.percentage:.1f}%**{km2_str}")

        return "\n".join(lines) + _format_limitations(result.limitations, validation)

    # ── Road Network ────────────────────────────────────────────────────────

    def _road_network(self, qi, result, validation) -> str:
        dets = [d for d in result.detections if d.class_name in ("road","runway","bridge")]
        lines = ["**ROAD / NETWORK ANALYSIS**\n"]
        if dets:
            lines.append(f"  **{len(dets)} road / linear feature(s) detected**")
            lines.append(_format_detection_list(dets))
        else:
            lines.append("  No prominent road or network features detected.")
        return "\n".join(lines) + _format_limitations(result.limitations, validation)

    # ── Region / Direction ──────────────────────────────────────────────────

    def _region_direction(self, qi, result, validation) -> str:
        constraint = qi.spatial_constraint or "the specified region"
        lines = [f"**REGION ANALYSIS: {constraint.upper()}**\n"]

        if result.scene_description:
            lines.append(result.scene_description)

        if result.segments:
            lines.append("\n\n**Land-Cover in Analysed Region:**")
            for s in sorted(result.segments, key=lambda s: -s.percentage):
                if s.percentage > 2:
                    lines.append(
                        f"  • {s.class_name.replace('_',' ').title()}: **{s.percentage:.1f}%**"
                    )

        note = ""
        if not result.geo_metadata:
            note = (
                "\n\n  ⚠ Image orientation is unknown. 'Upper/lower/left/right' refer to image "
                "coordinates, not geographic north/south/east/west."
            )

        return "\n".join(lines) + note + _format_limitations(result.limitations, validation)

    # ── Scene Understanding ─────────────────────────────────────────────────

    def _scene_understanding(self, qi, result, validation) -> str:
        lines = ["**SCENE ANALYSIS**\n"]

        if result.scene_description:
            lines.append(result.scene_description)
        elif result.segments:
            dominant = max(result.segments, key=lambda s: s.percentage)
            desc = f"The scene is dominated by **{dominant.class_name.replace('_',' ').title()}** ({dominant.percentage:.1f}%)."
            lines.append(desc)
        else:
            lines.append("Scene analysis completed.")

        if result.segments:
            lines.append("\n\n**Observed Land-Cover:**")
            for s in sorted(result.segments, key=lambda s: -s.percentage):
                if s.percentage > 1.0:
                    km2_str = f" ({s.area_km2:.2f} km²)" if s.area_km2 else ""
                    lines.append(
                        f"  • {s.class_name.replace('_',' ').title()}: **{s.percentage:.1f}%**{km2_str}"
                    )

        return "\n".join(lines) + _format_limitations(result.limitations, validation)

    # ── Comparison ──────────────────────────────────────────────────────────

    def _comparison(self, qi, result, validation) -> str:
        lines = ["**COMPARATIVE ANALYSIS**\n"]

        if result.scene_description:
            lines.append(result.scene_description)

        if result.segments:
            lines.append("\n\n**Full Scene Land-Cover Distribution:**")
            for s in sorted(result.segments, key=lambda s: -s.percentage):
                if s.percentage > 1.0:
                    lines.append(
                        f"  • {s.class_name.replace('_',' ').title()}: **{s.percentage:.1f}%**"
                    )

        return "\n".join(lines) + _format_limitations(result.limitations, validation)

    # ── Coordinate Request ──────────────────────────────────────────────────

    def _coordinate_request(self, qi, result, validation) -> str:
        lines = ["**COORDINATES & GEOLOCATION**\n"]
        lines.append(_format_coord_note(result))

        if result.detections:
            lines.append("\n\n**Object Coordinates:**")
            for det in result.detections[:10]:
                if det.geographic_position:
                    lines.append(
                        f"  • **{det.display_name}**: "
                        f"{det.geographic_position['lat']:.5f}°N, "
                        f"{det.geographic_position['lon']:.5f}°E"
                    )
                else:
                    bbox = det.bbox_norm
                    lines.append(
                        f"  • **{det.display_name}**: "
                        f"Normalized position — "
                        f"x:[{bbox['x_min']:.3f}–{bbox['x_max']:.3f}], "
                        f"y:[{bbox['y_min']:.3f}–{bbox['y_max']:.3f}] "
                        f"(image coords, top-left origin)"
                    )

        return "\n".join(lines) + _format_limitations(result.limitations, validation)

    # ── Multi-Task ──────────────────────────────────────────────────────────

    def _multi_task(self, qi, result, validation) -> str:
        sections = []
        for sub_intent in (qi.sub_intents or [qi.intent]):
            qi_copy = QueryIntent(
                raw_query=qi.raw_query,
                intent=sub_intent,
                targets=qi.targets,
                operation=qi.operation,
                spatial_constraint=qi.spatial_constraint,
                requires_coordinates=qi.requires_coordinates,
                requires_area_calculation=qi.requires_area_calculation,
                requires_count=qi.requires_count,
                confidence=qi.confidence,
            )
            section = self._route(qi_copy, result, None)
            sections.append(section)

        return "\n\n---\n\n".join(sections) + _format_limitations(result.limitations, validation)


# Singleton
answer_generator = AnswerGenerator()
