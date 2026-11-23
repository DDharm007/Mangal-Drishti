"""Configuration and JSON schema validation for Terrain Intelligence Engine reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


DEFAULT_TRAVERSABILITY_WEIGHTS: Dict[str, float] = {
    "soil": 0.10,
    "bedrock": 0.25,
    "sand": 0.60,
    "big_rock": 1.00,
    "component_terrain": 0.45,
    "component_big_rock_density": 0.20,
    "component_sand_concentration": 0.15,
    "component_uncertainty": 0.15,
    "component_fragmentation": 0.05,
}


@dataclass
class IntelligenceConfig:
    """Configurable thresholds and parameters for terrain intelligence modules."""

    ignore_index: int = 255
    min_region_area_px: int = 120
    min_obstacle_area_px: int = 64
    min_obstacle_confidence: float = 0.35
    high_confidence_threshold: float = 0.80
    medium_confidence_threshold: float = 0.55
    high_uncertainty_threshold: float = 0.55
    min_valid_pixel_pct_for_assessment: float = 15.0
    min_mean_confidence_for_assessment: float = 0.50
    max_low_conf_pct_for_assessment: float = 45.0
    traversability_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TRAVERSABILITY_WEIGHTS)
    )


REQUIRED_TOP_KEYS = {
    "project",
    "schema_version",
    "model",
    "checkpoint",
    "device",
    "image_metadata",
    "analysis",
    "artifacts",
}

REQUIRED_ANALYSIS_KEYS = {
    "terrain_summary",
    "terrain_details",
    "regions",
    "obstacle_candidates",
    "confidence",
    "uncertainty",
    "traversability",
}

ALLOWED_TRAVERSABILITY_ASSESSMENTS = {
    "lower_visual_concern",
    "moderate_visual_concern",
    "higher_visual_concern",
    "insufficient_visual_evidence",
}


def validate_intelligence_report(report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate structured JSON report integrity and mandatory scientific safety invariants."""
    errors: List[str] = []
    if not isinstance(report, dict):
        return False, ["Report root must be a dictionary."]

    missing_top = REQUIRED_TOP_KEYS - set(report.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")

    analysis = report.get("analysis")
    if not isinstance(analysis, dict):
        errors.append("'analysis' must be a dictionary.")
        return False, errors

    missing_analysis = REQUIRED_ANALYSIS_KEYS - set(analysis.keys())
    if missing_analysis:
        errors.append(f"Missing analysis keys: {sorted(missing_analysis)}")

    ts = analysis.get("terrain_summary", {})
    if not isinstance(ts, dict):
        errors.append("'terrain_summary' must be a dictionary.")
    else:
        for k in ("soil", "bedrock", "sand", "big_rock"):
            if k not in ts:
                errors.append(f"terrain_summary missing class '{k}'")
            elif not (0.0 <= float(ts[k]) <= 100.0):
                errors.append(f"terrain_summary['{k}'] out of [0, 100]: {ts[k]}")

    td = analysis.get("terrain_details", {})
    if not isinstance(td, dict) or "total_valid_pixels" not in td or "valid_pixel_percentage" not in td:
        errors.append("'terrain_details' missing total_valid_pixels or valid_pixel_percentage.")

    regions = analysis.get("regions")
    if not isinstance(regions, list):
        errors.append("'regions' must be a list.")

    candidates = analysis.get("obstacle_candidates")
    if not isinstance(candidates, list):
        errors.append("'obstacle_candidates' must be a list.")
    else:
        for idx, cand in enumerate(candidates):
            if cand.get("is_confirmed_physical_obstacle") is not False:
                errors.append(f"obstacle_candidates[{idx}].is_confirmed_physical_obstacle must be False.")
            for req_k in ("candidate_id", "bbox", "pixel_area", "confidence", "centroid", "shape", "relative_position"):
                if req_k not in cand:
                    errors.append(f"obstacle_candidates[{idx}] missing '{req_k}'")

    conf = analysis.get("confidence", {})
    if not isinstance(conf, dict):
        errors.append("'confidence' must be a dictionary.")
    else:
        if conf.get("is_calibrated_probability") is not False:
            errors.append("confidence.is_calibrated_probability must be False.")
        if "mean_confidence" in conf and not (0.0 <= float(conf["mean_confidence"]) <= 1.0):
            errors.append(f"confidence.mean_confidence out of [0, 1]: {conf['mean_confidence']}")

    unc = analysis.get("uncertainty", {})
    if not isinstance(unc, dict):
        errors.append("'uncertainty' must be a dictionary.")
    else:
        if "mean_entropy" in unc and float(unc["mean_entropy"]) < 0.0:
            errors.append(f"uncertainty.mean_entropy cannot be negative: {unc['mean_entropy']}")

    trav = analysis.get("traversability", {})
    if not isinstance(trav, dict):
        errors.append("'traversability' must be a dictionary.")
    else:
        if trav.get("physical_safety_verified") is not False:
            errors.append("traversability.physical_safety_verified MUST be strictly False.")
        if trav.get("is_experimental") is not True:
            errors.append("traversability.is_experimental MUST be True.")
        if trav.get("assessment") not in ALLOWED_TRAVERSABILITY_ASSESSMENTS:
            errors.append(f"Invalid traversability.assessment: {trav.get('assessment')}")
        missing_meas = trav.get("missing_measurements")
        if not isinstance(missing_meas, list) or len(missing_meas) == 0:
            errors.append("traversability.missing_measurements must be a non-empty list.")

    return len(errors) == 0, errors
