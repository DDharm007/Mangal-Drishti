"""Experimental 2D image-based terrain visual concern assessment (non-physical)."""

from __future__ import annotations

from typing import Any, Dict, List

from src.intelligence.schemas import IntelligenceConfig

MISSING_PHYSICAL_MEASUREMENTS = [
    "3D depth / stereo range",
    "terrain slope / elevation gradient",
    "obstacle physical height & clearance",
    "wheel-terrain slip & soil mechanics",
    "rover kinematics & suspension geometry",
]


def assess_experimental_traversability(
    terrain_result: Dict[str, Any],
    regions: List[Dict[str, Any]],
    obstacle_candidates: List[Dict[str, Any]],
    confidence_result: Dict[str, Any],
    uncertainty_result: Dict[str, Any],
    config: IntelligenceConfig,
) -> Dict[str, Any]:
    """Compute experimental 2D visual terrain concern index; NEVER certifies physical safety."""
    ts = terrain_result["terrain_summary"]
    td = terrain_result["terrain_details"]
    w = config.traversability_weights

    valid_pct = float(td["valid_pixel_percentage"])
    mean_conf = float(confidence_result["mean_confidence"])
    low_conf_pct = float(confidence_result["low_confidence_pixel_percentage"])
    mean_norm_ent = float(uncertainty_result["mean_normalized_entropy"])

    # Weighted 2D terrain class visual hazard proxy in [0, 1]
    terrain_score = min(
        1.0,
        (
            w["soil"] * (float(ts["soil"]) / 100.0)
            + w["bedrock"] * (float(ts["bedrock"]) / 100.0)
            + w["sand"] * (float(ts["sand"]) / 100.0)
            + w["big_rock"] * (float(ts["big_rock"]) / 100.0)
        ),
    )

    # Big Rock candidate density factor: combines pixel coverage (saturates at 10%) and count (saturates at 5)
    total_br_cov = sum(float(c["image_coverage_pct"]) for c in obstacle_candidates)
    br_count = len(obstacle_candidates)
    big_rock_density_score = min(1.0, 0.6 * min(total_br_cov / 10.0, 1.0) + 0.4 * min(br_count / 5.0, 1.0))

    # Sand concentration factor (saturates at 60% sand coverage, reflecting loose-regolith sinkage risk proxy)
    sand_concentration_score = min(1.0, float(ts["sand"]) / 60.0)

    # Segmentation uncertainty factor combining entropy & low-confidence fraction
    uncertainty_score = min(1.0, 0.55 * mean_norm_ent + 0.45 * min(low_conf_pct / 50.0, 1.0))

    # Region distribution fragmentation factor (saturates at 25 distinct regions)
    fragmentation_score = min(1.0, len(regions) / 25.0)

    concern_index = min(
        1.0,
        max(
            0.0,
            w["component_terrain"] * terrain_score
            + w["component_big_rock_density"] * big_rock_density_score
            + w["component_sand_concentration"] * sand_concentration_score
            + w["component_uncertainty"] * uncertainty_score
            + w["component_fragmentation"] * fragmentation_score,
        ),
    )

    insufficient_reasons: List[str] = []
    if valid_pct < config.min_valid_pixel_pct_for_assessment:
        insufficient_reasons.append(
            f"valid_pixel_percentage={valid_pct:.2f}% < {config.min_valid_pixel_pct_for_assessment}%"
        )
    if mean_conf < config.min_mean_confidence_for_assessment:
        insufficient_reasons.append(
            f"mean_confidence={mean_conf:.3f} < {config.min_mean_confidence_for_assessment}"
        )
    if low_conf_pct > config.max_low_conf_pct_for_assessment:
        insufficient_reasons.append(
            f"low_confidence_pixel_percentage={low_conf_pct:.2f}% > {config.max_low_conf_pct_for_assessment}%"
        )

    if insufficient_reasons:
        assessment = "insufficient_visual_evidence"
    elif concern_index < 0.30:
        assessment = "lower_visual_concern"
    elif concern_index < 0.58:
        assessment = "moderate_visual_concern"
    else:
        assessment = "higher_visual_concern"

    return {
        "assessment": assessment,
        "experimental_visual_concern_index": round(concern_index, 6),
        "is_experimental": True,
        "physical_safety_verified": False,
        "insufficient_evidence_triggers": insufficient_reasons,
        "component_scores": {
            "terrain_composition_score": round(terrain_score, 6),
            "big_rock_candidate_density_score": round(big_rock_density_score, 6),
            "sand_concentration_score": round(sand_concentration_score, 6),
            "segmentation_uncertainty_score": round(uncertainty_score, 6),
            "region_fragmentation_score": round(fragmentation_score, 6),
        },
        "terrain_class_indicators": {
            "soil_pct": float(ts["soil"]),
            "bedrock_pct": float(ts["bedrock"]),
            "sand_pct": float(ts["sand"]),
            "big_rock_pct": float(ts["big_rock"]),
            "big_rock_candidate_count": br_count,
            "retained_region_count": len(regions),
        },
        "missing_measurements": list(MISSING_PHYSICAL_MEASUREMENTS),
        "formula_summary": (
            "0.45*terrain_composition + 0.20*big_rock_density + 0.15*sand_concentration + "
            "0.15*segmentation_uncertainty + 0.05*region_fragmentation; "
            "overrides to insufficient_visual_evidence when valid pixels < 15% or mean_confidence < 0.50 or low_conf > 45%."
        ),
        "scientific_disclaimer": (
            "Experimental 2D semantic indicator ONLY. Visual terrain segmentation cannot verify physical "
            "traversability or rover safety without 3D stereo depth, slope, rock height, and wheel-soil mechanics."
        ),
    }
