"""
Evidence analyzer.

Pure functions that turn raw YOLO detections (image or video) into a
structured EvidenceAnalysis: counts, severity, suggested crime category,
and key observations.

Design rules:
- No I/O. No model loading. No side effects.
- Same interface for image and video via AnalysisInput.
- All weights/thresholds live in config/settings.py.

Phase 46 — weapons are split into *verified* and *candidate* buckets.
`verified_weapon_count` is what drives severity and `has_threat`;
`candidate_weapon_count` is surfaced for transparency but never
inflates severity or changes the suggested category.
"""

from __future__ import annotations

from config import CATEGORY_RULES, SEVERITY_THRESHOLDS, SEVERITY_WEIGHTS
from models.schemas import AnalysisInput, EvidenceAnalysis


def analyze(input_data: AnalysisInput) -> EvidenceAnalysis:
    """
    Run the evidence analysis on an AnalysisInput.

    Returns an EvidenceAnalysis dataclass. Never raises on empty
    input — returns a safe "unknown / low severity" analysis instead.
    """
    counts = dict(input_data.counts_by_label)  # copy

    person_count  = counts.get("person", 0)
    vehicle_count = counts.get("vehicle", 0)
    bag_count     = counts.get("bag", 0)

    # The mapping layer in yolo_detector emits `weapon` for verified
    # weapon-class detections (COCO `knife` OR weapon-model output)
    # and weapon class synonyms (revolver, gun, pistol, etc.).
    _WEAPON_CLASSES = {"weapon", "knife", "revolver", "Revolver", "shotgun", "Shotgun", "gun", "Gun", "pistol", "Pistol", "rifle", "Rifle", "handgun", "Handgun", "firearm", "Firearm", "grenade", "Grenade"}
    verified_weapon_count = sum(counts.get(k, 0) for k in _WEAPON_CLASSES)
    candidate_weapon_count = counts.get("candidate_weapon", 0)
    # Back-compat alias (downstream code reads `weapon_count`).
    weapon_count = verified_weapon_count

    total_objects = sum(counts.values())
    unique_labels = sorted(k for k, v in counts.items() if v > 0)
    avg_conf      = float(input_data.average_confidence)

    # --- Severity score -------------------------------------------------
    severity = _compute_severity(
        verified_weapon_count=verified_weapon_count,
        candidate_weapon_count=candidate_weapon_count,
        person_count=person_count,
        vehicle_count=vehicle_count,
        bag_count=bag_count,
        avg_conf=avg_conf,
    )
    severity_level = _severity_level(severity)

    # --- Suggested crime category ---------------------------------------
    # Only VERIFIED weapons count toward category rules. A "candidate"
    # alone cannot promote the scene to assault / robbery.
    present_labels = set(unique_labels)
    if verified_weapon_count > 0:
        present_labels.add("verified_weapon")
    suggested = _suggest_category(present_labels)

    # --- Key observations ----------------------------------------------
    observations = _build_observations(
        person_count=person_count,
        verified_weapon_count=verified_weapon_count,
        candidate_weapon_count=candidate_weapon_count,
        knife_count=counts.get("knife", 0),
        bottle_count=counts.get("bottle", 0),
        vehicle_count=vehicle_count,
        bag_count=bag_count,
        total_objects=total_objects,
        source_type=input_data.source_type,
        frame_count=input_data.frame_count,
    )

    # Threat only when we have a VERIFIED weapon — not just a hint.
    has_threat = verified_weapon_count > 0

    return EvidenceAnalysis(
        source_name=input_data.source_name,
        source_type=input_data.source_type,
        counts_by_label=counts,
        total_objects=total_objects,
        unique_labels=unique_labels,
        average_confidence=avg_conf,
        person_count=person_count,
        verified_weapon_count=verified_weapon_count,
        candidate_weapon_count=candidate_weapon_count,
        weapon_count=weapon_count,
        vehicle_count=vehicle_count,
        bag_count=bag_count,
        severity_score=severity,
        severity_level=severity_level,
        suggested_category=suggested,
        key_observations=observations,
        has_threat=has_threat,
        frame_count=input_data.frame_count,
    )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _compute_severity(
    verified_weapon_count: int,
    candidate_weapon_count: int,
    person_count: int,
    vehicle_count: int,
    bag_count: int,
    avg_conf: float,
) -> int:
    """
    Compute a 0-100 severity score from per-label counts and confidence.

    The score is a transparent linear combination of weights, clamped
    to [0, 100]. Verified weapons carry the full weight; candidates
    contribute a small hint so the UI can flag them but they never
    push severity to "critical" on their own.
    """
    w = SEVERITY_WEIGHTS
    score = 0.0
    if verified_weapon_count > 0:
        score += w["verified_weapon"] * (1 + 0.2 * (verified_weapon_count - 1))
    if candidate_weapon_count > 0:
        # A single candidate contributes the full "candidate" weight;
        # additional candidates add only 20% per extra.
        score += w["candidate_weapon"] * (1 + 0.2 * (candidate_weapon_count - 1))
    score += person_count * w["person"]
    if vehicle_count > 0:
        score += w["vehicle"]
    score += bag_count * w["bag"]
    if avg_conf > 0.85:
        score += w["high_conf"]
    return int(min(100, round(score)))


def _severity_level(score: int) -> str:
    """
    Map a 0-100 severity score to a level label.
    """
    for threshold, level in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


def _suggest_category(present_labels: set[str]) -> str:
    """
    Pick the first category whose required labels are all present.
    Falls back to "unknown" if no rule matches.
    """
    for label, required in CATEGORY_RULES:
        if required.issubset(present_labels):
            return label
    return "unknown"


def _build_observations(
    person_count: int,
    verified_weapon_count: int,
    candidate_weapon_count: int,
    knife_count: int,
    bottle_count: int,
    vehicle_count: int,
    bag_count: int,
    total_objects: int,
    source_type: str,
    frame_count: int,
) -> list[str]:
    """Generate short, factual observation strings."""
    obs: list[str] = []

    if total_objects == 0:
        obs.append("No crime-relevant objects detected in the provided media.")
        return obs

    obs.append(
        f"Detected {total_objects} crime-relevant object(s) "
        f"across {frame_count} frame(s)."
    )

    # Persons
    if person_count == 0:
        obs.append("No persons detected.")
    elif person_count == 1:
        obs.append("1 person detected at the scene.")
    else:
        obs.append(f"{person_count} persons detected at the scene.")

    # Weapons — verified and candidate are surfaced SEPARATELY so the
    # investigator (and the AI summary) can never blur the line.
    if verified_weapon_count == 0 and candidate_weapon_count == 0:
        obs.append("No verified weapons detected.")
    else:
        if verified_weapon_count > 0:
            obs.append(
                f"{verified_weapon_count} VERIFIED weapon(s) detected "
                f"(passed verification threshold)."
            )
        if candidate_weapon_count > 0:
            obs.append(
                f"{candidate_weapon_count} possible weapon candidate(s) "
                f"require manual verification (low confidence)."
            )
        if knife_count and verified_weapon_count:
            obs.append(f"Of which {knife_count} are confirmed knife(s).")

    # Vehicles
    if vehicle_count == 0:
        obs.append("No vehicles detected.")
    else:
        obs.append(f"{vehicle_count} vehicle(s) detected near the scene.")

    # Bags (potential theft indicator)
    if bag_count:
        obs.append(f"{bag_count} bag(s) detected (possible theft indicator).")

    # Source-aware note
    if source_type == "video":
        obs.append("Evidence is based on multiple sampled frames from the video clip.")
        if candidate_weapon_count and verified_weapon_count == 0:
            obs.append(
                "Single-frame weapon candidates were NOT promoted to verified; "
                "human review required."
            )

    return obs
