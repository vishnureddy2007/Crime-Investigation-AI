"""
Two-stage weapon verification layer (Phase 46).

The raw YOLO output is *not* trusted for weapon detection. This module
sits between the detector and `evidence_analyzer.analyze()` and
classifies each weapon-class detection as:

* **verified** — crossed `WEAPON_VERIFY_THRESHOLD` (or `WEAPON_HIGH_CONF_THRESHOLD`).
  Counted as a real weapon; drives severity and `has_threat`.
* **candidate** — labelled as a possible weapon but confidence is
  below the verify threshold. Surfaced in the UI for manual review;
  contributes a small hint to severity but never changes the
  suggested crime category.
* **non-weapon** — excluded from weapon counts entirely.

The verifier operates on `Detection` objects with their `.label` and
`.confidence` already set. It does NOT call into YOLO.

This is a pure-Python helper — testable without Streamlit or model
loading.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import (
    WEAPON_CONF_THRESHOLD,
    WEAPON_HIGH_CONF_THRESHOLD,
    WEAPON_VERIFY_THRESHOLD,
)


@dataclass(frozen=True)
class WeaponVerification:
    """The verified/candidate outcome for one weapon-class detection."""

    status: str            # "verified" | "candidate" | "non_weapon"
    label: str             # "weapon" | "knife" | "candidate_weapon" | original label
    confidence: float
    source: str            # "general" | "weapon-scan" | "weapon" | "threat-weapon" | "temporal"

    @property
    def is_verified(self) -> bool:
        return self.status == "verified"


# Labels the analyzer treats as weapons when present in `counts_by_label`.
# Labels the analyzer treats as weapons when present in `counts_by_label`.
WEAPON_LABELS: frozenset[str] = frozenset({
    "weapon", "knife", "revolver", "Revolver", "shotgun", "Shotgun",
    "gun", "Gun", "pistol", "Pistol", "rifle", "Rifle", "handgun", "Handgun",
    "firearm", "Firearm", "grenade", "Grenade", "candidate_weapon"
})


def classify_weapon_detection(
    label: str,
    confidence: float,
    source: str = "general",
    bbox: Any | None = None,
    img_width: float | None = None,
    img_height: float | None = None,
) -> tuple[str, str]:
    """
    Classify a detection into one of: 'WEAPON', 'NOT_WEAPON', 'UNCERTAIN'.
    Returns tuple of (final_state, post_mapping_label).
    """
    if label not in WEAPON_LABELS:
        return "NOT_WEAPON", label

    target_label = "weapon" if label != "knife" else "knife"

    # Box Quality & Size Validation (Supporting Signals)
    if bbox is not None and hasattr(bbox, "width") and hasattr(bbox, "height"):
        bw, bh = bbox.width, bbox.height
        area = bbox.area if hasattr(bbox, "area") else bw * bh
        aspect_ratio = (bw / max(1.0, bh)) if bw >= bh else (bh / max(1.0, bw))

        # Extreme artifact box (e.g. 1px line or tiny <16px dot with sub-0.70 confidence)
        if (bw < 4 or bh < 4 or area < 16.0 or aspect_ratio > 15.0) and confidence < WEAPON_HIGH_CONF_THRESHOLD:
            return "NOT_WEAPON", target_label

    # High Confidence Shortcut
    if confidence >= WEAPON_HIGH_CONF_THRESHOLD:
        return "WEAPON", target_label

    # Credibility Boost for weapon-trained models or temporal tracking
    source_boost = 0.10 if source in {"weapon", "threat-weapon", "temporal", "weapon-scan"} else 0.0
    effective_score = confidence + source_boost

    if effective_score >= WEAPON_VERIFY_THRESHOLD:
        return "WEAPON", target_label
    elif confidence >= WEAPON_CONF_THRESHOLD:
        return "UNCERTAIN", target_label
    else:
        return "NOT_WEAPON", target_label


def separate_detection_states(detections: list) -> dict[str, list]:
    """
    Categorize raw detections into separate structured lists:
      - raw_detections
      - weapon_candidates
      - verified_weapons
      - rejected_detections
      - uncertain_detections
    """
    raw_list = list(detections)
    candidates = []
    verified = []
    rejected = []
    uncertain = []

    for d in raw_list:
        label = getattr(d, "label", getattr(d, "class_name", ""))
        conf = getattr(d, "confidence", 0.0)
        source = getattr(d, "source", "general")
        bbox = getattr(d, "bbox", None)

        if label in WEAPON_LABELS or getattr(d, "class_name", "").lower() in WEAPON_LABELS:
            candidates.append(d)
            state, target_label = classify_weapon_detection(label, conf, source, bbox)

            if state == "WEAPON":
                # Ensure weapon_status is 'verified'
                if hasattr(d, "weapon_status"):
                    d.weapon_status = "verified"
                verified.append(d)
            elif state == "UNCERTAIN":
                if hasattr(d, "weapon_status"):
                    d.weapon_status = "candidate"
                uncertain.append(d)
            else:
                if hasattr(d, "weapon_status"):
                    d.weapon_status = "rejected"
                rejected.append(d)

    return {
        "raw_detections": raw_list,
        "weapon_candidates": candidates,
        "verified_weapons": verified,
        "rejected_detections": rejected,
        "uncertain_detections": uncertain,
    }


def verify_detection(label: str, confidence: float, source: str = "general") -> WeaponVerification:
    """
    Classify a single weapon-class detection.
    """
    if label not in WEAPON_LABELS:
        return WeaponVerification(
            status="non_weapon",
            label=label,
            confidence=confidence,
            source=source,
        )

    state, target_label = classify_weapon_detection(label, confidence, source)
    if state == "WEAPON":
        status = "verified"
    elif state == "UNCERTAIN":
        status = "candidate"
    else:
        status = "non_weapon"

    return WeaponVerification(
        status=status,
        label=target_label if status == "verified" else label,
        confidence=confidence,
        source=source,
    )


def verify_all(detections: list) -> list[WeaponVerification]:
    """Run `verify_detection` over a list of `Detection`-like objects."""
    return [
        verify_detection(d.label, d.confidence, getattr(d, "source", "general"))
        for d in detections
    ]


def verified_counts(detections: list) -> tuple[int, int]:
    """
    Convenience: return `(verified_weapon_count, candidate_weapon_count)`
    """
    verified = 0
    candidate = 0
    for v in verify_all(detections):
        if v.status == "verified":
            verified += 1
        elif v.status == "candidate":
            candidate += 1
    return verified, candidate


def apply_to_detections(detections: list) -> list:
    """
    Mutate a list of `Detection` objects: return a NEW list with verified weapon labels.
    """
    from models.schemas import Detection  # local import to avoid cycle

    out: list[Detection] = []
    for d in detections:
        state, target_label = classify_weapon_detection(
            d.label, d.confidence, getattr(d, "source", "general"), d.bbox
        )
        new_label = d.label
        status = "non_weapon"

        if d.label in WEAPON_LABELS or d.class_name.lower() in WEAPON_LABELS:
            if state == "WEAPON":
                new_label = "weapon" if d.label != "knife" else "knife"
                status = "verified"
            elif state == "UNCERTAIN":
                new_label = "candidate_weapon"
                status = "candidate"
            else:
                status = "rejected"

        out.append(
            Detection(
                class_name=d.class_name,
                label=new_label,
                confidence=d.confidence,
                bbox=d.bbox,
                source=d.source,
                weapon_status=status,
            )
        )
    return out


# ----------------------------------------------------------------------
# Phase 51 — human-review overrides (the user's decision is final)
# ----------------------------------------------------------------------
def apply_human_review_to_analysis(
    analysis,
    human_decisions: dict[str, str],
):
    """
    Return a NEW `EvidenceAnalysis` with weapon-class detections
    suppressed when the human reviewer REJECTED them.

    Parameters
    ----------
    analysis         : `EvidenceAnalysis` (or a duck-typed equivalent).
    human_decisions  : `{detection_key: decision}` mapping. `decision`
                       is one of "CONFIRM" / "REJECT" / "UNCERTAIN".
                       Keys are constructed by the caller — typically
                       ``f"{label}@{confidence:.2f}"`` or any stable
                       per-detection string. REJECTED keys remove
                       every matching weapon-class detection from
                       the analysis; CONFIRMED/UNCERTAIN do nothing
                       here (they're informational).

    Returns
    -------
    A new `EvidenceAnalysis` with:
      * `detections` filtered: REJECTED detections removed.
      * `counts_by_label` updated: every weapon-class count
         decremented by the number of REJECTED hits.
      * `has_threat` flipped to False if all verified weapons are
         rejected.
      * `severity_score` post-processed through `analysis.with_counts`
         if available (so removing verified weapons actually lowers
         severity instead of leaving a stale number).
    """
    if analysis is None:
        return analysis
    if not human_decisions:
        return analysis

    rejected_labels = {
        key.split("@", 1)[0] for key, dec in human_decisions.items()
        if dec == "REJECT"
    }
    if not rejected_labels:
        return analysis

    detections = getattr(analysis, "detections", None)
    if not detections:
        new_counts = dict(getattr(analysis, "counts_by_label", {}) or {})
        for label in rejected_labels:
            if label in new_counts:
                new_counts[label] = 0

        new_v_weapon = new_counts.get("weapon", 0) + new_counts.get("knife", 0)
        new_c_weapon = new_counts.get("candidate_weapon", 0)
        has_threat = bool(getattr(analysis, "has_threat", False) and new_v_weapon > 0)
        try:
            from dataclasses import replace
            return replace(
                analysis,
                counts_by_label=new_counts,
                has_threat=has_threat,
                verified_weapon_count=new_v_weapon,
                candidate_weapon_count=new_c_weapon,
                weapon_count=new_v_weapon,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            from core.logging import get_logger
            get_logger(__name__).warning("dataclasses.replace failed: %s", exc)
            analysis.counts_by_label = new_counts
            analysis.has_threat = has_threat
            analysis.verified_weapon_count = new_v_weapon
            analysis.candidate_weapon_count = new_c_weapon
            analysis.weapon_count = new_v_weapon
            return analysis

    new_detections = []
    removed_by_label: dict[str, int] = {}
    for d in detections:
        label = getattr(d, "label", "")
        key = f"{label}@{getattr(d, 'confidence', 0.0):.2f}"
        decision = human_decisions.get(key)
        if decision == "REJECT" or (label in rejected_labels and decision == "REJECT"):
            removed_by_label[label] = removed_by_label.get(label, 0) + 1
            continue
        new_detections.append(d)

    # Rebuild counts_by_label from the surviving detections so it stays
    # consistent with `detections`. Verified-only weapons are removed
    # fully; candidate weapons, too (the human has the final word).
    new_counts: dict[str, int] = {}
    for d in new_detections:
        new_counts[d.label] = new_counts.get(d.label, 0) + 1

    new_v_weapon = new_counts.get("weapon", 0) + new_counts.get("knife", 0)
    new_c_weapon = new_counts.get("candidate_weapon", 0)
    new_person = new_counts.get("person", 0)

    # If no verified weapons remain, the threat flag must be cleared.
    has_threat = bool(
        getattr(analysis, "has_threat", False)
        and new_v_weapon > 0
    )

    # Rebuild the dataclass with the new fields.
    try:
        from dataclasses import replace, fields
        valid_fields = {f.name for f in fields(analysis)}
        kwargs = {
            "counts_by_label": new_counts,
            "has_threat": has_threat,
            "verified_weapon_count": new_v_weapon,
            "candidate_weapon_count": new_c_weapon,
            "weapon_count": new_v_weapon,
            "person_count": new_person,
        }
        if "detections" in valid_fields:
            kwargs["detections"] = new_detections

        updated_obj = replace(analysis, **kwargs)
        setattr(updated_obj, "detections", new_detections)
        return updated_obj
    except (TypeError, ValueError) as exc:
        from core.logging import get_logger
        get_logger(__name__).warning("dataclasses.replace failed on analysis object: %s", exc)
        setattr(analysis, "detections", new_detections)
        try:
            analysis.counts_by_label = new_counts
            analysis.has_threat = has_threat
            analysis.verified_weapon_count = new_v_weapon
            analysis.candidate_weapon_count = new_c_weapon
            analysis.weapon_count = new_v_weapon
            analysis.person_count = new_person
        except (AttributeError, TypeError) as attr_exc:
            get_logger(__name__).warning("Failed to set mutated attributes on analysis object: %s", attr_exc)
        return analysis

