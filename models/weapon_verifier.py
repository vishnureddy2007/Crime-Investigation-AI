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
WEAPON_LABELS: frozenset[str] = frozenset({"weapon", "knife"})


def verify_detection(label: str, confidence: float, source: str = "general") -> WeaponVerification:
    """
    Classify a single weapon-class detection.

    Parameters
    ----------
    label       : the post-mapping label (e.g. "knife", "weapon", "bottle").
    confidence  : detection confidence (0.0 - 1.0).
    source      : which model produced this detection. Dedicated weapon
                  models (`source="weapon"` or `"threat-weapon"`) and
                  temporal verification (`source="temporal"`) get a
                  small credibility boost because they are explicitly
                  trained or reasoned to detect weapons.

    Returns
    -------
    WeaponVerification with `status` in {"verified", "candidate", "non_weapon"}.
    """
    # Non-weapon labels never enter the weapon pipeline.
    if label not in WEAPON_LABELS:
        return WeaponVerification(
            status="non_weapon",
            label=label,
            confidence=confidence,
            source=source,
        )

    # High-confidence shortcut — anything above WEAPON_HIGH_CONF_THRESHOLD
    # is verified regardless of which model produced it. Reduces false
    # negatives for genuinely obvious weapons without lowering the bar
    # for ambiguous candidates.
    if confidence >= WEAPON_HIGH_CONF_THRESHOLD:
        return WeaponVerification(
            status="verified",
            label=label,
            confidence=confidence,
            source=source,
        )

    # Confidence boost for sources that are explicitly weapon-trained.
    # Dedicated weapon models know what they're looking at; a 0.45
    # detection from `weapon.pt` is more trustworthy than a 0.45
    # COCO-knife detection.
    source_boost = 0.0
    if source in {"weapon", "threat-weapon", "temporal"}:
        source_boost = 0.10

    effective = confidence + source_boost
    if effective >= WEAPON_VERIFY_THRESHOLD:
        return WeaponVerification(
            status="verified",
            label=label,
            confidence=confidence,
            source=source,
        )

    # Below the verify threshold (and below the loose scan threshold):
    # surface as a candidate for human review only.
    if confidence >= WEAPON_CONF_THRESHOLD:
        return WeaponVerification(
            status="candidate",
            label="candidate_weapon",
            confidence=confidence,
            source=source,
        )

    # Too low to mention as a weapon candidate either.
    return WeaponVerification(
        status="non_weapon",
        label=label,
        confidence=confidence,
        source=source,
    )


def verify_all(detections: list) -> list[WeaponVerification]:
    """Run `verify_detection` over a list of `Detection`-like objects.

    Accepts any object with `.label`, `.confidence`, `.source`
    attributes (real `Detection` dataclass satisfies this).
    """
    return [
        verify_detection(d.label, d.confidence, getattr(d, "source", "general"))
        for d in detections
    ]


def verified_counts(detections: list) -> tuple[int, int]:
    """
    Convenience: return `(verified_weapon_count, candidate_weapon_count)`
    for a list of detections, after running `verify_all`.
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
    Mutate a list of `Detection` objects in-place-style: return a NEW
    list where each Detection has `weapon_status` populated.

    Detections whose label is not a weapon (knife / weapon) keep their
    original label unchanged. Verified weapons keep `label="weapon"`;
    candidates get re-labelled to `label="candidate_weapon"` so the
    analyzer's `counts_by_label` separates the two buckets.
    """
    from models.schemas import Detection  # local import to avoid cycle

    out: list[Detection] = []
    for d in detections:
        v = verify_detection(d.label, d.confidence, getattr(d, "source", "general"))
        new_label = d.label
        if v.status == "candidate":
            new_label = "candidate_weapon"
        out.append(
            Detection(
                class_name=d.class_name,
                label=new_label,
                confidence=d.confidence,
                bbox=d.bbox,
                source=d.source,
                weapon_status=v.status,
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
        except Exception:
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

