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
WEAPON_LABELS: frozenset[str] = frozenset({
    "weapon", "Weapon", "knife", "Knife", "revolver", "Revolver",
    "shotgun", "Shotgun", "gun", "Gun", "pistol", "Pistol",
    "rifle", "Rifle", "handgun", "Handgun", "firearm", "Firearm",
    "grenade", "Grenade", "candidate_weapon"
})

CANONICAL_WEAPON_SUBTYPES: dict[str, str] = {
    "revolver": "revolver",
    "Revolver": "revolver",
    "pistol": "pistol",
    "Pistol": "pistol",
    "handgun": "handgun",
    "Handgun": "handgun",
    "rifle": "rifle",
    "Rifle": "rifle",
    "shotgun": "shotgun",
    "Shotgun": "shotgun",
    "firearm": "firearm",
    "Firearm": "firearm",
    "gun": "gun",
    "Gun": "gun",
    "knife": "knife",
    "Knife": "knife",
    "grenade": "grenade",
    "Grenade": "grenade",
    "weapon": "weapon",
    "Weapon": "weapon",
}


@dataclass
class WeaponVerificationResult:
    """Canonical verification output for a set of detections."""

    raw_detections: list[Any]
    weapon_candidates: list[Any]
    verified_weapons: list[Any]
    uncertain_detections: list[Any]
    rejected_detections: list[Any]
    verified_count: int
    candidate_count: int
    reasons: dict[str, str]
    weapon_status: str  # "VERIFIED_WEAPON_PRESENT" | "NO_VERIFIED_WEAPON"
    canonical_subtype: str  # "revolver" | "pistol" | "handgun" | "rifle" | "shotgun" | "knife" | "weapon"


def _compute_bbox_iou(a: Any, b: Any) -> float:
    """Compute IoU between two bounding box objects or dicts."""
    ax1 = getattr(a, "x1", a.get("x1", 0) if isinstance(a, dict) else 0)
    ay1 = getattr(a, "y1", a.get("y1", 0) if isinstance(a, dict) else 0)
    ax2 = getattr(a, "x2", a.get("x2", 0) if isinstance(a, dict) else 0)
    ay2 = getattr(a, "y2", a.get("y2", 0) if isinstance(a, dict) else 0)

    bx1 = getattr(b, "x1", b.get("x1", 0) if isinstance(b, dict) else 0)
    by1 = getattr(b, "y1", b.get("y1", 0) if isinstance(b, dict) else 0)
    bx2 = getattr(b, "x2", b.get("x2", 0) if isinstance(b, dict) else 0)
    by2 = getattr(b, "y2", b.get("y2", 0) if isinstance(b, dict) else 0)

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = getattr(a, "area", (ax2 - ax1) * (ay2 - ay1))
    area_b = getattr(b, "area", (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _compute_bbox_ios(a: Any, b: Any) -> float:
    """Compute Intersection-over-Smaller box area."""
    ax1 = getattr(a, "x1", a.get("x1", 0) if isinstance(a, dict) else 0)
    ay1 = getattr(a, "y1", a.get("y1", 0) if isinstance(a, dict) else 0)
    ax2 = getattr(a, "x2", a.get("x2", 0) if isinstance(a, dict) else 0)
    ay2 = getattr(a, "y2", a.get("y2", 0) if isinstance(a, dict) else 0)

    bx1 = getattr(b, "x1", b.get("x1", 0) if isinstance(b, dict) else 0)
    by1 = getattr(b, "y1", b.get("y1", 0) if isinstance(b, dict) else 0)
    bx2 = getattr(b, "x2", b.get("x2", 0) if isinstance(b, dict) else 0)
    by2 = getattr(b, "y2", b.get("y2", 0) if isinstance(b, dict) else 0)

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = getattr(a, "area", (ax2 - ax1) * (ay2 - ay1))
    area_b = getattr(b, "area", (bx2 - bx1) * (by2 - by1))
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0 else 0.0


def verify_weapon_crop_visual(
    image: Any | None,
    bbox: Any | None,
    class_name: str = "weapon",
    confidence: float = 0.0,
) -> tuple[float, str]:
    """
    Perform visual crop verification on candidate image region.
    Returns (score_delta, diagnostic_reason).
    """
    if image is None or bbox is None:
        return 0.0, "NO_IMAGE_CROP"

    try:
        import numpy as np
        from PIL import Image

        if not isinstance(image, Image.Image):
            return 0.0, "INVALID_IMAGE_FORMAT"

        img_w, img_h = image.size
        if hasattr(bbox, "x1"):
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            return 0.0, "INVALID_BBOX"

        # Contextual padding (15%)
        pad_w = (x2 - x1) * 0.15
        pad_h = (y2 - y1) * 0.15
        cx1 = max(0, int(x1 - pad_w))
        cy1 = max(0, int(y1 - pad_h))
        cx2 = min(img_w, int(x2 + pad_w))
        cy2 = min(img_h, int(y2 + pad_h))

        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            return -0.15, "TINY_CROP"

        crop = image.crop((cx1, cy1, cx2, cy2))
        np_crop = np.array(crop.convert("RGB"))
        gray = np.mean(np_crop, axis=2).astype(np.uint8)

        try:
            import cv2
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        except Exception:
            gx, gy = np.gradient(gray.astype(float))
            grad = np.sqrt(gx**2 + gy**2)
            edge_ratio = float(np.count_nonzero(grad > 25.0)) / float(grad.size)

        cw = max(1, cx2 - cx1)
        ch = max(1, cy2 - cy1)
        aspect = max(cw, ch) / max(1.0, min(cw, ch))

        if edge_ratio >= 0.04 and 1.1 <= aspect <= 6.0:
            return 0.10, "FIREARM_STRUCTURE_VALIDATED"
        elif edge_ratio < 0.015:
            return -0.12, "FLAT_GRADIENT_NOISE"
        elif aspect > 8.0:
            return -0.15, "EXTREME_ASPECT_RATIO"
        else:
            return 0.02, "NEUTRAL_CROP_VISUAL"
    except Exception as exc:
        return 0.0, f"CROP_VERIFICATION_ERROR: {exc}"


def classify_weapon_detection(
    label: str,
    confidence: float,
    source: str = "general",
    bbox: Any | None = None,
    img_width: float | None = None,
    img_height: float | None = None,
    class_name: str | None = None,
    image: Any | None = None,
) -> tuple[str, str]:
    """
    Multi-stage weapon verification logic.
    Returns tuple of (final_state, post_mapping_label).
    final_state: 'WEAPON' | 'NOT_WEAPON' | 'UNCERTAIN'
    """
    raw_label = (class_name or label or "").strip()
    lbl_lower = raw_label.lower()

    if label not in WEAPON_LABELS and lbl_lower not in WEAPON_LABELS:
        return "NOT_WEAPON", label

    # Preserve canonical weapon subtype (e.g. revolver, pistol, rifle, knife)
    subtype = CANONICAL_WEAPON_SUBTYPES.get(raw_label, CANONICAL_WEAPON_SUBTYPES.get(lbl_lower, "weapon"))
    target_label = "weapon" if subtype != "knife" else "knife"

    # Stage 1: Bounding Box & Aspect Ratio Validation
    bw, bh, aspect_ratio, rel_area = 0.0, 0.0, 1.0, None
    if bbox is not None and hasattr(bbox, "width") and hasattr(bbox, "height"):
        bw, bh = bbox.width, bbox.height
        area = bbox.area if hasattr(bbox, "area") else bw * bh
        aspect_ratio = (max(bw, bh) / max(1.0, min(bw, bh)))

        if img_width is not None and img_height is not None and img_width > 0 and img_height > 0:
            rel_area = area / max(1.0, float(img_width * img_height))

        # Rejection Rule 1A: Tiny artifact box (< 6px or area < 25px) with sub-0.78 confidence
        if (bw < 6 or bh < 6 or area < 25.0) and confidence < WEAPON_HIGH_CONF_THRESHOLD:
            return "NOT_WEAPON", target_label

        # Rejection Rule 1B: Extreme aspect ratio (> 8.0) with sub-0.78 confidence
        if aspect_ratio > 8.0 and confidence < WEAPON_HIGH_CONF_THRESHOLD:
            return "NOT_WEAPON", target_label

        # Rejection Rule 1C: Excessive screen area (> 50% screen) with sub-0.78 confidence
        if rel_area is not None and rel_area > 0.50 and confidence < WEAPON_HIGH_CONF_THRESHOLD:
            return "NOT_WEAPON", target_label

    # Stage 2: High Confidence Shortcut
    if confidence >= WEAPON_HIGH_CONF_THRESHOLD:
        return "WEAPON", target_label

    # Stage 3: Crop Visual Feature Analysis
    visual_boost, visual_diag = verify_weapon_crop_visual(image, bbox, class_name=subtype, confidence=confidence)
    if visual_diag == "FLAT_GRADIENT_NOISE" and confidence < 0.65:
        return "NOT_WEAPON", target_label

    # Stage 4: Multi-Signal Verification Score Computation
    score = confidence + visual_boost

    # Source credibility boost (only for dedicated weapon models with conf >= 0.32)
    if source in {"weapon", "threat-weapon", "temporal"} and confidence >= 0.32:
        score += 0.08

    # Specific subclass boost (specific firearm classes like revolver/pistol have lower false positive priors)
    if subtype in {"revolver", "pistol", "handgun", "rifle", "shotgun"}:
        score += 0.07

    # Shape & scale quality boost
    if 1.1 <= aspect_ratio <= 5.5 and (rel_area is None or 0.0005 <= rel_area <= 0.25):
        score += 0.05

    # Stage 5: Threshold Evaluation
    if score >= WEAPON_VERIFY_THRESHOLD and confidence >= 0.32:
        return "WEAPON", target_label
    elif confidence >= WEAPON_CONF_THRESHOLD:
        return "UNCERTAIN", target_label
    else:
        return "NOT_WEAPON", target_label


def separate_detection_states(detections: list, image: Any | None = None) -> dict[str, Any]:
    """
    Categorize raw detections into separate structured lists with duplicate suppression:
      - raw_detections
      - weapon_candidates
      - verified_weapons
      - rejected_detections
      - uncertain_detections
    """
    from core.logging import get_logger
    logger = get_logger("weapon_verifier")

    raw_list = list(detections)
    candidates = []
    verified = []
    rejected = []
    uncertain = []
    reasons: dict[str, str] = {}

    # 1. Filter initial candidate weapon detections
    raw_candidates = []
    for d in raw_list:
        lbl = getattr(d, "label", getattr(d, "class_name", ""))
        cls_name = getattr(d, "class_name", "")
        if lbl in WEAPON_LABELS or cls_name in WEAPON_LABELS or cls_name.lower() in WEAPON_LABELS:
            raw_candidates.append(d)

    # 2. Class-aware candidate deduplication (IoU > 0.35 / IoS > 0.60)
    sorted_candidates = sorted(raw_candidates, key=lambda c: getattr(c, "confidence", 0.0), reverse=True)
    kept_candidates = []

    for cand in sorted_candidates:
        is_dup = False
        b1 = getattr(cand, "bbox", None)
        if b1 is not None:
            for kept in kept_candidates:
                b2 = getattr(kept, "bbox", None)
                if b2 is not None:
                    iou_val = _compute_bbox_iou(b1, b2)
                    ios_val = _compute_bbox_ios(b1, b2)
                    if iou_val > 0.35 or ios_val > 0.60:
                        is_dup = True
                        if hasattr(cand, "weapon_status"):
                            cand.weapon_status = "rejected"
                        rejected.append(cand)
                        lbl_key = getattr(cand, "label", getattr(cand, "class_name", "weapon"))
                        conf_val = getattr(cand, "confidence", 0.0)
                        reasons[f"{lbl_key}@{conf_val:.2f}"] = "DUPLICATE_DETECTION"
                        logger.info("[WEAPON] Candidate rejected: duplicate overlap (IoU=%.2f, IoS=%.2f)", iou_val, ios_val)
                        break
        if not is_dup:
            kept_candidates.append(cand)
            candidates.append(cand)

    # 3. Multi-stage verification for non-duplicate candidates
    for idx, d in enumerate(kept_candidates, start=1):
        lbl = getattr(d, "label", getattr(d, "class_name", ""))
        cls_name = getattr(d, "class_name", "")
        conf = getattr(d, "confidence", 0.0)
        source = getattr(d, "source", "general")
        bbox = getattr(d, "bbox", None)

        state, target_label = classify_weapon_detection(lbl, conf, source, bbox, class_name=cls_name, image=image)

        if state == "WEAPON":
            if hasattr(d, "weapon_status"):
                d.weapon_status = "verified"
            verified.append(d)
            logger.info(
                "[WEAPON] Candidate #%d VERIFIED: class=%s label=%s yolo_conf=%.2f state=VERIFIED_WEAPON",
                idx, cls_name, target_label, conf
            )
        elif state == "UNCERTAIN":
            if hasattr(d, "weapon_status"):
                d.weapon_status = "candidate"
            uncertain.append(d)
            reasons[f"{lbl}@{conf:.2f}"] = "INSUFFICIENT_VISUAL_EVIDENCE"
            logger.info(
                "[WEAPON] Candidate #%d UNCERTAIN: class=%s yolo_conf=%.2f state=UNCERTAIN (candidate)",
                idx, cls_name, conf
            )
        else:
            if hasattr(d, "weapon_status"):
                d.weapon_status = "rejected"
            rejected.append(d)
            reasons[f"{lbl}@{conf:.2f}"] = "LOW_CONFIDENCE_OR_INVALID_SHAPE"
            logger.info(
                "[WEAPON] Candidate #%d REJECTED: class=%s yolo_conf=%.2f state=NOT_WEAPON",
                idx, cls_name, conf
            )

    canonical_subtype = "weapon"
    if verified:
        subtypes = [getattr(v, "class_name", "").lower() for v in verified]
        for s in ["revolver", "pistol", "handgun", "rifle", "shotgun", "knife"]:
            if s in subtypes:
                canonical_subtype = s
                break

    return {
        "raw_detections": raw_list,
        "weapon_candidates": candidates,
        "verified_weapons": verified,
        "rejected_detections": rejected,
        "uncertain_detections": uncertain,
        "verified_count": len(verified),
        "candidate_count": len(uncertain),
        "weapon_status": "VERIFIED_WEAPON_PRESENT" if len(verified) > 0 else "NO_VERIFIED_WEAPON",
        "canonical_subtype": canonical_subtype,
        "reasons": reasons,
    }


def verify_detection(label: str, confidence: float, source: str = "general") -> WeaponVerification:
    """
    Classify a single weapon-class detection into WeaponVerification.
    """
    if label not in WEAPON_LABELS and label.lower() not in WEAPON_LABELS:
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
    Mutate a list of `Detection` objects: return a NEW list with verified weapon labels and statuses.
    """
    from models.schemas import Detection  # local import to avoid cycle

    out: list[Detection] = []
    for d in detections:
        cls_name = getattr(d, "class_name", "")
        raw_label = d.label or cls_name
        state, target_label = classify_weapon_detection(
            raw_label, d.confidence, getattr(d, "source", "general"), d.bbox, class_name=cls_name
        )
        new_label = d.label
        status = "non_weapon"

        if raw_label in WEAPON_LABELS or cls_name.lower() in WEAPON_LABELS:
            if state == "WEAPON":
                new_label = target_label
                status = "verified"
            elif state == "UNCERTAIN":
                new_label = "candidate_weapon"
                status = "candidate"
            else:
                new_label = d.label
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

