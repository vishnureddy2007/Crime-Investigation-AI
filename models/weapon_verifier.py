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

    status: str            # "verified" | "candidate" | "non_weapon" | "rejected"
    label: str             # "weapon" | "knife" | "candidate_weapon" | original label
    confidence: float
    source: str            # "general" | "weapon-scan" | "weapon" | "threat-weapon" | "temporal"
    verification_reason: str = ""

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

LOOKALIKE_NON_WEAPON_CLASSES: frozenset[str] = frozenset({
    "bottle", "bottles", "scissors", "baseball bat", "baseball_bat",
    "toy", "toys", "shadow", "shadows", "cell phone", "phone",
    "remote", "cup", "book", "umbrella", "chair", "bench"
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
        except (ImportError, RuntimeError, OSError, ValueError, TypeError, AttributeError):
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
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
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
    return_reason: bool = False,
) -> tuple[str, str] | tuple[str, str, str]:
    """
    Multi-stage weapon verification logic (Requirement 1, 2, 3).
    Ensures model outputs are treated as candidates and never verified
    based solely on high confidence or image-shape shortcuts.
    Returns (state, target_label) or (state, target_label, reason).
    final_state: 'WEAPON' | 'NOT_WEAPON' | 'UNCERTAIN'
    """
    raw_label = (class_name or label or "").strip()
    lbl_lower = raw_label.lower()

    # Rule 0: Reject lookalikes and unrelated classes immediately
    if (
        lbl_lower in LOOKALIKE_NON_WEAPON_CLASSES
        or (class_name and class_name.lower() in LOOKALIKE_NON_WEAPON_CLASSES)
    ):
        reason = "LOOKALIKE_OBJECT_REJECTED"
        return ("NOT_WEAPON", label, reason) if return_reason else ("NOT_WEAPON", label)

    if label not in WEAPON_LABELS and lbl_lower not in WEAPON_LABELS:
        reason = "UNSUPPORTED_CLASS"
        return ("NOT_WEAPON", label, reason) if return_reason else ("NOT_WEAPON", label)

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

        # Rejection Rule 1A: Tiny artifact box (< 6px or area < 25px)
        if bw < 6 or bh < 6 or area < 25.0:
            reason = "TINY_CROP"
            return ("NOT_WEAPON", target_label, reason) if return_reason else ("NOT_WEAPON", target_label)

        # Rejection Rule 1B: Extreme aspect ratio (> 8.0)
        if aspect_ratio > 8.0:
            reason = "EXTREME_ASPECT_RATIO"
            return ("NOT_WEAPON", target_label, reason) if return_reason else ("NOT_WEAPON", target_label)

        # Rejection Rule 1C: Excessive screen area (> 50% screen)
        if rel_area is not None and rel_area > 0.50:
            reason = "EXCESSIVE_SCREEN_AREA"
            return ("NOT_WEAPON", target_label, reason) if return_reason else ("NOT_WEAPON", target_label)

    # Stage 2: Crop Visual Feature Analysis (Runs for ALL candidates with crops)
    visual_boost, visual_diag = verify_weapon_crop_visual(image, bbox, class_name=subtype, confidence=confidence)
    if visual_diag == "FLAT_GRADIENT_NOISE":
        reason = "FLAT_GRADIENT_NOISE"
        return ("NOT_WEAPON", target_label, reason) if return_reason else ("NOT_WEAPON", target_label)
    if visual_diag == "EXTREME_ASPECT_RATIO":
        reason = "EXTREME_ASPECT_RATIO"
        return ("NOT_WEAPON", target_label, reason) if return_reason else ("NOT_WEAPON", target_label)
    if visual_diag == "TINY_CROP":
        reason = "TINY_CROP"
        return ("NOT_WEAPON", target_label, reason) if return_reason else ("NOT_WEAPON", target_label)

    # Stage 3: Multi-Signal Verification Score Computation
    score = confidence + visual_boost

    # Source credibility boost (specialized forensic models)
    if source in {"weapon", "threat-weapon", "temporal"} and confidence >= 0.32:
        score += 0.08

    # Specific subclass boost (specific firearm classes have lower false positive priors)
    if subtype in {"revolver", "pistol", "handgun", "rifle", "shotgun"}:
        score += 0.07

    # Shape & scale quality boost
    if 1.1 <= aspect_ratio <= 5.5 and (rel_area is None or 0.0005 <= rel_area <= 0.25):
        score += 0.05

    # Stage 4: Threshold Evaluation
    if score >= WEAPON_VERIFY_THRESHOLD and confidence >= 0.32:
        reason = visual_diag if visual_diag in {"FIREARM_STRUCTURE_VALIDATED"} else "MULTI_SIGNAL_STRUCTURE_VERIFIED"
        return ("WEAPON", target_label, reason) if return_reason else ("WEAPON", target_label)
    elif confidence >= WEAPON_CONF_THRESHOLD:
        reason = "INSUFFICIENT_VISUAL_EVIDENCE"
        return ("UNCERTAIN", target_label, reason) if return_reason else ("UNCERTAIN", target_label)
    else:
        reason = "LOW_CONFIDENCE_OR_INVALID_SHAPE"
        return ("NOT_WEAPON", target_label, reason) if return_reason else ("NOT_WEAPON", target_label)


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

    # 2. Class-aware candidate deduplication (IoU > 0.35 or (IoS > 0.70 and IoU > 0.20))
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
                    if iou_val > 0.35 or (ios_val > 0.70 and iou_val > 0.20):
                        is_dup = True
                        if hasattr(cand, "weapon_status"):
                            cand.weapon_status = "rejected"
                        if hasattr(cand, "verification_reason"):
                            cand.verification_reason = "DUPLICATE_DETECTION"
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

        res = classify_weapon_detection(lbl, conf, source, bbox, class_name=cls_name, image=image, return_reason=True)
        state, target_label, reason = res  # type: ignore[misc]

        if hasattr(d, "verification_reason"):
            d.verification_reason = reason
        lbl_key = f"{lbl}@{conf:.2f}"
        reasons[lbl_key] = reason

        if state == "WEAPON":
            if hasattr(d, "weapon_status"):
                d.weapon_status = "verified"
            verified.append(d)
            logger.info(
                "[WEAPON] Candidate #%d VERIFIED: class=%s label=%s yolo_conf=%.2f state=VERIFIED_WEAPON reason=%s",
                idx, cls_name, target_label, conf, reason
            )
        elif state == "UNCERTAIN":
            if hasattr(d, "weapon_status"):
                d.weapon_status = "candidate"
            uncertain.append(d)
            logger.info(
                "[WEAPON] Candidate #%d UNCERTAIN: class=%s yolo_conf=%.2f state=UNCERTAIN (candidate) reason=%s",
                idx, cls_name, conf, reason
            )
        else:
            if hasattr(d, "weapon_status"):
                d.weapon_status = "rejected"
            rejected.append(d)
            logger.info(
                "[WEAPON] Candidate #%d REJECTED: class=%s yolo_conf=%.2f state=NOT_WEAPON reason=%s",
                idx, cls_name, conf, reason
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
            verification_reason="UNSUPPORTED_CLASS",
        )

    res = classify_weapon_detection(label, confidence, source, return_reason=True)
    state, target_label, reason = res  # type: ignore[misc]

    if state == "WEAPON":
        status = "verified"
        resolved_label = target_label
    elif state == "UNCERTAIN":
        status = "candidate"
        resolved_label = "candidate_weapon" if (label in WEAPON_LABELS or label.lower() in WEAPON_LABELS) else label
    else:
        status = "non_weapon"
        resolved_label = label

    return WeaponVerification(
        status=status,
        label=resolved_label,
        confidence=confidence,
        source=source,
        verification_reason=reason,
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


def apply_to_detections(detections: list, image: Any | None = None) -> list:
    """
    Mutate a list of `Detection` objects: return a NEW list with verified weapon labels and statuses.
    """
    from models.schemas import Detection  # local import to avoid cycle

    out: list[Detection] = []
    for d in detections:
        cls_name = getattr(d, "class_name", "")
        raw_label = d.label or cls_name
        res = classify_weapon_detection(
            raw_label, d.confidence, getattr(d, "source", "general"), d.bbox, class_name=cls_name, image=image, return_reason=True
        )
        state, target_label, reason = res  # type: ignore[misc]
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
                verification_reason=reason,
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
    confirmed or rejected by human review (Requirement 4).

    When an investigator rejects a candidate/verified weapon:
    - Excluded from detections and counts_by_label.
    - Excluded from verified_weapon_count and candidate_weapon_count.
    - Excluded from severity scoring (severity recalculated).
    - If verified weapon count drops to 0, has_threat becomes False.

    When an investigator confirms a candidate weapon:
    - Promoted to verified weapon.
    - Verified weapon count incremented.
    - Severity score and threat updated.
    """
    if analysis is None:
        return analysis
    if not human_decisions:
        return analysis

    has_active_decisions = any(dec in {"CONFIRM", "REJECT"} for dec in human_decisions.values())
    if not has_active_decisions:
        return analysis

    rejected_labels = {
        key.split("@", 1)[0] for key, dec in human_decisions.items()
        if dec == "REJECT"
    }

    detections = getattr(analysis, "detections", None)
    new_counts = dict(getattr(analysis, "counts_by_label", {}) or {})

    new_detections = []
    if detections:
        for d in detections:
            label = getattr(d, "label", "")
            conf = getattr(d, "confidence", 0.0)
            key = f"{label}@{conf:.2f}"
            dec = human_decisions.get(key)
            if not dec and hasattr(d, "class_name"):
                dec = human_decisions.get(f"{d.class_name}@{conf:.2f}")

            if dec == "REJECT" or (label in rejected_labels and dec == "REJECT"):
                continue

            if dec == "CONFIRM":
                if hasattr(d, "weapon_status"):
                    d.weapon_status = "verified"
                if hasattr(d, "verification_reason"):
                    d.verification_reason = "CONFIRMED_BY_INVESTIGATOR"
                if label == "candidate_weapon":
                    cls_n = getattr(d, "class_name", "weapon")
                    d.label = "knife" if cls_n == "knife" else "weapon"

            new_detections.append(d)

        new_counts = {}
        for d in new_detections:
            new_counts[d.label] = new_counts.get(d.label, 0) + 1
    else:
        for key, dec in human_decisions.items():
            lbl = key.split("@", 1)[0]
            if dec == "REJECT":
                if lbl in new_counts:
                    new_counts[lbl] = 0
            elif dec == "CONFIRM":
                if lbl == "candidate_weapon":
                    new_counts["candidate_weapon"] = max(0, new_counts.get("candidate_weapon", 0) - 1)
                    new_counts["weapon"] = new_counts.get("weapon", 0) + 1

    _WEAPON_CLASSES = {
        "weapon", "knife", "revolver", "Revolver", "shotgun", "Shotgun",
        "gun", "Gun", "pistol", "Pistol", "rifle", "Rifle", "handgun", "Handgun",
        "firearm", "Firearm", "grenade", "Grenade"
    }
    new_v_weapon = sum(new_counts.get(k, 0) for k in _WEAPON_CLASSES)
    new_c_weapon = new_counts.get("candidate_weapon", 0)
    new_person = new_counts.get("person", 0)
    new_vehicle = new_counts.get("vehicle", 0)
    new_bag = new_counts.get("bag", 0)
    total_objects = sum(new_counts.values())

    has_threat = new_v_weapon > 0

    from models.evidence_analyzer import (
        _compute_severity,
        _severity_level,
        _suggest_category,
        _build_observations,
    )
    avg_conf = getattr(analysis, "average_confidence", 0.0)
    new_severity = _compute_severity(
        verified_weapon_count=new_v_weapon,
        candidate_weapon_count=new_c_weapon,
        person_count=new_person,
        vehicle_count=new_vehicle,
        bag_count=new_bag,
        avg_conf=avg_conf,
    )
    new_severity_level = _severity_level(new_severity)
    present_labels = set(new_counts.keys())
    if new_v_weapon > 0:
        present_labels.add("verified_weapon")
    new_suggested = _suggest_category(present_labels)
    new_obs = _build_observations(
        person_count=new_person,
        verified_weapon_count=new_v_weapon,
        candidate_weapon_count=new_c_weapon,
        knife_count=new_counts.get("knife", 0),
        bottle_count=new_counts.get("bottle", 0),
        vehicle_count=new_vehicle,
        bag_count=new_bag,
        total_objects=total_objects,
        source_type=getattr(analysis, "source_type", "image"),
        frame_count=getattr(analysis, "frame_count", 1),
    )

    try:
        from dataclasses import replace, fields
        valid_fields = {f.name for f in fields(analysis)} if hasattr(analysis, "__dataclass_fields__") else set()
        kwargs: dict[str, Any] = {}
        if "counts_by_label" in valid_fields:
            kwargs["counts_by_label"] = new_counts
        if "has_threat" in valid_fields:
            kwargs["has_threat"] = has_threat
        if "verified_weapon_count" in valid_fields:
            kwargs["verified_weapon_count"] = new_v_weapon
        if "candidate_weapon_count" in valid_fields:
            kwargs["candidate_weapon_count"] = new_c_weapon
        if "weapon_count" in valid_fields:
            kwargs["weapon_count"] = new_v_weapon
        if "person_count" in valid_fields:
            kwargs["person_count"] = new_person
        if "vehicle_count" in valid_fields:
            kwargs["vehicle_count"] = new_vehicle
        if "bag_count" in valid_fields:
            kwargs["bag_count"] = new_bag
        if "total_objects" in valid_fields:
            kwargs["total_objects"] = total_objects
        if "severity_score" in valid_fields:
            kwargs["severity_score"] = new_severity
        if "severity_level" in valid_fields:
            kwargs["severity_level"] = new_severity_level
        if "suggested_category" in valid_fields:
            kwargs["suggested_category"] = new_suggested
        if "key_observations" in valid_fields:
            kwargs["key_observations"] = new_obs
        if "human_decisions" in valid_fields:
            kwargs["human_decisions"] = dict(human_decisions)
        if "detections" in valid_fields:
            kwargs["detections"] = new_detections

        updated_obj = replace(analysis, **kwargs)
        if hasattr(updated_obj, "detections"):
            setattr(updated_obj, "detections", new_detections)
        return updated_obj
    except (TypeError, ValueError, AttributeError) as exc:
        from core.logging import get_logger
        get_logger(__name__).warning("dataclasses.replace fallback on analysis object: %s", exc)
        if hasattr(analysis, "counts_by_label"):
            analysis.counts_by_label = new_counts
        if hasattr(analysis, "has_threat"):
            analysis.has_threat = has_threat
        if hasattr(analysis, "verified_weapon_count"):
            analysis.verified_weapon_count = new_v_weapon
        if hasattr(analysis, "candidate_weapon_count"):
            analysis.candidate_weapon_count = new_c_weapon
        if hasattr(analysis, "weapon_count"):
            analysis.weapon_count = new_v_weapon
        if hasattr(analysis, "person_count"):
            analysis.person_count = new_person
        if hasattr(analysis, "severity_score"):
            analysis.severity_score = new_severity
        if hasattr(analysis, "severity_level"):
            analysis.severity_level = new_severity_level
        if hasattr(analysis, "suggested_category"):
            analysis.suggested_category = new_suggested
        if hasattr(analysis, "human_decisions"):
            analysis.human_decisions = dict(human_decisions)
        if hasattr(analysis, "detections"):
            setattr(analysis, "detections", new_detections)
        return analysis

