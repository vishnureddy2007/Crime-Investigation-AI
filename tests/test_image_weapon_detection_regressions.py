"""
Regression tests for Image Weapon Detection enhancements (Requirements 1-7).

Covers:
1. Overlapping duplicate boxes from multiple models (confidence-aware, class-aware NMS).
2. Separate nearby weapons that must remain distinct.
3. False detections on common lookalike objects (bottles, scissors, bats, cellphones, shadows).
4. Low-confidence candidates (marked "Possible weapon — review needed", reasons exposed).
5. Confirmed and rejected human reviews (counts, threat, severity, category recalculated).
6. Images with no weapon detections ("No verified weapon detected in this image.").
7. Configurable threshold range validation and trade-off documentation.
"""

from __future__ import annotations

import pytest
from PIL import Image

from config.settings import (
    validate_detection_thresholds,
    AUDITED_MODEL_REGISTRY,
    COCO_WEAPON_CLASSES,
)
from models.evidence_analyzer import analyze
from models.schemas import AnalysisInput, BoundingBox, Detection, EvidenceAnalysis
from models.weapon_verifier import (
    apply_human_review_to_analysis,
    apply_to_detections,
    classify_weapon_detection,
    separate_detection_states,
    verify_detection,
    verify_weapon_crop_visual,
)
from models.yolo_detector import _cross_model_nms


# ----------------------------------------------------------------------
# 1. Overlapping duplicate boxes from multiple models
# ----------------------------------------------------------------------
def test_regression_overlapping_duplicate_boxes_from_multiple_models():
    """
    Multiple models (weapon.pt, threat_weapon.pt, general scan) detect the
    same physical weapon with overlapping boxes. NMS must deduplicate to
    the strongest detection (highest confidence) and mark duplicates as rejected.
    """
    det_weapon = Detection(
        class_name="gun",
        label="weapon",
        confidence=0.88,
        bbox=BoundingBox(50, 50, 150, 150),
        source="weapon",
    )
    det_threat = Detection(
        class_name="pistol",
        label="weapon",
        confidence=0.76,
        bbox=BoundingBox(55, 55, 145, 145),
        source="threat-weapon",
    )
    det_scan = Detection(
        class_name="knife",
        label="knife",
        confidence=0.35,
        bbox=BoundingBox(52, 52, 148, 148),
        source="weapon-scan",
    )
    det_person = Detection(
        class_name="person",
        label="person",
        confidence=0.91,
        bbox=BoundingBox(200, 50, 300, 400),
        source="general",
    )

    raw_detections = [det_weapon, det_threat, det_scan, det_person]

    # 1. Cross-model NMS deduplication
    merged = _cross_model_nms(raw_detections, iou_threshold=0.35)
    weapon_merged = [d for d in merged if d.label in {"weapon", "knife"}]
    assert len(weapon_merged) == 1, "Must merge overlapping duplicate weapon boxes into exactly 1 detection"
    assert weapon_merged[0].confidence == 0.88, "Must retain strongest detection by confidence"
    assert len(merged) == 2, "Person must not be suppressed by weapon"

    # 2. Separation of states
    states = separate_detection_states(raw_detections)
    assert states["verified_count"] == 1
    assert len(states["verified_weapons"]) == 1
    assert states["verified_weapons"][0].confidence == 0.88

    # Suppressed duplicates must be in rejected_detections with reason DUPLICATE_DETECTION
    assert len(states["rejected_detections"]) >= 2
    for rej in states["rejected_detections"]:
        assert rej.weapon_status == "rejected"
        assert rej.verification_reason == "DUPLICATE_DETECTION"


# ----------------------------------------------------------------------
# 2. Separate nearby weapons that must remain distinct
# ----------------------------------------------------------------------
def test_regression_separate_nearby_weapons_remain_distinct():
    """
    Two distinct nearby weapons with minimal or no overlap must remain separate
    and both be verified, without suppressing each other.
    """
    # Weapon 1 on the left
    det1 = Detection(
        class_name="revolver",
        label="weapon",
        confidence=0.86,
        bbox=BoundingBox(20, 30, 80, 90),
        source="weapon",
    )
    # Weapon 2 on the right (adjacent with minimal overlap)
    det2 = Detection(
        class_name="pistol",
        label="weapon",
        confidence=0.82,
        bbox=BoundingBox(85, 30, 145, 90),
        source="weapon",
    )

    merged = _cross_model_nms([det1, det2], iou_threshold=0.35)
    assert len(merged) == 2, "Separate nearby weapons must remain distinct and not be merged"

    states = separate_detection_states(merged)
    assert states["verified_count"] == 2
    assert len(states["verified_weapons"]) == 2
    assert states["weapon_status"] == "VERIFIED_WEAPON_PRESENT"


# ----------------------------------------------------------------------
# 3. False detections on common lookalike objects
# ----------------------------------------------------------------------
def test_regression_false_detections_on_common_lookalike_objects():
    """
    Ensure unrelated lookalike objects (bottles, scissors, bats, cellphones, toys)
    and flat gradient noise/shadows are strictly rejected and never verified as weapons.
    """
    lookalikes = [
        ("bottle", 0.85),
        ("bottles", 0.70),
        ("scissors", 0.80),
        ("baseball bat", 0.78),
        ("toy", 0.75),
        ("cell phone", 0.82),
        ("shadow", 0.88),
    ]

    for name, conf in lookalikes:
        state, target_label, reason = classify_weapon_detection(
            label=name, confidence=conf, class_name=name, return_reason=True
        )
        assert state == "NOT_WEAPON", f"Lookalike {name} must be classified as NOT_WEAPON"
        assert reason in {"LOOKALIKE_OBJECT_REJECTED", "UNSUPPORTED_CLASS"}

    # Test flat gradient noise / shadow image crop
    # Create a uniform flat gray image (shadow/wall patch)
    flat_image = Image.new("RGB", (200, 200), color=(45, 45, 45))
    shadow_bbox = BoundingBox(20, 20, 120, 120)

    # Even with high model confidence (0.80), visual feature analysis must detect flat noise
    v_boost, v_diag = verify_weapon_crop_visual(flat_image, shadow_bbox, class_name="gun", confidence=0.80)
    assert v_diag == "FLAT_GRADIENT_NOISE", "Flat uniform patch must be flagged as FLAT_GRADIENT_NOISE"

    state, target_label, reason = classify_weapon_detection(
        label="weapon",
        confidence=0.80,
        bbox=shadow_bbox,
        image=flat_image,
        class_name="gun",
        return_reason=True,
    )
    assert state == "NOT_WEAPON", "Flat gradient noise must be rejected as NOT_WEAPON"
    assert reason == "FLAT_GRADIENT_NOISE", "Diagnostic reason must be FLAT_GRADIENT_NOISE"


# ----------------------------------------------------------------------
# 4. Low-confidence candidates
# ----------------------------------------------------------------------
def test_regression_low_confidence_candidates():
    """
    Low-confidence weapon detections (e.g. conf 0.32) without decisive visual validation
    must be held as UNCERTAIN candidates ("Possible weapon — review needed").
    They must NOT be counted as verified or escalate severity to critical.
    """
    det = Detection(
        class_name="gun",
        label="weapon",
        confidence=0.32,
        bbox=BoundingBox(10, 10, 70, 70),
        source="general",
    )

    state, target_label, reason = classify_weapon_detection(
        label=det.label, confidence=det.confidence, bbox=det.bbox, class_name=det.class_name, return_reason=True
    )
    assert state == "UNCERTAIN"
    assert reason == "INSUFFICIENT_VISUAL_EVIDENCE"

    states = separate_detection_states([det])
    assert states["verified_count"] == 0, "Candidates must not be counted in verified_count"
    assert states["candidate_count"] == 1
    assert len(states["verified_weapons"]) == 0
    assert len(states["uncertain_detections"]) == 1
    assert states["uncertain_detections"][0].weapon_status == "candidate"
    assert states["uncertain_detections"][0].verification_reason == "INSUFFICIENT_VISUAL_EVIDENCE"

    # In EvidenceAnalysis, candidate weapon does not set has_threat to True
    inp = AnalysisInput(
        source_name="candidate_scene.jpg",
        source_type="image",
        counts_by_label={"person": 1, "candidate_weapon": 1},
        average_confidence=0.32,
    )
    analysis = analyze(inp)
    assert analysis.verified_weapon_count == 0
    assert analysis.candidate_weapon_count == 1
    assert analysis.has_threat is False, "Candidate weapon alone must not trigger threat flag"
    assert analysis.severity_level != "critical"
    assert any("require manual verification" in obs for obs in analysis.key_observations)


# ----------------------------------------------------------------------
# 5. Confirmed and rejected human reviews
# ----------------------------------------------------------------------
def test_regression_confirmed_and_rejected_human_reviews():
    """
    Test human investigator reviews:
    - REJECT removes the candidate/verified detection, clears threat, and lowers severity.
    - CONFIRM promotes candidate to verified weapon, sets threat, and updates severity.
    """
    analysis = EvidenceAnalysis(
        source_name="crime_scene.jpg",
        source_type="image",
        counts_by_label={"person": 1, "weapon": 1, "candidate_weapon": 1},
        total_objects=3,
        unique_labels=["candidate_weapon", "person", "weapon"],
        average_confidence=0.75,
        person_count=1,
        verified_weapon_count=1,
        candidate_weapon_count=1,
        weapon_count=1,
        vehicle_count=0,
        bag_count=0,
        severity_score=75,
        severity_level="critical",
        suggested_category="assault",
        key_observations=["1 VERIFIED weapon(s) detected", "1 possible weapon candidate(s)"],
        has_threat=True,
        frame_count=1,
    )
    analysis.detections = [
        Detection("person", "person", 0.90, BoundingBox(10, 10, 50, 100)),
        Detection("gun", "weapon", 0.85, BoundingBox(60, 60, 120, 120), weapon_status="verified"),
        Detection("knife", "candidate_weapon", 0.32, BoundingBox(150, 150, 180, 210), weapon_status="candidate"),
    ]

    # Scenario A: Investigator REJECTS the verified weapon and REJECTS the candidate weapon
    decisions_reject = {
        "weapon@0.85": "REJECT",
        "candidate_weapon@0.32": "REJECT",
    }
    updated_reject = apply_human_review_to_analysis(analysis, decisions_reject)

    assert updated_reject.verified_weapon_count == 0
    assert updated_reject.candidate_weapon_count == 0
    assert updated_reject.has_threat is False, "Threat must be False when all weapons rejected"
    assert updated_reject.severity_score < 25, "Severity score must drop when weapons are rejected"
    assert updated_reject.severity_level in {"low", "moderate"}
    assert updated_reject.suggested_category != "assault"
    assert len(updated_reject.detections) == 1
    assert updated_reject.detections[0].label == "person"
    assert "No verified weapons detected." in updated_reject.key_observations
    assert "No verified weapon detected in this image." in updated_reject.key_observations

    # Scenario B: Investigator CONFIRMS a candidate weapon
    candidate_analysis = EvidenceAnalysis(
        source_name="candidate_scene.jpg",
        source_type="image",
        counts_by_label={"person": 1, "candidate_weapon": 1},
        total_objects=2,
        unique_labels=["candidate_weapon", "person"],
        average_confidence=0.50,
        person_count=1,
        verified_weapon_count=0,
        candidate_weapon_count=1,
        weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=16,
        severity_level="low",
        suggested_category="suspicious_activity",
        key_observations=["1 possible weapon candidate(s)"],
        has_threat=False,
        frame_count=1,
    )
    candidate_analysis.detections = [
        Detection("person", "person", 0.88, BoundingBox(10, 10, 50, 100)),
        Detection("revolver", "candidate_weapon", 0.40, BoundingBox(60, 60, 120, 120), weapon_status="candidate"),
    ]

    decisions_confirm = {"candidate_weapon@0.40": "CONFIRM"}
    updated_confirm = apply_human_review_to_analysis(candidate_analysis, decisions_confirm)

    assert updated_confirm.verified_weapon_count == 1, "Confirmed candidate must promote to verified weapon"
    assert updated_confirm.candidate_weapon_count == 0
    assert updated_confirm.has_threat is True
    assert updated_confirm.severity_score > candidate_analysis.severity_score


# ----------------------------------------------------------------------
# 6. Images with no weapon detections
# ----------------------------------------------------------------------
def test_regression_images_with_no_weapon_detections():
    """
    A negative result on an image must state:
    'No verified weapon detected in this image.'
    and must not claim that no weapon is present in the physical world.
    """
    dets = [
        Detection(class_name="person", label="person", confidence=0.88, bbox=BoundingBox(10, 10, 80, 160), source="general"),
        Detection(class_name="car", label="vehicle", confidence=0.82, bbox=BoundingBox(100, 100, 250, 200), source="general"),
    ]

    states = separate_detection_states(dets)
    assert states["verified_count"] == 0
    assert states["candidate_count"] == 0
    assert states["weapon_status"] == "NO_VERIFIED_WEAPON"

    inp = AnalysisInput(
        source_name="clean_scene.jpg",
        source_type="image",
        counts_by_label={"person": 1, "vehicle": 1},
        average_confidence=0.85,
    )
    res = analyze(inp)
    assert res.verified_weapon_count == 0
    assert res.has_threat is False

    obs_str = " ".join(res.key_observations)
    assert "No verified weapon detected in this image." in res.key_observations, (
        "Negative image result must explicitly state 'No verified weapon detected in this image.'"
    )
    assert "negative results are not proof that weapons are absent" in obs_str


# ----------------------------------------------------------------------
# 7. Configurable threshold ranges and trade-off validation
# ----------------------------------------------------------------------
def test_regression_configurable_threshold_validation_and_tradeoffs():
    """
    Validate threshold ranges, logical ordering, and trade-off documentation.
    """
    # Valid thresholds within [0.01, 0.99]
    res = validate_detection_thresholds(
        confidence=0.25,
        iou=0.45,
        scan_confidence=0.15,
        verify_threshold=0.48,
    )
    assert res["confidence"] == 0.25
    assert res["iou"] == 0.45
    assert res["scan_confidence"] == 0.15
    assert res["verify_threshold"] == 0.48

    # Out of range (< 0.01 or > 0.99) raises ValueError
    with pytest.raises(ValueError, match="Confidence threshold"):
        validate_detection_thresholds(confidence=1.20, iou=0.45)

    with pytest.raises(ValueError, match="IoU threshold"):
        validate_detection_thresholds(confidence=0.25, iou=-0.05)

    # Inverted order: scan_confidence > confidence raises ValueError
    with pytest.raises(ValueError, match="Scan confidence .* cannot exceed"):
        validate_detection_thresholds(confidence=0.20, iou=0.45, scan_confidence=0.35)

    # Inverted order: confidence > verify_threshold raises ValueError
    with pytest.raises(ValueError, match="Primary confidence .* cannot exceed"):
        validate_detection_thresholds(confidence=0.60, iou=0.45, verify_threshold=0.48)

    # Check that trade-off documentation is present in docstring
    doc = validate_detection_thresholds.__doc__
    assert doc is not None
    assert "Trade-Off" in doc or "trade-off" in doc.lower()
    assert "false positives" in doc
    assert "missed detections" in doc

    # Audited model registry covers supported classes and SHA-256 hashes
    assert "yolov8n.pt" in AUDITED_MODEL_REGISTRY
    assert AUDITED_MODEL_REGISTRY["yolov8n.pt"]["weapon_classes"] == ["knife"]
    assert "sha256" in AUDITED_MODEL_REGISTRY["yolov8n.pt"]
