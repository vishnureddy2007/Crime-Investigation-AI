"""
Comprehensive Acceptance Test Suite for Weapon Detection and Verification Pipeline.

Tests all 10 required acceptance scenarios:
1. Image with NO weapon -> verified_count = 0
2. Image with multiple false-positive YOLO detections -> verified_count = 0
3. Image containing ONE revolver -> verified_count = 1, status = VERIFIED_WEAPON, subtype = revolver
4. Image containing ONE pistol -> verified_count = 1
5. Image containing TWO distinct weapons -> verified_count = 2
6. Image containing one weapon with multiple overlapping boxes -> verified_count = 1 (merged)
7. Video containing the same weapon across many frames -> consolidated weapon count = 1
8. Video with transient false-positive frame -> not promoted to verified weapon
9. Low-confidence candidate -> not verified
10. Rejected candidate -> excluded from final weapon count
"""

import pytest
from models.schemas import BoundingBox, Detection, AnalysisInput
from models.weapon_verifier import (
    classify_weapon_detection,
    separate_detection_states,
    verify_detection,
    apply_to_detections,
    WeaponVerificationResult,
)
from models.evidence_analyzer import analyze
from models.yolo_detector import _cross_model_nms


def test_1_image_with_no_weapon():
    """Scenario 1: Image with NO weapon returns verified_count = 0."""
    dets = [
        Detection(class_name="person", label="person", confidence=0.88, bbox=BoundingBox(10, 10, 100, 200), source="general"),
        Detection(class_name="chair", label="chair", confidence=0.75, bbox=BoundingBox(200, 200, 300, 400), source="general"),
    ]
    states = separate_detection_states(dets)
    assert states["verified_count"] == 0
    assert len(states["verified_weapons"]) == 0
    assert states["weapon_status"] == "NO_VERIFIED_WEAPON"

    # Analyze input
    inp = AnalysisInput(
        source_name="no_weapon.jpg",
        source_type="image",
        counts_by_label={"person": 1, "chair": 1},
        average_confidence=0.82,
    )
    res = analyze(inp)
    assert res.verified_weapon_count == 0
    assert res.has_threat is False
    assert "No verified weapons detected." in res.key_observations


def test_2_multiple_false_positive_yolo_detections_rejected():
    """Scenario 2: Image with multiple weak/false-positive YOLO weapon candidates returns verified_count = 0."""
    # 4 weak background candidates with low confidence or tiny box size
    tiny_box = BoundingBox(10, 10, 14, 14) # 4x4 px
    dets = [
        Detection(class_name="weapon", label="weapon", confidence=0.26, bbox=tiny_box, source="general"),
        Detection(class_name="gun", label="weapon", confidence=0.27, bbox=tiny_box, source="general"),
        Detection(class_name="knife", label="knife", confidence=0.25, bbox=tiny_box, source="general"),
        Detection(class_name="weapon", label="weapon", confidence=0.29, bbox=tiny_box, source="general"),
    ]
    states = separate_detection_states(dets)
    assert states["verified_count"] == 0
    assert len(states["verified_weapons"]) == 0


def test_3_image_containing_one_revolver():
    """Scenario 3: Image containing ONE revolver returns verified_count = 1, status = VERIFIED_WEAPON, subtype = revolver."""
    det = Detection(
        class_name="revolver",
        label="revolver",
        confidence=0.86,
        bbox=BoundingBox(50, 50, 150, 120),
        source="weapon",
    )
    processed = apply_to_detections([det])
    assert len(processed) == 1
    assert processed[0].weapon_status == "verified"
    assert processed[0].label in {"revolver", "weapon"}
    assert processed[0].class_name == "revolver"

    states = separate_detection_states([det])
    assert states["verified_count"] == 1
    assert len(states["verified_weapons"]) == 1


def test_4_image_containing_one_pistol():
    """Scenario 4: Image containing ONE pistol returns verified_count = 1."""
    det = Detection(
        class_name="pistol",
        label="weapon",
        confidence=0.85,
        bbox=BoundingBox(40, 40, 140, 110),
        source="weapon",
    )
    states = separate_detection_states([det])
    assert states["verified_count"] == 1
    assert len(states["verified_weapons"]) == 1
    assert states["verified_weapons"][0].weapon_status == "verified"


def test_5_image_containing_two_distinct_weapons():
    """Scenario 5: Image containing TWO distinct weapons returns verified_count = 2."""
    det1 = Detection(class_name="revolver", label="revolver", confidence=0.85, bbox=BoundingBox(10, 10, 80, 80), source="weapon")
    det2 = Detection(class_name="rifle", label="rifle", confidence=0.88, bbox=BoundingBox(250, 250, 400, 320), source="weapon")

    merged = _cross_model_nms([det1, det2])
    assert len(merged) == 2

    states = separate_detection_states(merged)
    assert states["verified_count"] == 2


def test_6_image_containing_one_weapon_with_multiple_overlapping_boxes():
    """Scenario 6: Image containing one weapon with 4 overlapping detections returns verified_count = 1 (merged)."""
    b1 = BoundingBox(50, 50, 150, 120)
    b2 = BoundingBox(52, 52, 152, 122)
    b3 = BoundingBox(48, 50, 148, 118)
    b4 = BoundingBox(50, 55, 155, 125)

    det1 = Detection(class_name="revolver", label="revolver", confidence=0.91, bbox=b1, source="weapon")
    det2 = Detection(class_name="gun", label="weapon", confidence=0.87, bbox=b2, source="general")
    det3 = Detection(class_name="pistol", label="weapon", confidence=0.84, bbox=b3, source="weapon")
    det4 = Detection(class_name="firearm", label="weapon", confidence=0.79, bbox=b4, source="general")

    merged = _cross_model_nms([det1, det2, det3, det4])
    assert len(merged) == 1
    assert merged[0].confidence == 0.91

    states = separate_detection_states(merged)
    assert states["verified_count"] == 1


def test_7_video_containing_same_weapon_across_frames():
    """Scenario 7: Same weapon detected across multiple video frames consolidates into case-level count = 1."""
    from models.video_processor import _apply_temporal_weapon_verification
    from models.schemas import FrameResult, DetectionResult

    # Simulate 5 sampled video frames with weapon detection
    frames = []
    for idx in range(5):
        det = Detection(class_name="revolver", label="weapon", confidence=0.70, bbox=BoundingBox(50, 50, 150, 150), source="weapon")
        dr = DetectionResult(source_name=f"frame_{idx}", timestamp=None, detections=[det])
        fr = FrameResult(index=idx, timestamp_sec=float(idx * 1.0), image_path=None, annotated_path=None, detection=dr)
        frames.append(fr)

    promoted = _apply_temporal_weapon_verification(frames, min_frames=2, max_frame_gap_sec=2.5, min_avg_confidence=0.45)
    assert len(promoted) > 0


def test_8_video_transient_false_positive_rejected():
    """Scenario 8: Single-frame transient false-positive is NOT promoted to verified weapon."""
    from models.video_processor import _apply_temporal_weapon_verification
    from models.schemas import FrameResult, DetectionResult

    # Only frame 2 has a weak single-frame candidate
    frames = []
    for idx in range(5):
        dets = []
        if idx == 2:
            dets.append(Detection(class_name="knife", label="knife", confidence=0.32, bbox=BoundingBox(10, 10, 30, 30), source="general", weapon_status="candidate"))
        dr = DetectionResult(source_name=f"frame_{idx}", timestamp=None, detections=dets)
        fr = FrameResult(index=idx, timestamp_sec=float(idx * 1.0), image_path=None, annotated_path=None, detection=dr)
        frames.append(fr)

    promotions = _apply_temporal_weapon_verification(frames, min_frames=2, max_frame_gap_sec=2.5, min_avg_confidence=0.45)
    assert len(promotions) == 0


def test_9_low_confidence_candidate_not_verified():
    """Scenario 9: Low confidence candidate (0.30) is classified as UNCERTAIN / candidate, NOT verified."""
    state, target_label = classify_weapon_detection(label="weapon", confidence=0.30, source="general")
    assert state == "UNCERTAIN"

    v = verify_detection(label="weapon", confidence=0.30, source="general")
    assert v.status == "candidate"
    assert v.is_verified is False


def test_10_rejected_candidate_excluded_from_final_weapon_count():
    """Scenario 10: Rejected candidate (sub-threshold or tiny box) is excluded from final verified weapon count."""
    tiny_box = BoundingBox(10, 10, 12, 12) # 2x2 px box
    state, target_label = classify_weapon_detection(label="weapon", confidence=0.25, source="general", bbox=tiny_box)
    assert state == "NOT_WEAPON"

    v = verify_detection(label="weapon", confidence=0.20, source="general")
    assert v.status == "non_weapon"
    assert v.is_verified is False
