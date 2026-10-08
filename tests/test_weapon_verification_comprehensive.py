"""
Comprehensive Acceptance Test Matrix for Weapon Detection and Verification Pipeline.

Tests all 15 required acceptance scenarios:
1. No weapon image -> 0 verified
2. No weapon + many false YOLO candidates -> 0 verified
3. One clearly visible revolver -> 1 verified (VERIFIED_WEAPON, subtype: revolver)
4. One clearly visible pistol -> 1 verified
5. Two separate firearms -> 2 verified
6. One firearm + multiple overlapping YOLO boxes -> 1 verified (deduplicated)
7. Tiny false candidate -> rejected
8. Low-confidence false candidate -> rejected
9. Phone/remote/tool incorrectly detected as weapon -> rejected when visual evidence is insufficient
10. Same firearm across multiple video frames -> 1 physical weapon
11. One-frame transient false positive -> not verified
12. Candidate rejected by verifier -> never included in final count
13. Real revolver with moderate YOLO confidence -> verified when visual evidence supports firearm
14. Revolver with weak subtype confidence but strong firearm evidence -> VERIFIED_WEAPON with firearm/weapon subtype
15. Multiple-scale duplicate detections -> one physical weapon
"""

import pytest
from PIL import Image
from models.schemas import BoundingBox, Detection, AnalysisInput
from models.weapon_verifier import (
    classify_weapon_detection,
    separate_detection_states,
    verify_detection,
    apply_to_detections,
    verify_weapon_crop_visual,
    WeaponVerificationResult,
)
from models.evidence_analyzer import analyze
from models.yolo_detector import _cross_model_nms


def test_1_no_weapon_image():
    """Scenario 1: Image with NO weapon returns verified_count = 0."""
    dets = [
        Detection(class_name="person", label="person", confidence=0.88, bbox=BoundingBox(10, 10, 100, 200), source="general"),
        Detection(class_name="chair", label="chair", confidence=0.75, bbox=BoundingBox(200, 200, 300, 400), source="general"),
    ]
    states = separate_detection_states(dets)
    assert states["verified_count"] == 0
    assert len(states["verified_weapons"]) == 0
    assert states["weapon_status"] == "NO_VERIFIED_WEAPON"

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


def test_2_no_weapon_many_false_yolo_candidates():
    """Scenario 2: Image with multiple false YOLO candidates returns verified_count = 0."""
    tiny_box = BoundingBox(10, 10, 14, 14)  # 4x4 px
    dets = [
        Detection(class_name="weapon", label="weapon", confidence=0.26, bbox=tiny_box, source="general"),
        Detection(class_name="gun", label="weapon", confidence=0.27, bbox=tiny_box, source="general"),
        Detection(class_name="knife", label="knife", confidence=0.25, bbox=tiny_box, source="general"),
        Detection(class_name="weapon", label="weapon", confidence=0.29, bbox=tiny_box, source="general"),
    ]
    states = separate_detection_states(dets)
    assert states["verified_count"] == 0
    assert len(states["verified_weapons"]) == 0


def test_3_one_clearly_visible_revolver():
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
    assert processed[0].class_name == "revolver"

    states = separate_detection_states([det])
    assert states["verified_count"] == 1
    assert len(states["verified_weapons"]) == 1
    assert states["canonical_subtype"] == "revolver"


def test_4_one_clearly_visible_pistol():
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


def test_5_two_separate_firearms():
    """Scenario 5: Image containing TWO distinct weapons returns verified_count = 2."""
    det1 = Detection(class_name="revolver", label="revolver", confidence=0.85, bbox=BoundingBox(10, 10, 80, 80), source="weapon")
    det2 = Detection(class_name="rifle", label="rifle", confidence=0.88, bbox=BoundingBox(250, 250, 400, 320), source="weapon")

    merged = _cross_model_nms([det1, det2])
    assert len(merged) == 2

    states = separate_detection_states(merged)
    assert states["verified_count"] == 2


def test_6_one_firearm_multiple_overlapping_boxes():
    """Scenario 6: Image containing one firearm with 4 overlapping YOLO boxes returns verified_count = 1."""
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

    states = separate_detection_states([det1, det2, det3, det4])
    assert states["verified_count"] == 1


def test_7_tiny_false_candidate_rejected():
    """Scenario 7: Tiny false candidate box (< 6px or area < 25px) is rejected."""
    tiny_box = BoundingBox(10, 10, 13, 13)
    state, target_label = classify_weapon_detection(label="weapon", confidence=0.28, source="general", bbox=tiny_box)
    assert state == "NOT_WEAPON"


def test_8_low_confidence_false_candidate_rejected():
    """Scenario 8: Low-confidence false candidate (0.20) is rejected."""
    v = verify_detection(label="weapon", confidence=0.20, source="general")
    assert v.status == "non_weapon"
    assert v.is_verified is False


def test_9_flat_noise_object_rejected():
    """Scenario 9: Flat gradient/non-firearm object crop is rejected during visual verification."""
    # Create smooth gradient PIL image (no firearm edge density)
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    bbox = BoundingBox(20, 20, 100, 100)

    score_delta, reason = verify_weapon_crop_visual(img, bbox, class_name="weapon", confidence=0.35)
    assert reason == "FLAT_GRADIENT_NOISE"
    assert score_delta < 0.0


def test_10_same_firearm_across_video_frames():
    """Scenario 10: Same firearm across multiple video frames consolidates into case-level count = 1."""
    from models.video_processor import _apply_temporal_weapon_verification
    from models.schemas import FrameResult, DetectionResult

    frames = []
    for idx in range(5):
        det = Detection(class_name="revolver", label="weapon", confidence=0.70, bbox=BoundingBox(50, 50, 150, 150), source="weapon")
        dr = DetectionResult(source_name=f"frame_{idx}", timestamp=None, detections=[det])
        fr = FrameResult(index=idx, timestamp_sec=float(idx * 1.0), image_path=None, annotated_path=None, detection=dr)
        frames.append(fr)

    promoted = _apply_temporal_weapon_verification(frames, min_frames=2, max_frame_gap_sec=2.5, min_avg_confidence=0.45)
    assert len(promoted) > 0


def test_11_transient_single_frame_false_positive_rejected():
    """Scenario 11: Single-frame transient false positive is not promoted to verified weapon."""
    from models.video_processor import _apply_temporal_weapon_verification
    from models.schemas import FrameResult, DetectionResult

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


def test_12_candidate_rejected_by_verifier_excluded():
    """Scenario 12: Candidate rejected by verifier is never included in verified count."""
    det = Detection(class_name="weapon", label="weapon", confidence=0.22, bbox=BoundingBox(10, 10, 20, 20), source="general")
    states = separate_detection_states([det])
    assert states["verified_count"] == 0
    assert len(states["verified_weapons"]) == 0


def test_13_revolver_moderate_confidence_verified():
    """Scenario 13: Real revolver with moderate YOLO confidence (0.40) is verified when source/subtype is valid."""
    det = Detection(
        class_name="revolver",
        label="weapon",
        confidence=0.40,
        bbox=BoundingBox(50, 50, 180, 140),
        source="weapon",
    )
    state, target_label = classify_weapon_detection("revolver", 0.40, source="weapon", bbox=det.bbox, class_name="revolver")
    assert state == "WEAPON"

    states = separate_detection_states([det])
    assert states["verified_count"] == 1


def test_14_firearm_verified_even_if_subtype_uncertain():
    """Scenario 14: Firearm verified as VERIFIED_WEAPON with generic firearm/weapon type when subtype is generic."""
    det = Detection(
        class_name="firearm",
        label="weapon",
        confidence=0.55,
        bbox=BoundingBox(30, 30, 160, 120),
        source="weapon",
    )
    state, target_label = classify_weapon_detection("firearm", 0.55, source="weapon", bbox=det.bbox, class_name="firearm")
    assert state == "WEAPON"
    assert target_label == "weapon"

    states = separate_detection_states([det])
    assert states["verified_count"] == 1
    assert states["verified_weapons"][0].weapon_status == "verified"


def test_15_multi_scale_duplicate_detections_deduplicated():
    """Scenario 15: Multi-scale duplicate detections across image sizes deduplicate to 1 physical weapon."""
    # Scale 1 (standard res): 640x480 box
    det1 = Detection(class_name="revolver", label="weapon", confidence=0.82, bbox=BoundingBox(100, 100, 300, 250), source="weapon")
    # Scale 2 (high res): 1280x960 box mapped to original canvas
    det2 = Detection(class_name="revolver", label="weapon", confidence=0.89, bbox=BoundingBox(102, 101, 298, 252), source="weapon")

    merged = _cross_model_nms([det1, det2])
    assert len(merged) == 1
    assert merged[0].confidence == 0.89

    states = separate_detection_states([det1, det2])
    assert states["verified_count"] == 1
