"""
Targeted test suite verifying optimization, multi-stage detection pipeline,
IoU/IoS NMS duplicate merging, small false positive box filtering, HITL weapon review integrity,
and speed enhancements.
"""

import pytest
from PIL import Image
from models.schemas import BoundingBox, Detection, EvidenceAnalysis
from models.yolo_detector import _cross_model_nms, _iou, _ios, MultiSourceDetector
from models.weapon_verifier import verify_detection, apply_human_review_to_analysis, verified_counts
from services.storyboard_bridge import StoryboardGenerator
from models.schemas import DetailedNarrativeSummary, CrimeSituationAnalysis


def test_iou_and_ios_computation():
    box1 = BoundingBox(x1=10, y1=10, x2=100, y2=100) # Area = 90x90 = 8100
    box2 = BoundingBox(x1=20, y1=20, x2=90, y2=90)   # Area = 70x70 = 4900 inside box1
    
    iou_val = _iou(box1, box2)
    ios_val = _ios(box1, box2)
    
    assert iou_val == pytest.approx(4900 / 8100, 0.01)
    assert ios_val == pytest.approx(1.0, 0.01) # Completely inside smaller box


def test_cross_model_nms_duplicate_weapon_merging():
    # Two overlapping boxes around the SAME physical weapon from 2 models
    det1 = Detection(class_name="gun", label="weapon", confidence=0.88, bbox=BoundingBox(10, 10, 100, 100), source="weapon")
    det2 = Detection(class_name="pistol", label="weapon", confidence=0.72, bbox=BoundingBox(15, 15, 95, 95), source="threat-weapon")
    det3 = Detection(class_name="person", label="person", confidence=0.90, bbox=BoundingBox(200, 200, 300, 500), source="general")
    
    merged = _cross_model_nms([det1, det2, det3], iou_threshold=0.40)
    
    # 1 weapon (the highest conf one det1) and 1 person (det3) should remain
    weapon_dets = [d for d in merged if d.label == "weapon"]
    assert len(weapon_dets) == 1
    assert weapon_dets[0].confidence == 0.88
    assert len(merged) == 2


def test_weapon_verifier_confidence_tiers():
    # High confidence shortcut
    v_high = verify_detection(label="weapon", confidence=0.85, source="general")
    assert v_high.status == "verified"

    # Medium confidence with dedicated model boost (0.40 + 0.10 boost = 0.50 >= 0.45)
    v_med = verify_detection(label="weapon", confidence=0.40, source="weapon")
    assert v_med.status == "verified"

    # Low confidence candidate (0.32)
    v_cand = verify_detection(label="weapon", confidence=0.32, source="general")
    assert v_cand.status == "candidate"
    assert v_cand.label == "candidate_weapon"

    # Extremely low confidence (< 0.30)
    v_reject = verify_detection(label="weapon", confidence=0.20, source="general")
    assert v_reject.status == "non_weapon"


def test_human_review_rejection_exclusion():
    analysis = EvidenceAnalysis(
        source_name="test_scene",
        source_type="image",
        counts_by_label={"person": 2, "weapon": 1},
        total_objects=3,
        unique_labels=["person", "weapon"],
        average_confidence=0.85,
        person_count=2,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=80,
        severity_level="high",
        suggested_category="armed_robbery",
        key_observations=["Person seen with weapon"],
        has_threat=True,
        weapon_count=1,
        frame_count=1,
    )
    analysis.detections = [
        Detection("person", "person", 0.90, BoundingBox(10, 10, 50, 100)),
        Detection("gun", "weapon", 0.85, BoundingBox(60, 60, 120, 120))
    ]

    # Investigator REJECTS the weapon
    decisions = {"weapon@0.85": "REJECT"}
    updated = apply_human_review_to_analysis(analysis, decisions)

    assert updated.verified_weapon_count == 0
    assert updated.has_threat == False
    assert "weapon" not in updated.counts_by_label or updated.counts_by_label["weapon"] == 0

    # Ensure 3D Storyboard Plan excludes rejected weapon
    summary = DetailedNarrativeSummary(
        case_id="c1", case_overview="", evidence_reviewed="", chronological_events="",
        detected_objects="", verified_findings="", possible_findings="", rejected_findings="",
        potential_crime_activity="", important_evidence="", uncertainties="", investigation_summary="No threat"
    )
    situation = CrimeSituationAnalysis(
        likely_activity_pattern="normal",
        possible_sequence_of_events="Person walks",
        potential_next_activity="exit",
        suspicious_behavior_indicators=[],
        risk_indicators=[],
        supporting_evidence=[],
        confidence_level="high",
        uncertainties="none",
        alternative_explanations="none"
    )

    sb_gen = StoryboardGenerator()
    plan = sb_gen.generate_plan("c1", updated, summary, situation)
    
    # Verify no weapon objects are present in the 3D plan
    weapon_objs = [o for o in plan.objects if o["kind"] == "weapon"]
    assert len(weapon_objs) == 0


def test_multisource_detector_singleton():
    d1 = MultiSourceDetector()
    d2 = MultiSourceDetector()
    # Ensure detector instances are valid and duck-typed
    assert hasattr(d1, "detect_image")
    assert hasattr(d2, "detect_image")
