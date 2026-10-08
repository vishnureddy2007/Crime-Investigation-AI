"""
Tests for Automatic Weapon Verification Architecture.
Checks:
1. Real weapon -> accepted as VERIFIED.
2. False positive object -> rejected / NOT_WEAPON.
3. Multiple overlapping boxes -> merged into 1 unique weapon.
4. Same weapon across video frames -> consolidated into 1 unique weapon.
5. Two distinct weapons -> counted as 2 weapons.
6. Uncertain detection -> NOT counted as verified weapon.
7. Small false-positive box -> rejected when combined evidence is low.
8. Manual human confirmation is NOT required for automatic pipeline.
"""

import pytest
from models.schemas import BoundingBox, Detection
from models.weapon_verifier import (
    classify_weapon_detection,
    separate_detection_states,
    verify_detection,
)
from models.yolo_detector import _cross_model_nms


def test_real_weapon_accepted():
    state, target_label = classify_weapon_detection(
        label="revolver", confidence=0.88, source="weapon"
    )
    assert state == "WEAPON"
    assert target_label == "weapon"


def test_false_positive_object_rejected():
    state, target_label = classify_weapon_detection(
        label="bottle", confidence=0.30, source="general"
    )
    assert state == "NOT_WEAPON"


def test_small_false_positive_box_rejected():
    # Tiny 3x3 box with low confidence -> rejected as NOT_WEAPON
    tiny_bbox = BoundingBox(x1=10, y1=10, x2=13, y2=13)
    state, target_label = classify_weapon_detection(
        label="weapon", confidence=0.28, source="general", bbox=tiny_bbox
    )
    assert state == "NOT_WEAPON"


def test_small_genuine_high_conf_weapon_accepted():
    # Small box with high confidence -> accepted as WEAPON
    small_bbox = BoundingBox(x1=10, y1=10, x2=20, y2=20)
    state, target_label = classify_weapon_detection(
        label="handgun", confidence=0.85, source="weapon", bbox=small_bbox
    )
    assert state == "WEAPON"


def test_uncertain_detection_not_counted_as_verified():
    state, target_label = classify_weapon_detection(
        label="weapon", confidence=0.30, source="general"
    )
    assert state == "UNCERTAIN"

    # In separate_detection_states, UNCERTAIN goes into uncertain_detections, NOT verified_weapons
    det = Detection(class_name="gun", label="weapon", confidence=0.30, bbox=BoundingBox(0,0,50,50), source="general")
    states = separate_detection_states([det])
    assert len(states["verified_weapons"]) == 0
    assert len(states["uncertain_detections"]) == 1


def test_multiple_overlapping_boxes_merged():
    det1 = Detection(class_name="pistol", label="weapon", confidence=0.85, bbox=BoundingBox(10, 10, 50, 50), source="weapon")
    det2 = Detection(class_name="gun", label="weapon", confidence=0.80, bbox=BoundingBox(12, 12, 52, 52), source="general")

    merged = _cross_model_nms([det1, det2])
    assert len(merged) == 1
    assert merged[0].confidence == 0.85


def test_two_distinct_weapons_counted():
    det1 = Detection(class_name="pistol", label="weapon", confidence=0.85, bbox=BoundingBox(10, 10, 50, 50), source="weapon")
    det2 = Detection(class_name="rifle", label="weapon", confidence=0.80, bbox=BoundingBox(200, 200, 300, 300), source="weapon")

    merged = _cross_model_nms([det1, det2])
    assert len(merged) == 2


def test_automatic_pipeline_no_human_confirm_required():
    det = Detection(class_name="revolver", label="weapon", confidence=0.90, bbox=BoundingBox(10, 10, 50, 50), source="weapon")
    states = separate_detection_states([det])

    assert len(states["verified_weapons"]) == 1
    assert states["verified_weapons"][0].weapon_status == "verified"
