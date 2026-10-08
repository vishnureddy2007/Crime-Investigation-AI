"""
Unit tests confirming that weapons like revolver are mapped into the weapon category.
"""

from config.settings import (
    SUPPORTED_DETECTION_CLASSES,
    WEAPON_CANDIDATE_LABELS,
    YOLO_CLASS_MAPPING,
)
from models.evidence_analyzer import analyze
from models.schemas import AnalysisInput, Detection
from models.weapon_verifier import (
    WEAPON_LABELS,
    apply_to_detections,
    verify_detection,
)
from models.yolo_detector import _WEAPON_LABEL_SYNONYMS


def test_revolver_in_configs():
    assert "revolver" in YOLO_CLASS_MAPPING
    assert YOLO_CLASS_MAPPING["revolver"] == "weapon"
    assert "revolver" in SUPPORTED_DETECTION_CLASSES
    assert "revolver" in WEAPON_CANDIDATE_LABELS
    assert "revolver" in _WEAPON_LABEL_SYNONYMS
    assert "revolver" in WEAPON_LABELS


def test_revolver_verification_and_category():
    v = verify_detection(label="revolver", confidence=0.85, source="weapon")
    assert v.status == "verified"
    assert v.label == "weapon"

    det = Detection(
        class_name="revolver",
        label="weapon",
        confidence=0.85,
        bbox=None,
        source="weapon",
    )
    processed = apply_to_detections([det])
    assert len(processed) == 1
    assert processed[0].label == "weapon"
    assert processed[0].weapon_status == "verified"


def test_revolver_in_evidence_analyzer():
    inp = AnalysisInput(
        source_name="test_revolver.jpeg",
        source_type="image",
        counts_by_label={"weapon": 1, "person": 1},
        average_confidence=0.85,
    )
    res = analyze(inp)
    assert res.verified_weapon_count == 1
    assert res.weapon_count == 1
    assert res.has_threat is True
