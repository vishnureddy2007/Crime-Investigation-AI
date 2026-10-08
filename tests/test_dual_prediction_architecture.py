"""
Tests for Dual Prediction Architecture & Local Prediction Engine Fallback.
Checks:
1. Local prediction engine generates meaningful evidence-grounded prediction.
2. When Qwen is offline/unavailable, a valid prediction is ALWAYS generated.
3. Exact user failure scenario: 4 verified weapons, 3 persons, critical severity, Qwen offline
   -> MUST NOT produce 'AI Analysis Unavailable', 'Sequential evidence review required', 'Human verification'.
4. Prediction source and confidence are properly populated.
"""

from models.schemas import EvidenceAnalysis
from services.crime_situation import CrimeSituationAnalyzer
from services.prediction_engine import generate_local_prediction


def test_local_prediction_engine_output():
    analysis = EvidenceAnalysis(
        source_name="scene_alpha.mp4",
        source_type="video",
        counts_by_label={"weapon": 2, "person": 3, "car": 1},
        total_objects=6,
        unique_labels=["car", "person", "weapon"],
        average_confidence=0.82,
        person_count=3,
        verified_weapon_count=2,
        candidate_weapon_count=0,
        vehicle_count=1,
        bag_count=0,
        severity_score=88,
        severity_level="critical",
        suggested_category="armed_robbery",
        key_observations=["2 VERIFIED weapons", "3 persons"],
        has_threat=True,
        weapon_count=2,
        frame_count=10,
    )

    pred = generate_local_prediction(analysis)
    assert pred.source == "local_evidence_engine"
    assert "AI Analysis Unavailable" not in pred.likely_activity_pattern
    assert "AI Analysis Unavailable" not in pred.predicted_summary
    assert "Human verification" not in pred.potential_next_activity
    assert pred.confidence_level in {"High", "Medium", "Low"}
    assert len(pred.possible_sequence_of_events) > 0


def test_exact_failure_scenario_reproduction():
    """
    Test exact scenario:
    4 verified weapons, 3 persons, CRITICAL severity, Qwen unavailable.
    """
    analysis = EvidenceAnalysis(
        source_name="high_risk_scene.jpeg",
        source_type="image",
        counts_by_label={"weapon": 4, "person": 3},
        total_objects=7,
        unique_labels=["person", "weapon"],
        average_confidence=0.89,
        person_count=3,
        verified_weapon_count=4,
        candidate_weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=95,
        severity_level="critical",
        suggested_category="assault",
        key_observations=["4 verified weapons", "3 persons"],
        has_threat=True,
        weapon_count=4,
        frame_count=1,
    )

    analyzer = CrimeSituationAnalyzer()
    # Force fallback / offline test
    res = analyzer._fallback_analysis(analysis)

    # Must NOT produce the old bad placeholders
    assert res.likely_activity_pattern != "AI Analysis Unavailable"
    assert "AI Analysis Unavailable" not in res.likely_activity_pattern
    assert res.possible_sequence_of_events != "Sequential evidence review required."
    assert "Human verification of detected objects." not in res.potential_next_activity
    assert res.confidence_level != "Deterministic Fallback"

    # MUST produce evidence-grounded summary
    assert "weapon" in res.likely_activity_pattern.lower() or "armed" in res.likely_activity_pattern.lower()
    assert res.source == "local_evidence_engine"
    assert "4" in str(res.supporting_evidence) or "weapon" in str(res.supporting_evidence).lower()
