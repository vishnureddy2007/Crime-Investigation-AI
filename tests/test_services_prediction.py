"""Unit tests for `services/crime_prediction.py`."""
from __future__ import annotations

from datetime import datetime

from models.schemas import EvidenceAnalysis
from services.crime_prediction import (
    CATEGORIES,
    RiskAssessment,
    predict_risk,
)


def _a(**kw) -> EvidenceAnalysis:
    base = dict(
        source_name="c.png",
        source_type="image",
        counts_by_label={},
        total_objects=0,
        unique_labels=[],
        average_confidence=0.5,
        person_count=0,
        verified_weapon_count=0,
        candidate_weapon_count=0,
        weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=10,
        severity_level="low",
        suggested_category="unknown",
        key_observations=[],
        has_threat=False,
        frame_count=1,
        timestamp=datetime(2026, 1, 1),
    )
    base.update(kw)
    return EvidenceAnalysis(**base)


class TestPredictRisk:
    def test_none_analysis_returns_zero_distribution(self) -> None:
        r = predict_risk(None)  # type: ignore[arg-type]
        assert r.source_name == ""
        assert r.scores == {c: 0.0 for c in CATEGORIES}

    def test_empty_evidence_pushes_suspicious(self) -> None:
        r = predict_risk(_a())
        assert r.scores["suspicious_activity"] >= 0.5
        assert r.top_category == "suspicious_activity"

    def test_weapon_raises_assault_and_robbery(self) -> None:
        r = predict_risk(_a(weapon_count=1, has_threat=True))
        assert r.scores["assault"] > 0.10
        assert r.scores["robbery"] > 0.10

    def test_bags_raise_theft(self) -> None:
        r = predict_risk(_a(bag_count=1))
        assert r.scores["theft"] > 0.10

    def test_vehicle_raises_vehicle_incident(self) -> None:
        r = predict_risk(_a(vehicle_count=1))
        assert r.scores["vehicle_incident"] > 0.10

    def test_distribution_sums_to_one(self) -> None:
        for analysis in [
            _a(),
            _a(weapon_count=2, person_count=3, vehicle_count=1, bag_count=1,
               average_confidence=0.9, severity_score=90),
            _a(weapon_count=1, person_count=5, average_confidence=0.95,
               severity_score=70),
        ]:
            r = predict_risk(analysis)
            assert abs(sum(r.scores.values()) - 1.0) < 1e-6

    def test_deterministic(self) -> None:
        a = _a(weapon_count=1, person_count=2)
        r1 = predict_risk(a)
        r2 = predict_risk(a)
        assert r1.as_dict() == r2.as_dict()

    def test_top_category_property(self) -> None:
        r = predict_risk(_a(weapon_count=1))
        assert r.top_category in r.scores
        assert r.scores[r.top_category] == r.top_score


class TestRiskAssessment:
    def test_as_dict(self) -> None:
        r = RiskAssessment(source_name="x", scores={"a": 0.5, "b": 0.5})
        d = r.as_dict()
        assert d["source_name"] == "x"
        assert d["scores"] == {"a": 0.5, "b": 0.5}