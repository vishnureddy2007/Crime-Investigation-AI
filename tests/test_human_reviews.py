"""Tests for the Phase 51 human-review override path.

Covers:
1. Repository persistence (save / list / lookup / invalid decision).
2. `apply_human_review_to_analysis` strips REJECTED detections from
   an `EvidenceAnalysis`, updates counts, and clears `has_threat`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from database.db import init_db
from database.repository import (
    human_review_for,
    list_human_reviews,
    save_case,
    save_human_review,
)
from models.weapon_verifier import apply_human_review_to_analysis


# ----------------------------------------------------------------------
# Lightweight duck-typed "analysis" for the override test
# ----------------------------------------------------------------------
@dataclass
class _StubDetection:
    label: str
    confidence: float
    class_name: str = "weapon"
    bbox: object = None
    source: str = "weapon"
    weapon_status: str = "verified"


@dataclass
class _StubAnalysis:
    detections: list = field(default_factory=list)
    counts_by_label: dict = field(default_factory=dict)
    has_threat: bool = False
    severity_score: int = 50

    def __post_init__(self) -> None:
        if not self.counts_by_label and self.detections:
            counts: dict[str, int] = {}
            for d in self.detections:
                counts[d.label] = counts.get(d.label, 0) + 1
            self.counts_by_label = counts


# ----------------------------------------------------------------------
# DB persistence tests
# ----------------------------------------------------------------------
def test_save_human_review_persists(db_path) -> None:
    init_db(db_path)
    cid = save_case(db_path, source_name="clip.mp4", source_type="video")

    rid = save_human_review(
        db_path,
        case_id=cid,
        detection_key="weapon@0.62:frame_001",
        label="weapon",
        confidence=0.62,
        decision="REJECT",
        reviewer="Det. Anand",
        note="Looked like a remote control.",
    )
    assert rid >= 1
    rows = list_human_reviews(db_path, case_id=cid)
    assert len(rows) == 1
    r = rows[0]
    assert r["decision"] == "REJECT"
    assert r["reviewer"] == "Det. Anand"
    assert r["original_status"] == "candidate"


def test_human_review_for_lookup(db_path) -> None:
    init_db(db_path)
    cid = save_case(db_path, source_name="clip2.mp4", source_type="video")
    save_human_review(
        db_path,
        case_id=cid,
        detection_key="weapon@0.62:frame_001",
        label="weapon",
        confidence=0.62,
        decision="CONFIRM",
        reviewer="Officer",
    )
    got = human_review_for(db_path, case_id=cid, detection_key="weapon@0.62:frame_001")
    assert got is not None
    assert got["decision"] == "CONFIRM"
    missing = human_review_for(db_path, case_id=cid, detection_key="nonexistent@0.99")
    assert missing is None


def test_save_human_review_rejects_bad_decision(db_path) -> None:
    init_db(db_path)
    cid = save_case(db_path, source_name="x.mp4", source_type="video")
    with pytest.raises(ValueError):
        save_human_review(
            db_path,
            case_id=cid,
            detection_key="weapon@0.5",
            label="weapon",
            confidence=0.5,
            decision="MAYBE",
        )


# ----------------------------------------------------------------------
# apply_human_review_to_analysis behaviour
# ----------------------------------------------------------------------
def test_rejected_weapon_is_removed_from_analysis() -> None:
    """REJECT strips every matching weapon-class detection from the analysis."""
    analysis = _StubAnalysis(
        detections=[
            _StubDetection(label="weapon",  confidence=0.62),
            _StubDetection(label="weapon",  confidence=0.71),
            _StubDetection(label="person",  confidence=0.88),
            _StubDetection(label="vehicle", confidence=0.80),
        ],
        counts_by_label={"weapon": 2, "person": 1, "vehicle": 1},
        has_threat=True,
    )
    decisions = {
        "weapon@0.62": "REJECT",
        "weapon@0.71": "REJECT",
    }
    out = apply_human_review_to_analysis(analysis, decisions)

    assert all(d.label != "weapon" for d in out.detections), (
        "REJECT must remove all matching weapon-class detections"
    )
    assert out.counts_by_label == {"person": 1, "vehicle": 1}
    assert out.has_threat is False, (
        "Removing every verified weapon must clear the threat flag"
    )


def test_rejected_does_not_touch_non_weapon_classes() -> None:
    """REJECT only suppresses weapon-class detections — persons stay."""
    analysis = _StubAnalysis(
        detections=[
            _StubDetection(label="weapon",  confidence=0.62),
            _StubDetection(label="person",  confidence=0.88),
            _StubDetection(label="bag",     confidence=0.55),
        ],
        counts_by_label={"weapon": 1, "person": 1, "bag": 1},
        has_threat=True,
    )
    out = apply_human_review_to_analysis(
        analysis, {"weapon@0.62": "REJECT"}
    )
    labels = sorted(d.label for d in out.detections)
    assert labels == ["bag", "person"]


def test_confirm_and_uncertain_are_no_ops() -> None:
    """Only REJECT removes a detection; CONFIRM / UNCERTAIN leave it alone."""
    analysis = _StubAnalysis(
        detections=[
            _StubDetection(label="weapon", confidence=0.62),
            _StubDetection(label="weapon", confidence=0.71),
        ],
        counts_by_label={"weapon": 2},
        has_threat=True,
    )
    no_op = apply_human_review_to_analysis(
        analysis,
        {"weapon@0.62": "CONFIRM", "weapon@0.71": "UNCERTAIN"},
    )
    assert len(no_op.detections) == 2
    assert no_op.has_threat is True


def test_empty_decisions_is_a_no_op() -> None:
    """No decisions → analysis returned unchanged (same object identity)."""
    analysis = _StubAnalysis(
        detections=[_StubDetection(label="weapon", confidence=0.62)],
        counts_by_label={"weapon": 1},
        has_threat=True,
    )
    out = apply_human_review_to_analysis(analysis, {})
    assert out is analysis


def test_missing_detection_key_is_ignored() -> None:
    """A decision for a key that doesn't match any detection is silently ignored."""
    analysis = _StubAnalysis(
        detections=[_StubDetection(label="weapon", confidence=0.62)],
        counts_by_label={"weapon": 1},
        has_threat=True,
    )
    out = apply_human_review_to_analysis(analysis, {"knife@0.80": "REJECT"})
    assert out is analysis or len(out.detections) == 1
