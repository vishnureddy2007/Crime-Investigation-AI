"""
Tests for the evidence analyzer and EvidenceAnalysis schema.
"""

from __future__ import annotations

import datetime

import pytest

from models.evidence_analyzer import analyze
from models.schemas import (
    AnalysisInput,
    BoundingBox,
    Detection,
    DetectionResult,
    EvidenceAnalysis,
    FrameResult,
    VideoAnalysisResult,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _det(class_name: str, label: str, conf: float) -> Detection:
    return Detection(
        class_name=class_name, label=label,
        confidence=conf, bbox=BoundingBox(0, 0, 10, 10),
    )


def _make_image_result(detections: list[Detection]) -> DetectionResult:
    return DetectionResult(
        source_name="test.jpg",
        timestamp=datetime.datetime.now(),
        detections=detections,
    )


def _make_video_result(frame_counts: list[dict[str, int]], conf: float = 0.7) -> VideoAnalysisResult:
    """frame_counts: a list of {label: count} dicts, one per frame."""
    frames: list[FrameResult] = []
    for i, counts in enumerate(frame_counts):
        dets: list[Detection] = []
        for label, n in counts.items():
            dets.extend(_det(label, label, conf) for _ in range(n))
        dr = DetectionResult(
            source_name=f"f{i}",
            timestamp=datetime.datetime.now(),
            detections=dets,
        )
        frames.append(FrameResult(index=i, timestamp_sec=float(i), detection=dr))
    return VideoAnalysisResult(
        source_name="clip.mp4",
        timestamp=datetime.datetime.now(),
        frame_results=frames,
    )


# ----------------------------------------------------------------------
# analyze() — empty / safe defaults
# ----------------------------------------------------------------------
def test_analyze_empty_input_returns_low_severity_unknown() -> None:
    inp = AnalysisInput(source_name="empty.jpg", counts_by_label={}, average_confidence=0.0)
    a = analyze(inp)

    assert a.total_objects == 0
    assert a.person_count == 0
    assert a.weapon_count == 0
    assert a.vehicle_count == 0
    assert a.bag_count == 0
    assert a.severity_score == 0
    assert a.severity_level == "low"
    assert a.suggested_category == "unknown"
    assert a.has_threat is False
    assert "No crime-relevant" in a.key_observations[0]


# ----------------------------------------------------------------------
# analyze() — single-image cases
# ----------------------------------------------------------------------
def test_analyze_image_with_person_only_is_suspicious_activity() -> None:
    img = _make_image_result([_det("person", "person", 0.9)])
    a = analyze(AnalysisInput.from_image(img))
    assert a.person_count == 1
    assert a.weapon_count == 0
    assert a.suggested_category == "suspicious_activity"
    assert a.has_threat is False
    assert a.severity_level in ("low", "moderate")  # 1 person + high_conf = 8+10=18


def test_analyze_image_with_knife_is_assault() -> None:
    img = _make_image_result([
        _det("person", "person", 0.9),
        _det("person", "person", 0.7),
        _det("knife",  "knife",  0.85),
    ])
    a = analyze(AnalysisInput.from_image(img))
    assert a.person_count == 2
    assert a.weapon_count == 1
    assert a.has_threat is True
    assert a.suggested_category == "assault"
    assert a.severity_score >= 50   # weapon 35 + 2*8 + high_conf 10 = 61
    assert a.severity_level in ("high", "critical")


def test_analyze_image_with_weapon_and_bag_is_robbery() -> None:
    img = _make_image_result([
        _det("person", "person", 0.8),
        _det("knife",  "knife",  0.8),
        _det("backpack", "bag",  0.8),
    ])
    a = analyze(AnalysisInput.from_image(img))
    assert a.suggested_category == "robbery"
    assert a.has_threat is True


def test_analyze_image_with_bag_only_is_theft() -> None:
    img = _make_image_result([
        _det("person", "person", 0.7),
        _det("handbag", "bag", 0.7),
    ])
    a = analyze(AnalysisInput.from_image(img))
    assert a.suggested_category == "theft"
    assert a.has_threat is False


def test_analyze_image_with_vehicle_only_is_vehicle_incident() -> None:
    img = _make_image_result([_det("car", "vehicle", 0.6)])
    a = analyze(AnalysisInput.from_image(img))
    assert a.suggested_category == "vehicle_incident"
    assert a.vehicle_count == 1


# ----------------------------------------------------------------------
# analyze() — video cases
# ----------------------------------------------------------------------
def test_analyze_video_aggregates_across_frames() -> None:
    vid = _make_video_result([
        {"person": 1},
        {"person": 2, "knife": 1},
        {"person": 1},
    ])
    a = analyze(AnalysisInput.from_video(vid))
    assert a.person_count == 4
    assert a.weapon_count == 1
    assert a.total_objects == 5
    assert a.source_type == "video"
    assert a.frame_count == 3
    assert a.suggested_category == "assault"
    # Video observation should mention multiple frames
    assert any("multiple sampled frames" in s for s in a.key_observations)


# ----------------------------------------------------------------------
# analyze() — severity score is bounded
# ----------------------------------------------------------------------
def test_severity_score_clamped_to_100() -> None:
    # Construct a ridiculous input: many persons + weapons.
    dets = (
        [_det("person", "person", 0.95) for _ in range(20)]
        + [_det("knife",  "knife",  0.95) for _ in range(10)]
        + [_det("bottle", "bottle", 0.95) for _ in range(10)]
        + [_det("backpack", "bag",  0.95) for _ in range(5)]
    )
    img = _make_image_result(dets)
    a = analyze(AnalysisInput.from_image(img))
    assert 0 <= a.severity_score <= 100
    assert a.severity_level == "critical"


# ----------------------------------------------------------------------
# AnalysisInput adapters
# ----------------------------------------------------------------------
def test_analysis_input_from_image_pulls_counts() -> None:
    img = _make_image_result([_det("person", "person", 0.5)])
    inp = AnalysisInput.from_image(img)
    assert inp.source_type == "image"
    assert inp.frame_count == 1
    assert inp.counts_by_label == {"person": 1}
    assert inp.average_confidence == pytest.approx(0.5)


def test_analysis_input_from_video_pulls_counts() -> None:
    vid = _make_video_result([{"person": 2}, {"knife": 1}])
    inp = AnalysisInput.from_video(vid)
    assert inp.source_type == "video"
    assert inp.frame_count == 2
    assert inp.counts_by_label == {"person": 2, "knife": 1}
    assert 0.0 < inp.average_confidence <= 1.0


# ----------------------------------------------------------------------
# EvidenceAnalysis.as_dict()
# ----------------------------------------------------------------------
def test_evidence_analysis_as_dict_has_keys() -> None:
    a = analyze(AnalysisInput(
        source_name="x.jpg",
        counts_by_label={"person": 1, "knife": 1},
        average_confidence=0.9,
    ))
    d = a.as_dict()
    for key in [
        "source_name", "source_type", "counts_by_label", "total_objects",
        "unique_labels", "average_confidence", "person_count", "weapon_count",
        "vehicle_count", "bag_count", "severity_score", "severity_level",
        "suggested_category", "has_threat", "key_observations", "timestamp",
    ]:
        assert key in d, f"Missing key: {key}"
