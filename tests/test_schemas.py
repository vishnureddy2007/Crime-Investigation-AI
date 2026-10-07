"""
Tests for the data schemas, including Milestone 3 additions.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from models.schemas import (
    BoundingBox,
    Detection,
    DetectionResult,
    FrameResult,
    VideoAnalysisResult,
)


def test_bounding_box_properties() -> None:
    bb = BoundingBox(10, 20, 110, 120)
    assert bb.width == 100
    assert bb.height == 100
    assert bb.area == 10_000


def test_bounding_box_negative_size_is_clamped() -> None:
    bb = BoundingBox(50, 50, 10, 10)
    assert bb.width == 0
    assert bb.height == 0
    assert bb.area == 0


def test_detection_as_dict_has_keys() -> None:
    d = Detection(
        class_name="person", label="person",
        confidence=0.91, bbox=BoundingBox(0, 0, 10, 10),
    )
    out = d.as_dict()
    assert out["class_name"] == "person"
    assert out["label"] == "person"
    assert out["confidence"] == 0.91
    assert out["bbox"] == {"x1": 0, "y1": 0, "x2": 10, "y2": 10}


def test_detection_result_count_helpers() -> None:
    result = DetectionResult(
        source_name="x", timestamp=datetime.datetime.now(),
    )
    result.detections = [
        Detection("person", "person", 0.9, BoundingBox(0, 0, 10, 10)),
        Detection("knife",  "knife",  0.8, BoundingBox(20, 20, 30, 30)),
        Detection("car",    "vehicle", 0.7, BoundingBox(40, 40, 50, 50)),
    ]
    assert result.count == 3
    assert result.counts_by_label() == {"person": 1, "knife": 1, "vehicle": 1}
    assert 0.0 < result.average_confidence() < 1.0


# ----------------------------------------------------------------------
# Milestone 3: FrameResult + VideoAnalysisResult
# ----------------------------------------------------------------------
def _make_frame(index: int, count: int, conf: float) -> FrameResult:
    dets = [
        Detection("person", "person", conf, BoundingBox(0, 0, 10, 10))
        for _ in range(count)
    ]
    dr = DetectionResult(
        source_name=f"f{index}",
        timestamp=datetime.datetime.now(),
        detections=dets,
    )
    return FrameResult(index=index, timestamp_sec=index * 1.0, detection=dr)


def test_frame_result_evidence_score() -> None:
    f0 = _make_frame(0, count=2, conf=0.9)
    assert f0.evidence_score == 2 * 0.9

    empty = FrameResult(index=99, timestamp_sec=99.0, detection=None)
    assert empty.evidence_score == 0.0


def test_video_analysis_result_keyframes_and_aggregates() -> None:
    frames = [
        _make_frame(0, count=1, conf=0.5),  # score 0.5
        _make_frame(1, count=3, conf=0.9),  # score 2.7  <- top
        _make_frame(2, count=2, conf=0.8),  # score 1.6  <- 2nd
        _make_frame(3, count=0, conf=0.0),  # score 0.0
    ]
    var = VideoAnalysisResult(
        source_name="vid",
        timestamp=datetime.datetime.now(),
        frame_results=frames,
        keyframe_indices=[1, 2],   # manually selected for test
    )
    assert var.frame_count == 4
    assert var.total_detections() == 6
    assert var.keyframes[0].index == 1
    assert var.keyframes[1].index == 2
    assert var.aggregate_counts_by_label() == {"person": 6}
