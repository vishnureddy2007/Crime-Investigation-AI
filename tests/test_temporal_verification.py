"""Tests for Phase 47 — temporal weapon verification."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from models.schemas import BoundingBox, Detection, DetectionResult, FrameResult
from models.video_processor import (
    _apply_temporal_weapon_verification,
)


def _det(label: str, conf: float, status: str = "") -> Detection:
    return Detection(
        class_name=label,
        label=label,
        confidence=conf,
        bbox=BoundingBox(0.0, 0.0, 10.0, 10.0),
        source="general",
        weapon_status=status,
    )


def _frame(ts: float, dets: list[Detection]) -> FrameResult:
    return FrameResult(
        index=int(ts),
        timestamp_sec=ts,
        image_path=None,
        annotated_path=None,
        detection=DetectionResult(
            source_name=f"frame_{ts}",
            timestamp=datetime.now(),
            detections=dets,
            raw_count=len(dets),
            annotated_image=None,
            model_name="stub",
            source_tag="general",
            models_used=["general"],
        ),
    )


def _knife_count(frame_results: list[FrameResult]) -> int:
    n = 0
    for fr in frame_results:
        if fr.detection is None:
            continue
        for d in fr.detection.detections:
            if d.label in ("knife", "weapon"):
                n += 1
    return n


class TestTemporalVerification:
    def test_single_frame_does_not_promote(self) -> None:
        """One frame with a knife is not enough."""
        frames = [
            _frame(0.0, [_det("knife", 0.9, status="candidate")]),
            _frame(2.0, []),
        ]
        meta = _apply_temporal_weapon_verification(
            frames,
            min_frames=2,
            max_frame_gap_sec=2.5,
            min_avg_confidence=0.45,
        )
        assert meta == {}

    def test_persistent_weapon_is_promoted(self) -> None:
        """A knife across 3 frames within the gap window gets promoted."""
        frames = [
            _frame(0.0, [_det("knife", 0.55, status="candidate")]),
            _frame(1.0, [_det("knife", 0.55, status="candidate")]),
            _frame(2.0, [_det("knife", 0.55, status="candidate")]),
            _frame(3.0, []),
        ]
        meta = _apply_temporal_weapon_verification(
            frames,
            min_frames=2,
            max_frame_gap_sec=2.5,
            min_avg_confidence=0.45,
        )
        assert "knife" in meta
        assert meta["knife"]["frames"] >= 2
        # All knife detections should now be verified weapons.
        assert _knife_count(frames) == 3
        for fr in frames:
            if fr.detection is None:
                continue
            for d in fr.detection.detections:
                if d.label == "weapon":
                    assert d.weapon_status == "verified"

    def test_low_confidence_persistent_still_not_promoted(self) -> None:
        """3 frames of knife at 0.30 conf → still candidate (avg < 0.45)."""
        frames = [
            _frame(0.0, [_det("knife", 0.30, status="candidate")]),
            _frame(1.0, [_det("knife", 0.30, status="candidate")]),
            _frame(2.0, [_det("knife", 0.30, status="candidate")]),
        ]
        meta = _apply_temporal_weapon_verification(
            frames,
            min_frames=2,
            max_frame_gap_sec=2.5,
            min_avg_confidence=0.45,
        )
        assert meta == {}

    def test_gap_too_large_breaks_run(self) -> None:
        """If the gap exceeds the limit, the run is broken."""
        frames = [
            _frame(0.0, [_det("knife", 0.9, status="candidate")]),
            _frame(20.0, [_det("knife", 0.9, status="candidate")]),  # gap > 2.5s
        ]
        meta = _apply_temporal_weapon_verification(
            frames,
            min_frames=2,
            max_frame_gap_sec=2.5,
            min_avg_confidence=0.45,
        )
        assert meta == {}

    def test_empty_frames_returns_empty(self) -> None:
        assert _apply_temporal_weapon_verification(
            [],
            min_frames=2,
            max_frame_gap_sec=2.5,
            min_avg_confidence=0.45,
        ) == {}

    def test_mixed_labels_promoted_independently(self) -> None:
        """Knife persistent + bottle NOT a weapon."""
        frames = [
            _frame(0.0, [_det("knife", 0.9), _det("bottle", 0.9)]),
            _frame(1.0, [_det("knife", 0.9), _det("bottle", 0.9)]),
            _frame(2.0, [_det("knife", 0.9), _det("bottle", 0.9)]),
        ]
        meta = _apply_temporal_weapon_verification(
            frames,
            min_frames=2,
            max_frame_gap_sec=2.5,
            min_avg_confidence=0.45,
        )
        assert "knife" in meta
        assert "bottle" not in meta
        # Bottle detections must be untouched; knife detections must be promoted.
        for fr in frames:
            if fr.detection is None:
                continue
            bottles = [d for d in fr.detection.detections if d.label == "bottle"]
            weapons = [d for d in fr.detection.detections if d.label == "weapon"]
            assert len(bottles) == 1, "bottle detection must be untouched"
            assert len(weapons) == 1, "knife must be promoted to weapon"
            assert weapons[0].weapon_status == "verified"
