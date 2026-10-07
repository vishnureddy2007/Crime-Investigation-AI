"""
Tests for VideoProcessor using a stub detector (no real YOLO).

This lets us verify the frame-sampling, scoring, and keyframe-selection
logic without loading the model.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from config import DETECTION_DIR
from models.schemas import BoundingBox, Detection, DetectionResult, FrameResult
from models.video_processor import VideoProcessor

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_MP4 = FIXTURES / "sample.mp4"


# ----------------------------------------------------------------------
# Stub detector - returns a deterministic number of "person" detections
# ----------------------------------------------------------------------
class StubDetector:
    """Returns detections with a controllable confidence value."""

    model_name = "stub"

    def __init__(self, confidence: float = 0.5, n_persons: int = 1) -> None:
        self.confidence = confidence
        self.n_persons = n_persons
        self.calls: list[Any] = []

    def detect_image(self, image: Any, source_name: str = "img") -> DetectionResult:
        self.calls.append(source_name)
        dets = [
            Detection(
                class_name="person", label="person",
                confidence=self.confidence,
                bbox=BoundingBox(0, 0, 10, 10),
            )
            for _ in range(self.n_persons)
        ]
        return DetectionResult(
            source_name=source_name,
            timestamp=datetime.datetime.now(),
            detections=dets,
        )


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_video() -> None:
    if SAMPLE_MP4.exists() and SAMPLE_MP4.stat().st_size > 0:
        return
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(SAMPLE_MP4), fourcc, 10.0, (64, 64))
    for _ in range(30):  # 3 seconds at 10 fps
        writer.write(np.full((64, 64, 3), 128, dtype=np.uint8))
    writer.release()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_processor_samples_frames(tmp_path: Path) -> None:
    detector = StubDetector(confidence=0.9, n_persons=2)
    processor = VideoProcessor(
        detector=detector,          # type: ignore[arg-type]
        keyframe_count=3,
        frame_interval_sec=1.0,
    )

    # Save to tmp so we don't pollute outputs/
    test_video = tmp_path / "in.mp4"
    test_video.write_bytes(SAMPLE_MP4.read_bytes())

    result = processor.process(test_video, source_name="in.mp4", save_frames=False)

    # 2s video, 1s interval -> 2 sampled frames
    assert result.frame_count == 2, f"Expected 2 frames, got {result.frame_count}"
    assert len(detector.calls) == 2

    # Every frame has score = 2 * 0.9 = 1.8
    for fr in result.frame_results:
        assert fr.evidence_score == pytest.approx(1.8)

    # Top-K selection
    assert len(result.keyframe_indices) == 2
    assert result.total_detections() == 4


def test_processor_keyframe_selection_picks_highest(tmp_path: Path) -> None:
    """Vary stub detector per-call by replacing it between runs."""
    detector = StubDetector(confidence=0.5, n_persons=1)
    processor = VideoProcessor(
        detector=detector,          # type: ignore[arg-type]
        keyframe_count=2,
        frame_interval_sec=1.0,
    )

    test_video = tmp_path / "in.mp4"
    test_video.write_bytes(SAMPLE_MP4.read_bytes())

    # Increase the count after the first frame to make later frames score higher
    real_detect = detector.detect_image

    def variable_detect(image: Any, source_name: str = "img") -> DetectionResult:
        # Frames 0..2: 1 person @ 0.5; the rest unchanged.
        return real_detect(image, source_name)

    # We patch the detector's call so frame index 1 has 3 detections @ 0.9.
    call_counter = {"n": 0}

    def varying(image: Any, source_name: str = "img") -> DetectionResult:
        idx = call_counter["n"]
        call_counter["n"] += 1
        if idx == 1:
            return DetectionResult(
                source_name=source_name,
                timestamp=datetime.datetime.now(),
                detections=[
                    Detection("knife", "knife", 0.9, BoundingBox(0, 0, 5, 5))
                    for _ in range(3)
                ],
            )
        return real_detect(image, source_name)

    detector.detect_image = varying  # type: ignore[method-assign]

    result = processor.process(test_video, source_name="in.mp4", save_frames=False)

    # Top 2 keyframes: should include frame index 1 (score 2.7) and the rest (0.5)
    top_two = [result.frame_results[i].index for i in result.keyframe_indices]
    assert 1 in top_two, f"Expected frame 1 in keyframes, got {top_two}"


def test_processor_rejects_missing_file(tmp_path: Path) -> None:
    processor = VideoProcessor(StubDetector())  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        processor.process(tmp_path / "nope.mp4")


# ----------------------------------------------------------------------
# Phase 18 — save_frames=True keyframe save path
# ----------------------------------------------------------------------


class AnnotatedStubDetector:
    """Returns a DetectionResult with an `annotated_image` set."""

    model_name = "stub-annotated"

    def __init__(self, confidence: float = 0.9, n_persons: int = 1) -> None:
        self.confidence = confidence
        self.n_persons = n_persons
        self.calls: list[str] = []

    def detect_image(self, image: Any, source_name: str = "img") -> DetectionResult:
        self.calls.append(source_name)
        # Build a tiny annotated image (a single-color RGB PIL).
        from PIL import Image
        annotated = Image.new("RGB", (8, 8), color=(255, 0, 0))
        dets = [
            Detection(
                class_name="person", label="person",
                confidence=self.confidence,
                bbox=BoundingBox(0, 0, 10, 10),
            )
            for _ in range(self.n_persons)
        ]
        return DetectionResult(
            source_name=source_name,
            timestamp=datetime.datetime.now(),
            detections=dets,
            annotated_image=annotated,
        )


def test_processor_save_frames_writes_per_frame_files(tmp_path: Path, monkeypatch) -> None:
    """save_frames=True writes the per-frame files (creates the dirs)."""
    # Redirect config dirs to tmp so we don't pollute outputs/.
    # VideoProcessor captured DETECTION_DIR / KEYFRAMES_DIR at import
    # time, so we patch them in the video_processor namespace.
    test_det_dir = tmp_path / "detections"
    test_kf_dir = tmp_path / "keyframes"
    monkeypatch.setattr("models.video_processor.DETECTION_DIR", test_det_dir, raising=False)
    monkeypatch.setattr("models.video_processor.KEYFRAMES_DIR", test_kf_dir, raising=False)

    detector = AnnotatedStubDetector(confidence=0.9, n_persons=2)
    processor = VideoProcessor(
        detector=detector,          # type: ignore[arg-type]
        keyframe_count=2,
        frame_interval_sec=1.0,
    )

    test_video = tmp_path / "save_test.mp4"
    test_video.write_bytes(SAMPLE_MP4.read_bytes())

    result = processor.process(test_video, source_name="save_test.mp4", save_frames=True)

    # 2 frames sampled at 1s interval.
    assert len(result.frame_results) >= 2
    # Per-frame image files were saved.
    frames_dir = test_det_dir / "frames" / "save_test"
    assert frames_dir.exists(), f"expected frames dir {frames_dir}"
    jpegs = list(frames_dir.glob("*.jpg"))
    assert jpegs, "expected at least one per-frame jpg"
    # All sampled frames got an annotated_path because the detector
    # always returns annotated_image.
    assert all(fr.annotated_path is not None for fr in result.frame_results)
    # KEYFRAMES_DIR / stem was created during save_frames=True.
    assert (test_kf_dir / "save_test").exists()


def test_processor_keyframe_save_loop_runs_for_unannotated_frames(tmp_path: Path, monkeypatch) -> None:
    """The defensive `if fr.annotated_path is None and fr.detection.annotated_image`
    branch is hit when a frame has `annotated_image` set but `annotated_path`
    was not written by `_process_frame`.

    We simulate the unreachable production state by patching
    `_process_frame` to construct a FrameResult whose `annotated_path`
    is None, even though `detection.annotated_image` carries a PIL
    image.
    """
    from PIL import Image
    test_det_dir = tmp_path / "detections"
    test_kf_dir = tmp_path / "keyframes"
    monkeypatch.setattr("models.video_processor.DETECTION_DIR", test_det_dir, raising=False)
    monkeypatch.setattr("models.video_processor.KEYFRAMES_DIR", test_kf_dir, raising=False)

    detector = AnnotatedStubDetector(confidence=0.9, n_persons=2)
    processor = VideoProcessor(
        detector=detector,          # type: ignore[arg-type]
        keyframe_count=2,
        frame_interval_sec=1.0,
    )

    real_process_frame = processor._process_frame

    def patched_process_frame(frame, index, timestamp_sec, frames_dir, save):
        # Run the real logic to get the actual FrameResult, then rewrite
        # `annotated_path` to None without touching `detection.annotated_image`.
        fr = real_process_frame(frame, index, timestamp_sec, frames_dir, save)
        object.__setattr__(fr, "annotated_path", None)
        return fr

    processor._process_frame = patched_process_frame  # type: ignore[method-assign]

    test_video = tmp_path / "save_test2.mp4"
    test_video.write_bytes(SAMPLE_MP4.read_bytes())

    result = processor.process(test_video, source_name="save_test2.mp4", save_frames=True)

    # The keyframe save loop set `annotated_path` for each selected keyframe.
    # Top-K = min(keyframe_count, n_sampled) frames got their annotated_path
    # back-filled to the keyframe image.
    n_kf = min(processor.keyframe_count, len(result.frame_results))
    annotated_paths = [fr.annotated_path for fr in result.frame_results if fr.annotated_path is not None]
    assert len(annotated_paths) >= n_kf

    kf_dir = test_kf_dir / "save_test2"
    keyframes = list(kf_dir.glob("keyframe_*.jpg"))
    assert keyframes, f"expected keyframe_*.jpg under {kf_dir}"
    # The number of keyframes saved equals min(keyframe_count, n_sampled).
    assert len(keyframes) == min(processor.keyframe_count, len(result.frame_results))


def test_processor_keyframe_save_loop_skips_when_no_annotated_image(tmp_path: Path, monkeypatch) -> None:
    """When `detection.annotated_image` is None, the keyframe save loop
    silently skips (no exception, no keyframe files written for that frame)."""
    test_det_dir = tmp_path / "detections"
    test_kf_dir = tmp_path / "keyframes"
    monkeypatch.setattr("models.video_processor.DETECTION_DIR", test_det_dir, raising=False)
    monkeypatch.setattr("models.video_processor.KEYFRAMES_DIR", test_kf_dir, raising=False)

    # Use the regular StubDetector (no annotated_image).
    detector = StubDetector(confidence=0.9, n_persons=2)
    processor = VideoProcessor(
        detector=detector,          # type: ignore[arg-type]
        keyframe_count=2,
        frame_interval_sec=1.0,
    )

    test_video = tmp_path / "no_image.mp4"
    test_video.write_bytes(SAMPLE_MP4.read_bytes())

    result = processor.process(test_video, source_name="no_image.mp4", save_frames=True)

    # Frames have no annotated_path AND no annotated_image.
    assert all(
        fr.annotated_path is None and
        fr.detection is not None and
        fr.detection.annotated_image is None
        for fr in result.frame_results
    )
    # Keyframe dir was created but no keyframe files written.
    kf_dir = test_kf_dir / "no_image"
    assert kf_dir.exists()
    assert list(kf_dir.glob("keyframe_*.jpg")) == []


def test_processor_save_frames_creates_detection_dirs(tmp_path: Path, monkeypatch) -> None:
    """The mkdir branches at line 89-90 are exercised."""
    test_det_dir = tmp_path / "det_dir_only"
    test_kf_dir = tmp_path / "kf_dir_only"
    monkeypatch.setattr("models.video_processor.DETECTION_DIR", test_det_dir, raising=False)
    monkeypatch.setattr("models.video_processor.KEYFRAMES_DIR", test_kf_dir, raising=False)

    detector = StubDetector()  # no annotated_image at all
    processor = VideoProcessor(detector=detector, keyframe_count=1, frame_interval_sec=1.0)  # type: ignore[arg-type]
    test_video = tmp_path / "mkdir_test.mp4"
    test_video.write_bytes(SAMPLE_MP4.read_bytes())
    processor.process(test_video, source_name="mkdir_test.mp4", save_frames=True)
    assert (test_det_dir / "frames" / "mkdir_test").exists()
    assert (test_kf_dir / "mkdir_test").exists()


# ----------------------------------------------------------------------
# Phase 51 — consolidated video events (one event per run, not per frame)
# ----------------------------------------------------------------------
def _fr(ts: float, label: str, conf: float, status: str = "") -> FrameResult:
    """Build a single-detection FrameResult at timestamp `ts`."""
    det = Detection(
        class_name=label,
        label=label,
        confidence=conf,
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        source="general",
        weapon_status=status,
    )
    return FrameResult(
        index=int(ts * 10),
        timestamp_sec=ts,
        image_path=None,
        annotated_path=None,
        detection=DetectionResult(
            source_name=f"frame_t{ts:.1f}",
            timestamp=datetime.datetime.now(),
            detections=[det],
            raw_count=1,
            annotated_image=None,
            model_name="stub",
            source_tag="general",
        ),
    )


def test_consolidate_video_events_collapses_runs() -> None:
    """Same weapon in 5 contiguous frames → ONE event, not five."""
    from models.video_processor import consolidate_video_events

    frames = [_fr(t, "weapon", 0.60) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
    events = consolidate_video_events(frames, max_frame_gap_sec=2.5, min_frames=2)
    assert len(events) == 1
    e = events[0]
    assert e.label == "weapon"
    assert e.start_sec == 0.0
    assert e.end_sec == 2.0
    assert e.frame_count == 5
    assert e.status == "verified"


def test_consolidate_video_events_splits_on_large_gap() -> None:
    """A 4-second gap breaks one run into two events."""
    from models.video_processor import consolidate_video_events

    frames = (
        [_fr(t, "weapon", 0.50) for t in (0.0, 1.0, 2.0)]
        + [_fr(t, "weapon", 0.50) for t in (10.0, 11.0, 12.0)]
    )
    events = consolidate_video_events(frames, max_frame_gap_sec=2.5, min_frames=2)
    assert len(events) == 2
    starts = sorted(e.start_sec for e in events)
    assert starts == [0.0, 10.0]


def test_consolidate_video_events_drops_short_runs() -> None:
    """A run shorter than `min_frames` is silently dropped (noise filter)."""
    from models.video_processor import consolidate_video_events

    frames = [_fr(0.0, "weapon", 0.60), _fr(0.5, "weapon", 0.60)]
    events = consolidate_video_events(frames, max_frame_gap_sec=2.5, min_frames=3)
    assert events == []


def test_consolidate_video_events_marks_low_confidence_as_candidate() -> None:
    """A run that clears the frame-count bar but not the confidence bar → candidate."""
    from models.video_processor import consolidate_video_events

    frames = [_fr(t, "weapon", 0.30) for t in (0.0, 1.0, 2.0)]
    events = consolidate_video_events(frames, max_frame_gap_sec=2.5, min_frames=2, min_avg_confidence=0.45)
    assert len(events) == 1
    assert events[0].status == "candidate"


def test_consolidate_video_events_handles_multiple_labels() -> None:
    """Person + weapon in the same frames → one event per label, independent."""
    from models.video_processor import consolidate_video_events

    frames = [
        _fr(0.0, "person",  0.90),
        _fr(0.0, "weapon", 0.70),
        _fr(1.0, "person",  0.85),
        _fr(1.0, "weapon", 0.65),
        _fr(2.0, "person",  0.95),
        _fr(2.0, "weapon", 0.72),
    ]
    events = consolidate_video_events(frames, max_frame_gap_sec=2.5, min_frames=2)
    by_label = {e.label: e for e in events}
    assert set(by_label) == {"person", "weapon"}
    assert by_label["person"].frame_count == 3
    assert by_label["weapon"].frame_count == 3


def test_consolidate_video_events_empty_input() -> None:
    """Empty frame_results → empty event list (no crash)."""
    from models.video_processor import consolidate_video_events

    assert consolidate_video_events([]) == []
