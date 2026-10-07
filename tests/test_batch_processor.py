"""
Tests for the multi-source detector + batch processor.

These tests use stub detectors so they run without GPU, network, or
the Ultralytics runtime. The goal is to verify the *merge* logic
and the *batch orchestration* logic, not model accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from models.analysis_engine import build_combined_analysis
from models.batch_processor import (
    BatchEvidenceProcessor,
    FileEvidence,
    classify_file,
)
from models.schemas import (
    AnalysisInput,
    BoundingBox,
    Detection,
    DetectionResult,
)
from models.yolo_detector import (
    MultiSourceDetector,
    YOLODetector,
    _cross_model_nms,
    _iou,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _det(label: str, conf: float, x1: float, y1: float, x2: float, y2: float,
         source: str = "general") -> Detection:
    return Detection(
        class_name=label,
        label=label,
        confidence=conf,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        source=source,
    )


def _img(name: str = "x.png") -> Image.Image:
    return Image.new("RGB", (16, 16), color=(255, 0, 0))


# ----------------------------------------------------------------------
# classify_file
# ----------------------------------------------------------------------
def test_classify_file_recognises_supported_extensions() -> None:
    assert classify_file("a.jpg") == "image"
    assert classify_file("b.jpeg") == "image"
    assert classify_file("c.png") == "image"
    assert classify_file("d.webp") == "image"
    assert classify_file("e.bmp") == "image"
    assert classify_file("v.mp4") == "video"
    assert classify_file("v.avi") == "video"
    assert classify_file("v.mov") == "video"
    assert classify_file("v.mkv") == "video"
    assert classify_file("v.webm") == "video"


def test_classify_file_rejects_unsupported() -> None:
    assert classify_file("doc.pdf") == "unsupported"
    assert classify_file("data.csv") == "unsupported"
    assert classify_file("script.py") == "unsupported"


def test_classify_file_is_case_insensitive() -> None:
    assert classify_file("A.JPG") == "image"
    assert classify_file("B.Mp4") == "video"


# ----------------------------------------------------------------------
# IoU + cross-model NMS
# ----------------------------------------------------------------------
def test_iou_identical_boxes_is_one() -> None:
    a = BoundingBox(0, 0, 10, 10)
    assert _iou(a, a) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(20, 20, 30, 30)
    assert _iou(a, b) == 0.0


def test_cross_model_nms_drops_overlapping_same_label() -> None:
    a = _det("knife", 0.9, 10, 10, 60, 60, source="general")
    b = _det("knife", 0.7, 12, 12, 62, 62, source="weapon-scan")
    kept = _cross_model_nms([a, b])
    assert len(kept) == 1
    assert kept[0].source == "general"
    assert kept[0].confidence == 0.9


def test_cross_model_nms_keeps_different_labels() -> None:
    person = _det("person", 0.9, 10, 10, 60, 60)
    knife   = _det("knife",  0.7, 12, 12, 62, 62)
    kept = _cross_model_nms([person, knife])
    assert len(kept) == 2


def test_cross_model_nms_keeps_non_overlapping() -> None:
    a = _det("knife", 0.9, 0, 0, 10, 10)
    b = _det("knife", 0.7, 100, 100, 110, 110)
    kept = _cross_model_nms([a, b])
    assert len(kept) == 2


# ----------------------------------------------------------------------
# MultiSourceDetector
# ----------------------------------------------------------------------
class _StubDetector:
    """Stand-in for YOLODetector.detect_image_raw."""

    def __init__(self, detections_by_call: list[list[Detection]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._script = detections_by_call or []
        self._i = 0
        # `MultiSourceDetector.detect_image` reads `self.general.confidence`
        # as a fallback when no override is supplied. Stubs must expose it.
        self.confidence = 0.4
        self.iou = 0.45

    def detect_image_raw(self, image, source_name="", confidence=0.0,
                         iou=0.0, source_tag="general") -> list[Detection]:
        self.calls.append({
            "confidence": confidence,
            "source_tag": source_tag,
        })
        if self._i < len(self._script):
            out = self._script[self._i]
            self._i += 1
            return out
        return []


def test_multi_source_returns_merged_detections() -> None:
    a = _det("person", 0.9, 0, 0, 50, 50, source="general")
    b = _det("knife",  0.7, 60, 60, 80, 80, source="weapon-scan")
    stub = _StubDetector(detections_by_call=[
        [a],         # general run 1
        [b],         # general run 2 (weapon scan)
        [],          # optional weapon model
    ])
    detector = MultiSourceDetector(general=stub)  # type: ignore[arg-type]
    # bypass loading YOLO internally
    detector.general = stub  # type: ignore[assignment]
    result = detector.detect_image(image=_img(), source_name="x.png")
    labels = sorted(d.label for d in result.detections)
    assert labels == ["knife", "person"]
    assert "general" in result.models_used
    assert "weapon-scan" in result.models_used


def test_multi_source_weapon_model_missing_returns_none() -> None:
    detector = MultiSourceDetector(weapon_model_path="non_existent_model_path.pt")
    assert detector.weapon_model_loaded is False


def test_multi_source_dedups_same_label_overlap() -> None:
    a = _det("knife", 0.9, 10, 10, 60, 60, source="general")
    b = _det("knife", 0.85, 11, 11, 61, 61, source="weapon-scan")
    stub = _StubDetector(detections_by_call=[[a], [b], []])
    detector = MultiSourceDetector(general=stub)  # type: ignore[arg-type]
    detector.general = stub  # type: ignore[assignment]
    result = detector.detect_image(image=_img())
    knives = [d for d in result.detections if d.label == "knife"]
    assert len(knives) == 1
    assert knives[0].confidence == 0.9


# ----------------------------------------------------------------------
# BatchEvidenceProcessor
# ----------------------------------------------------------------------
class _StubBatchDetector:
    """Stand-in for MultiSourceDetector.detect_image."""

    def __init__(self, *, image_detections: dict[str, list[Detection]] | None = None,
                 raise_for: set[str] | None = None) -> None:
        self.image_detections = image_detections or {}
        self.raise_for = raise_for or set()
        self.last_confidence: float | None = None
        # expose the `general.confidence` attribute for the video path
        self.general = self

    def detect_image(self, image, source_name="", confidence=None, iou=None) -> DetectionResult:
        if source_name in self.raise_for:
            raise RuntimeError("model failure")
        self.last_confidence = confidence
        detections = list(self.image_detections.get(source_name, []))
        return DetectionResult(
            source_name=source_name,
            timestamp=__import__("datetime").datetime.now(),
            detections=detections,
            raw_count=len(detections),
            annotated_image=image,
            model_name="stub",
            source_tag="general",
            models_used=["general"],
        )

    def detect_image_raw(self, image, source_name="", confidence=None,
                         iou=None, source_tag="general") -> list[Detection]:
        # Used by the video path.
        return self.detect_image(
            image, source_name=source_name, confidence=confidence, iou=iou
        ).detections


def _build_stub_processor() -> tuple[_StubBatchDetector, BatchEvidenceProcessor]:
    stub = _StubBatchDetector(image_detections={
        "person.jpg": [_det("person", 0.9, 0, 0, 50, 50)],
        "knife.jpg":  [_det("knife", 0.7, 10, 10, 60, 60)],
        "car.jpg":    [_det("vehicle", 0.85, 0, 0, 100, 60)],
    })
    return stub, BatchEvidenceProcessor(detector=stub)


def _image_bytes() -> bytes:
    """A minimal valid PNG (1x1 red pixel)."""
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_batch_processes_mixed_images_per_file() -> None:
    _, processor = _build_stub_processor()
    files = [
        ("person.jpg", _image_bytes()),
        ("knife.jpg",  _image_bytes()),
        ("car.jpg",    _image_bytes()),
    ]
    batch = processor.process(files)
    assert batch.total_files == 3
    assert batch.succeeded_files == 3
    assert batch.failed_files == 0
    counts = batch.totals()
    assert counts["person_count"] == 1
    assert counts["weapon_count"] == 1
    assert counts["vehicle_count"] == 1


def test_batch_isolates_failures() -> None:
    stub = _StubBatchDetector(
        image_detections={"good.jpg": [_det("person", 0.9, 0, 0, 50, 50)]},
        raise_for={"bad.jpg"},
    )
    processor = BatchEvidenceProcessor(detector=stub)
    files = [
        ("good.jpg", _image_bytes()),
        ("bad.jpg",  _image_bytes()),
    ]
    batch = processor.process(files)
    assert batch.succeeded_files == 1
    assert batch.failed_files == 1
    failed = [f for f in batch.files if not f.succeeded]
    assert failed and failed[0].error is not None
    assert "bad.jpg" in failed[0].error["message"]


def test_batch_records_empty_file_as_error() -> None:
    stub = _StubBatchDetector()
    processor = BatchEvidenceProcessor(detector=stub)
    batch = processor.process([("empty.jpg", b"")])
    assert batch.failed_files == 1
    assert batch.files[0].error is not None
    assert batch.files[0].error["reason"] == "empty"


def test_batch_records_unsupported_extension() -> None:
    stub = _StubBatchDetector()
    processor = BatchEvidenceProcessor(detector=stub)
    batch = processor.process([("doc.pdf", b"%PDF-1.4")])
    assert batch.failed_files == 1
    assert batch.files[0].error is not None
    assert batch.files[0].error["reason"] == "extension"


def test_batch_combined_analysis_aggregates_counts() -> None:
    _, processor = _build_stub_processor()
    files = [
        ("person.jpg", _image_bytes()),
        ("knife.jpg",  _image_bytes()),
        ("car.jpg",    _image_bytes()),
    ]
    batch = processor.process(files)
    assert batch.combined_analysis is not None
    a = batch.combined_analysis
    assert a.source_type == "batch"
    assert a.person_count == 1
    assert a.weapon_count == 1
    assert a.vehicle_count == 1


def test_batch_no_successful_files_returns_no_combined_analysis() -> None:
    stub = _StubBatchDetector()
    processor = BatchEvidenceProcessor(detector=stub)
    batch = processor.process([("doc.pdf", b"x")])
    assert batch.combined_analysis is None


def test_batch_per_file_metrics_track_knives_and_guns() -> None:
    """Phase 46: bottle is no longer treated as a weapon. Knife and gun
    (verified weapons) DO count toward ``weapon_count``; bottle counts
    toward ``bottle_count`` only."""
    stub = _StubBatchDetector(image_detections={
        "knife.jpg":  [_det("knife", 0.9, 0, 0, 50, 50)],
        "gun.jpg":     [_det("gun", 0.9, 0, 0, 50, 50)],
        "bottle.jpg":  [_det("bottle", 0.9, 0, 0, 50, 50)],
    })
    processor = BatchEvidenceProcessor(detector=stub)
    batch = processor.process([
        ("knife.jpg", _image_bytes()),
        ("gun.jpg",   _image_bytes()),
        ("bottle.jpg", _image_bytes()),
    ])
    totals = batch.totals()
    assert totals["knife_count"] == 1
    assert totals["gun_count"] == 1
    assert totals["bottle_count"] == 1
    # Only knife and gun are weapons (2), bottle is not.
    assert totals["weapon_count"] == 2


# ----------------------------------------------------------------------
# Video path
# ----------------------------------------------------------------------
class _StubVideoProcessor:
    """Records the call and returns a fake VideoAnalysisResult."""

    def __init__(self, detections: list[Detection] | None = None,
                 frame_count: int = 3) -> None:
        self._detections = detections or []
        self._frame_count = frame_count
        self.calls: list[dict[str, Any]] = []

    def process(self, video_path, source_name="", save_frames=True):
        from datetime import datetime as _dt
        from models.schemas import DetectionResult, FrameResult, VideoAnalysisResult
        frames = []
        for i in range(self._frame_count):
            dr = DetectionResult(
                source_name=f"frame_{i}",
                timestamp=_dt.now(),
                detections=list(self._detections),
                raw_count=len(self._detections),
                annotated_image=_img(),
                model_name="stub",
                source_tag="general",
                models_used=["general"],
            )
            frames.append(FrameResult(
                index=i, timestamp_sec=float(i),
                image_path=None, annotated_path=None, detection=dr,
            ))
        self.calls.append({"path": str(video_path), "source": source_name})
        return VideoAnalysisResult(
            source_name=source_name,
            timestamp=_dt.now(),
            metadata={"fps": 1.0, "frame_count": self._frame_count,
                      "width": 16, "height": 16, "duration_sec": float(self._frame_count)},
            frame_results=frames,
            keyframe_indices=[0],
            model_name="stub",
        )


def test_batch_processes_video_with_stub_processor(tmp_path: Path) -> None:
    """Image-only stubs won't accept the video path; provide a
    dedicated video processor that returns frames populated with
    detections."""
    stub_detector = _StubBatchDetector(image_detections={})
    video_processor = _StubVideoProcessor(
        detections=[_det("person", 0.9, 0, 0, 50, 50)],
        frame_count=3,
    )
    processor = BatchEvidenceProcessor(
        detector=stub_detector,
        video_processor_factory=lambda d: video_processor,
    )
    # The video path requires a valid video file. The stub doesn't
    # actually decode the bytes — it just records the call. We
    # monkey-patch the validation step to skip the OpenCV round-trip.
    # Note: the batch processor imports `validate_video_bytes` from
    # `utils.video_io` at module load, so we patch both the source
    # module AND the bound reference inside `batch_processor`.
    import utils.video_io as vio
    import models.batch_processor as bp
    original_validate_source = vio.validate_video_bytes
    original_validate_target = bp.validate_video_bytes
    bp.validate_video_bytes = lambda data, filename="video": None
    vio.validate_video_bytes = lambda data, filename="video": None
    try:
        batch = processor.process([("clip.mp4", b"\x00\x00\x00\x18ftypmp42")])
    finally:
        bp.validate_video_bytes = original_validate_target
        vio.validate_video_bytes = original_validate_source
    assert batch.succeeded_files == 1
    assert batch.failed_files == 0
    f = batch.files[0]
    assert f.source_type == "video"
    assert f.video is not None
    assert f.video.frame_count == 3
    assert f.metrics["person_count"] == 3
    assert f.metrics["frame_count"] == 3
    assert batch.totals()["video_count"] == 1


def test_batch_video_validation_failure_isolated() -> None:
    """If validation_video_bytes raises, the file is marked failed
    but the rest of the batch continues."""
    stub = _StubBatchDetector(image_detections={
        "person.jpg": [_det("person", 0.9, 0, 0, 50, 50)],
    })
    processor = BatchEvidenceProcessor(detector=stub)
    import utils.video_io as vio
    import models.batch_processor as bp
    original_source = vio.validate_video_bytes
    original_target = bp.validate_video_bytes

    def _raise(*args, **kwargs):
        raise vio.VideoLoadError("corrupted")

    bp.validate_video_bytes = _raise
    vio.validate_video_bytes = _raise
    try:
        batch = processor.process([
            ("bad.mp4",  b"x"),
            ("good.jpg", _image_bytes()),
        ])
    finally:
        bp.validate_video_bytes = original_target
        vio.validate_video_bytes = original_source
    assert batch.succeeded_files == 1
    assert batch.failed_files == 1
    failed = [f for f in batch.files if not f.succeeded]
    assert "bad.mp4" in failed[0].filename


# ----------------------------------------------------------------------
# Stand-alone config checks
# ----------------------------------------------------------------------
def test_config_exports_weapon_settings() -> None:
    """`config` re-exports the new weapon-detection knobs so the
    multi-source detector can import them as cheaply as any other.

    Phase 46: bottle / scissors / baseball bat are NOT weapons.
    COCO_WEAPON_CLASSES is now only ``{"knife"}`` and
    SUPPORTED_DETECTION_CLASSES has been removed (the detector maps
    every detection; the analyzer decides what's a weapon).
    """
    from config import (
        COCO_WEAPON_CLASSES,
        CROSS_MODEL_IOU_THRESHOLD,
        WEAPON_CANDIDATE_LABELS,
        WEAPON_CONF_THRESHOLD,
        WEAPON_HIGH_CONF_THRESHOLD,
        WEAPON_MODEL_NAME,
        WEAPON_VERIFY_THRESHOLD,
        YOLO_CLASS_MAPPING,
    )
    # Knife is the only COCO class that maps to a weapon by default.
    assert "knife" in COCO_WEAPON_CLASSES
    assert "bottle" not in COCO_WEAPON_CLASSES
    assert "scissors" not in COCO_WEAPON_CLASSES
    # The verifier exposes its candidate label set.
    assert "knife" in WEAPON_CANDIDATE_LABELS
    assert "weapon" in WEAPON_CANDIDATE_LABELS
    # The mapping must NOT promote non-weapons to weapons.
    for forbidden in ("bottle", "scissors", "baseball bat", "explosion"):
        if forbidden in YOLO_CLASS_MAPPING:
            assert YOLO_CLASS_MAPPING[forbidden] != "weapon"
    # Thresholds are present and in the (0, 1) interval.
    assert 0 < WEAPON_CONF_THRESHOLD < 1
    assert 0 < WEAPON_VERIFY_THRESHOLD < 1
    assert 0 < WEAPON_HIGH_CONF_THRESHOLD < 1
    assert 0 < CROSS_MODEL_IOU_THRESHOLD < 1
    assert WEAPON_MODEL_NAME.endswith(".pt")


# ----------------------------------------------------------------------
# build_combined_analysis (standalone)
# ----------------------------------------------------------------------
def test_build_combined_analysis_empty_returns_safe_defaults() -> None:
    a = build_combined_analysis([], total_files=0)
    assert a.total_objects == 0
    assert a.severity_score == 0
    assert a.weapon_count == 0
    assert a.has_threat is False
    assert a.source_type == "batch"


def test_build_combined_analysis_aggregates_per_file_inputs() -> None:
    i1 = AnalysisInput(
        source_name="a.jpg",
        counts_by_label={"person": 2, "vehicle": 1},
        average_confidence=0.9,
        frame_count=1,
        source_type="image",
    )
    i2 = AnalysisInput(
        source_name="b.mp4",
        counts_by_label={"knife": 1, "person": 1},
        average_confidence=0.7,
        frame_count=10,
        source_type="video",
    )
    a = build_combined_analysis([i1, i2], total_files=2)
    assert a.person_count == 3
    assert a.counts_by_label["knife"] == 1
    assert a.weapon_count == 1
    assert a.vehicle_count == 1
    assert a.frame_count == 11
    assert a.has_threat is True
    assert a.source_type == "batch"
    # Prepended summary line mentions file counts.
    assert any("file(s)" in obs for obs in a.key_observations)
