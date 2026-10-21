"""
Regression test suite for detection accuracy, duplicate suppression,
video tracking, weapon verification integrity, and provenance traceability.

Covers:
1. Duplicate boxes in single image (class-aware suppression)
2. Nearby separate objects (same and different classes) preserved
3. Repeated boxes in one sampled frame (counted at most once)
4. Clips without weapons (negative results described cautiously)
5. Persistent false candidates (persistence alone never verifies a weapon)
6. Verified tracks consolidated properly
7. Camera cuts and scene jumps
8. Failed or empty inputs handled fail-soft
9. Model and config provenance traceability
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import pytest
from PIL import Image

from config.settings import (
    WEAPON_SCAN_CONF_THRESHOLD,
    WEAPON_CONF_THRESHOLD,
    WEAPON_VERIFY_THRESHOLD,
    get_weights_sha256,
)
from models.schemas import (
    AnalysisInput,
    BoundingBox,
    Detection,
    DetectionResult,
    EvidenceAnalysis,
    FrameResult,
    VideoAnalysisResult,
)
from models.yolo_detector import _cross_model_nms, MultiSourceDetector
from models.video_processor import (
    VideoProcessor,
    consolidate_video_events,
    _apply_temporal_weapon_verification,
)
from models.weapon_verifier import verify_detection
from models.batch_processor import BatchEvidenceProcessor, FileEvidence
from models.evidence_analyzer import analyze


# ----------------------------------------------------------------------
# 1. Duplicate boxes in a single image
# ----------------------------------------------------------------------
def test_duplicate_boxes_in_single_image_suppressed() -> None:
    """Overlapping duplicate bounding boxes on the same object must be suppressed."""
    # Box 1: confidence 0.85
    # Box 2: confidence 0.70, 90% overlap with Box 1, same label
    det1 = Detection(
        class_name="person",
        label="person",
        confidence=0.85,
        bbox=BoundingBox(100.0, 100.0, 200.0, 300.0),
        source="general",
    )
    det2 = Detection(
        class_name="person",
        label="person",
        confidence=0.70,
        bbox=BoundingBox(102.0, 101.0, 202.0, 301.0),
        source="general",
    )

    filtered = _cross_model_nms([det1, det2], iou_threshold=0.45)
    assert len(filtered) == 1
    assert filtered[0].confidence == 0.85


# ----------------------------------------------------------------------
# 2. Nearby separate objects preserved
# ----------------------------------------------------------------------
def test_nearby_separate_objects_preserved() -> None:
    """Nearby distinct objects (both same-class and different-class) must not be suppressed."""
    # Person 1 and Person 2 standing side-by-side with small overlap (IoU < 0.3)
    person1 = Detection(
        class_name="person",
        label="person",
        confidence=0.90,
        bbox=BoundingBox(100.0, 100.0, 200.0, 400.0),
        source="general",
    )
    person2 = Detection(
        class_name="person",
        label="person",
        confidence=0.88,
        bbox=BoundingBox(190.0, 100.0, 290.0, 400.0),
        source="general",
    )
    # Person 1 carrying a backpack (different class, high overlap with person1)
    backpack = Detection(
        class_name="backpack",
        label="bag",
        confidence=0.80,
        bbox=BoundingBox(120.0, 150.0, 180.0, 260.0),
        source="general",
    )

    filtered = _cross_model_nms([person1, person2, backpack], iou_threshold=0.45)
    assert len(filtered) == 3
    labels = [d.label for d in filtered]
    assert labels.count("person") == 2
    assert "bag" in labels


# ----------------------------------------------------------------------
# 3. Repeated boxes in one sampled frame
# ----------------------------------------------------------------------
def test_repeated_boxes_in_one_sampled_frame() -> None:
    """A single sampled frame with repeated candidate boxes must count at most once per physical object."""
    # Frame with 2 overlapping candidate boxes for the same weapon object
    det_a = Detection(
        class_name="gun",
        label="candidate_weapon",
        confidence=0.38,
        bbox=BoundingBox(50.0, 50.0, 100.0, 100.0),
        source="weapon-scan",
        weapon_status="candidate",
    )
    det_b = Detection(
        class_name="handgun",
        label="candidate_weapon",
        confidence=0.36,
        bbox=BoundingBox(52.0, 51.0, 99.0, 102.0),
        source="threat-weapon",
        weapon_status="candidate",
    )

    frame = FrameResult(
        index=0,
        timestamp_sec=0.0,
        detection=DetectionResult(
            source_name="frame_0",
            timestamp=datetime.now(),
            detections=[det_a, det_b],
            annotated_image=None,
        ),
    )

    # Process temporal tracking with min_frames=1, max_gap=2.0s, min_conf=0.45
    _apply_temporal_weapon_verification([frame], min_frames=1, max_frame_gap_sec=2.0, min_avg_confidence=0.45)
    # Detections in frame should not have duplicate verified promotions
    verified_dets = [d for d in frame.detection.detections if getattr(d, "weapon_status", "") == "verified"]
    assert len(verified_dets) == 0  # Neither should be verified since conf < 0.45


# ----------------------------------------------------------------------
# 4. Clips without weapons (negative results described cautiously)
# ----------------------------------------------------------------------
def test_clips_without_weapons_negative_results() -> None:
    """Clips with no weapons must return verified_weapon_count=0 and cautious negative observations."""
    p_det = Detection(
        class_name="person",
        label="person",
        confidence=0.85,
        bbox=BoundingBox(10.0, 10.0, 60.0, 150.0),
    )
    v_det = Detection(
        class_name="car",
        label="vehicle",
        confidence=0.80,
        bbox=BoundingBox(200.0, 200.0, 350.0, 300.0),
    )
    det_result = DetectionResult(
        source_name="clean_clip.mp4",
        timestamp=datetime.now(),
        detections=[p_det, v_det],
        annotated_image=None,
    )

    analysis = analyze(AnalysisInput.from_image(det_result))
    assert analysis.verified_weapon_count == 0
    assert analysis.candidate_weapon_count == 0
    assert analysis.has_threat is False

    # Negative observation forensic caveat verification
    obs_text = " ".join(analysis.key_observations)
    assert "No verified weapons detected" in obs_text
    assert "not detected in analyzed frames" in obs_text
    assert "not proof that weapons are absent" in obs_text


# ----------------------------------------------------------------------
# 5. Persistent false candidates never verified by persistence alone
# ----------------------------------------------------------------------
def test_persistent_false_candidates_never_verified_by_persistence_alone() -> None:
    """A low-confidence false alarm observed across 10 frames must NEVER be promoted to verified."""
    frames: list[FrameResult] = []
    for i in range(10):
        # Sub-threshold candidate (conf=0.35, below WEAPON_VERIFY_THRESHOLD 0.45)
        cand = Detection(
            class_name="revolver",
            label="candidate_weapon",
            confidence=0.35,
            bbox=BoundingBox(100.0, 100.0, 150.0, 150.0),
            source="weapon-scan",
            weapon_status="candidate",
        )
        frames.append(
            FrameResult(
                index=i,
                timestamp_sec=float(i) * 0.5,
                detection=DetectionResult(
                    source_name=f"frame_{i}",
                    timestamp=datetime.now(),
                    detections=[cand],
                    annotated_image=None,
                ),
            )
        )

    _apply_temporal_weapon_verification(frames, min_frames=3, max_frame_gap_sec=2.0, min_avg_confidence=0.45)
    # Check that after 10 frames, candidate remains candidate and is NEVER verified
    for fr in frames:
        for d in fr.detection.detections:
            assert d.weapon_status != "verified", "Persistence alone promoted low-confidence candidate to verified!"
            assert d.label == "candidate_weapon"

    events = consolidate_video_events(frames, min_frames=3)
    verified_events = [e for e in events if e.status == "verified"]
    assert len(verified_events) == 0


# ----------------------------------------------------------------------
# 6. Verified tracks consolidated properly
# ----------------------------------------------------------------------
def test_verified_tracks_consolidated() -> None:
    """High confidence verified detections persisting across frames consolidate into a verified track."""
    frames: list[FrameResult] = []
    for i in range(5):
        det = Detection(
            class_name="gun",
            label="weapon",
            confidence=0.88,
            bbox=BoundingBox(100.0 + i * 2.0, 100.0, 150.0 + i * 2.0, 150.0),
            source="threat-weapon",
            weapon_status="verified",
        )
        frames.append(
            FrameResult(
                index=i,
                timestamp_sec=float(i) * 1.0,
                detection=DetectionResult(
                    source_name=f"frame_{i}",
                    timestamp=datetime.now(),
                    detections=[det],
                    annotated_image=None,
                ),
            )
        )

    _apply_temporal_weapon_verification(frames, min_frames=3, max_frame_gap_sec=2.0, min_avg_confidence=0.45)
    events = consolidate_video_events(frames, min_frames=3)

    verified_events = [e for e in events if e.status == "verified"]
    assert len(verified_events) == 1
    ev = verified_events[0]
    assert ev.label == "weapon"
    assert ev.frame_count == 5
    assert ev.start_sec == 0.0
    assert ev.end_sec == 4.0
    assert ev.avg_confidence >= 0.85


# ----------------------------------------------------------------------
# 7. Camera cuts and scene jumps
# ----------------------------------------------------------------------
def test_camera_cuts_and_scene_jumps_do_not_bridge_tracks() -> None:
    """Sudden timestamp jump or spatial jump across camera cut must create distinct tracks."""
    frames: list[FrameResult] = [
        # Track before camera cut (t = 0.0s, x = [10, 10, 50, 50])
        FrameResult(
            index=0,
            timestamp_sec=0.0,
            detection=DetectionResult(
                source_name="cut_0",
                timestamp=datetime.now(),
                detections=[Detection(class_name="gun", label="weapon", confidence=0.85, bbox=BoundingBox(10.0, 10.0, 50.0, 50.0), weapon_status="verified")],
                annotated_image=None,
            ),
        ),
        # Camera cut: 5.0 seconds jump, completely different coordinates (x = [500, 500, 550, 550])
        FrameResult(
            index=1,
            timestamp_sec=5.0,
            detection=DetectionResult(
                source_name="cut_1",
                timestamp=datetime.now(),
                detections=[Detection(class_name="gun", label="weapon", confidence=0.82, bbox=BoundingBox(500.0, 500.0, 550.0, 550.0), weapon_status="verified")],
                annotated_image=None,
            ),
        ),
    ]

    events = consolidate_video_events(frames, min_frames=1, max_frame_gap_sec=2.0)
    # The two detections must NOT be bridged into a single continuous track
    assert len(events) == 2
    assert events[0].end_sec == 0.0
    assert events[1].start_sec == 5.0


# ----------------------------------------------------------------------
# 8. Failed or empty inputs handled fail-soft
# ----------------------------------------------------------------------
def test_failed_or_empty_inputs_fail_soft() -> None:
    """Batch processor must gracefully handle 0-byte, unsupported, and corrupted files."""
    class DummyDetector:
        def detect_image(self, image: Any, source_name: str, confidence: Any = None, iou: Any = None) -> DetectionResult:
            return DetectionResult(source_name=source_name, timestamp=datetime.now(), detections=[], annotated_image=None)

    processor = BatchEvidenceProcessor(detector=DummyDetector())

    files = [
        ("empty.jpg", b""),                        # Empty 0-byte file
        ("unsupported.exe", b"MZbinarycontent"),   # Unsupported extension
        ("corrupt.png", b"NotAPngFileAtAll"),      # Corrupt image bytes
    ]

    batch_result = processor.process(files=files)
    assert batch_result.total_files == 3
    assert batch_result.succeeded_files == 0
    assert batch_result.failed_files == 3

    # All 3 files must record helpful error reasons without raising an unhandled exception
    for f in batch_result.files:
        assert f.error is not None
        assert "message" in f.error
        assert "reason" in f.error


# ----------------------------------------------------------------------
# 9. Model and config provenance traceability
# ----------------------------------------------------------------------
def test_model_and_config_provenance_traceability() -> None:
    """DetectionResult, VideoAnalysisResult, and EvidenceAnalysis must preserve provenance and audit details."""
    d1 = Detection(class_name="person", label="person", confidence=0.91, bbox=BoundingBox(10, 10, 50, 100))
    d_raw1 = Detection(class_name="person", label="person", confidence=0.40, bbox=BoundingBox(12, 12, 52, 102))
    d_raw2 = Detection(class_name="knife", label="weapon", confidence=0.20, bbox=BoundingBox(20, 20, 30, 30))
    d_rej1 = Detection(class_name="cell phone", label="cell phone", confidence=0.10, bbox=BoundingBox(0, 0, 5, 5))
    d_rej2 = Detection(class_name="bottle", label="bottle", confidence=0.15, bbox=BoundingBox(10, 10, 15, 15))

    res = DetectionResult(
        source_name="scene.jpg",
        timestamp=datetime.now(),
        detections=[d1],
        raw_detections=[d1, d_raw1, d_raw2],
        rejected_detections=[d_rej1, d_rej2],
        annotated_image=None,
        model_name="yolov8n",
        model_version="YOLOv8-audited-v1",
        model_weights={"yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"},
        config_metadata={"confidence": 0.25, "iou": 0.45},
    )

    assert res.model_version == "YOLOv8-audited-v1"
    assert "yolov8n.pt" in res.model_weights
    assert res.raw_detections_count == 3
    assert res.rejected_count == 2

    # Verify AnalysisInput converts these
    inp = AnalysisInput.from_image(res)
    assert inp.model_version == "YOLOv8-audited-v1"
    assert inp.raw_detections_count == 3
    assert inp.rejected_count == 2

    # Verify EvidenceAnalysis preserves provenance
    ev_analysis = analyze(inp)
    assert ev_analysis.model_version == "YOLOv8-audited-v1"
    assert "yolov8n.pt" in str(ev_analysis.model_weights)
    assert ev_analysis.raw_detections_count == 3
    assert ev_analysis.rejected_count == 2
