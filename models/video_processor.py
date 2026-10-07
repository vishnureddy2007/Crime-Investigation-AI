"""
Video processor: extracts frames, runs YOLO per frame, selects keyframes.

The processor takes a YOLODetector instance (Milestone 2) and a path to
a saved video file. It returns a VideoAnalysisResult.

Phase 47 — temporal verification: after per-frame detection, a weapon
candidate is promoted to ``verified_weapon`` only when the same label
appears in at least ``MIN_WEAPON_FRAMES`` distinct sampled frames
within a ``MAX_FRAME_GAP_SEC`` window, AND the per-frame confidence
averages at least ``MIN_AVERAGE_WEAPON_CONFIDENCE``. This stops a
single spurious frame from flipping the whole clip to ``armed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from config import (
    DETECTION_DIR,
    FRAME_EXTRACTION_INTERVAL_SEC,
    KEYFRAMES_DIR,
    KEYFRAME_COUNT,
    MAX_FRAME_GAP_SEC,
    MIN_AVERAGE_WEAPON_CONFIDENCE,
    MIN_WEAPON_FRAMES,
    WEAPON_CANDIDATE_LABELS,
)
from models.schemas import DetectionResult, FrameResult, VideoAnalysisResult
from models.weapon_verifier import (
    apply_to_detections as _verify_detections,
)
from utils.frame_utils import cv2_to_pil


class VideoProcessor:
    """
    Extract frames from a video and run YOLO on each.

    The YOLO model is injected, so this class is testable with a stub
    detector that doesn't load weights.
    """

    def __init__(
        self,
        detector: Any,                        # duck-typed YOLODetector
        keyframe_count: int = KEYFRAME_COUNT,
        frame_interval_sec: float = FRAME_EXTRACTION_INTERVAL_SEC,
    ) -> None:
        self.detector = detector
        self.keyframe_count = max(1, keyframe_count)
        self.frame_interval_sec = max(0.1, float(frame_interval_sec))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        video_path: Path | str,
        source_name: str = "video",
        save_frames: bool = True,
    ) -> VideoAnalysisResult:
        """
        Analyze a video file.

        Steps:
        1. Open the video with OpenCV.
        2. Walk the video, sampling every `frame_interval_sec` seconds.
        3. For each sampled frame: run YOLO + save to disk.
        4. Sort frames by evidence_score and keep top-K keyframes.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_sec = (frame_count / fps) if fps > 0 else 0.0

            metadata: dict[str, Any] = {
                "fps":          round(fps, 2),
                "frame_count":  frame_count,
                "width":        width,
                "height":       height,
                "duration_sec": round(duration_sec, 2),
            }

            stem = path.stem
            frames_dir = DETECTION_DIR / "frames" / stem
            kf_dir = KEYFRAMES_DIR / stem
            if save_frames:
                frames_dir.mkdir(parents=True, exist_ok=True)
                kf_dir.mkdir(parents=True, exist_ok=True)

            frame_results: list[FrameResult] = []
            step = max(1, int(round(fps * self.frame_interval_sec)))
            idx = 0
            sample_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if idx % step == 0:
                    timestamp = (idx / fps) if fps > 0 else float(sample_idx)
                    fr = self._process_frame(
                        frame=frame,
                        index=sample_idx,
                        timestamp_sec=round(timestamp, 2),
                        frames_dir=frames_dir,
                        save=save_frames,
                    )
                    frame_results.append(fr)
                    sample_idx += 1

                idx += 1
        finally:
            cap.release()

        # ----- Pick top-K keyframes -------------------------------------
        ranked = sorted(
            range(len(frame_results)),
            key=lambda i: frame_results[i].evidence_score,
            reverse=True,
        )
        keyframe_indices = ranked[: self.keyframe_count]

        # Save keyframe copies (in addition to per-frame annotated ones)
        if save_frames:
            for rank, fr_idx in enumerate(keyframe_indices, start=1):
                fr = frame_results[fr_idx]
                if fr.annotated_path is None and fr.detection and fr.detection.annotated_image:
                    out = kf_dir / f"keyframe_{rank:02d}_t{fr.timestamp_sec:.1f}s.jpg"
                    fr.detection.annotated_image.save(out, format="JPEG", quality=90)
                    fr.annotated_path = out

        # ----- Phase 47 — temporal weapon verification ------------------
        # Walk the per-frame detections: a candidate weapon is promoted
        # to verified only when it persists across enough frames.
        promoted = _apply_temporal_weapon_verification(
            frame_results,
            min_frames=MIN_WEAPON_FRAMES,
            max_frame_gap_sec=MAX_FRAME_GAP_SEC,
            min_avg_confidence=MIN_AVERAGE_WEAPON_CONFIDENCE,
        )
        if promoted:
            metadata["temporal_weapon_promotions"] = promoted

        return VideoAnalysisResult(
            source_name=source_name,
            timestamp=datetime.now(),
            metadata=metadata,
            frame_results=frame_results,
            keyframe_indices=keyframe_indices,
            model_name=getattr(self.detector, "model_name", "unknown"),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process_frame(
        self,
        frame: Any,
        index: int,
        timestamp_sec: float,
        frames_dir: Path,
        save: bool,
    ) -> FrameResult:
        pil_img = cv2_to_pil(frame)
        detection: DetectionResult = self.detector.detect_image(
            image=pil_img,
            source_name=f"frame_{index:04d}_t{timestamp_sec:.1f}s",
        )

        image_path: Path | None = None
        annotated_path: Path | None = None

        if save:
            image_path = frames_dir / f"frame_{index:04d}_t{timestamp_sec:.1f}s.jpg"
            pil_img.save(image_path, format="JPEG", quality=85)
            if detection.annotated_image is not None:
                annotated_path = frames_dir / f"frame_{index:04d}_t{timestamp_sec:.1f}s_annotated.jpg"
                detection.annotated_image.save(annotated_path, format="JPEG", quality=90)

        return FrameResult(
            index=index,
            timestamp_sec=timestamp_sec,
            image_path=image_path,
            annotated_path=annotated_path,
            detection=detection,
        )


# ----------------------------------------------------------------------
# Phase 47 — temporal weapon verification
# ----------------------------------------------------------------------
def _apply_temporal_weapon_verification(
    frame_results: list[FrameResult],
    min_frames: int,
    max_frame_gap_sec: float,
    min_avg_confidence: float,
) -> dict[str, Any]:
    """
    Walk every frame and re-label weapon-class detections whose label
    appears persistently enough to be considered a real weapon.

    Rules:
      * Group detections by their ``label`` (knife / weapon).
      * For each group, find the longest run of frames whose
        consecutive timestamps differ by <= ``max_frame_gap_sec``.
      * If that run has at least ``min_frames`` hits AND the mean
        confidence across the run is >= ``min_avg_confidence``,
        promote those detections to ``verified`` (label="weapon").

    Returns a small dict describing what was promoted so the metadata
    block in the report can show it transparently.
    """
    if not frame_results:
        return {}

    # 1) Collect per-label hit lists: { label: [(ts, conf), ...] }
    hits: dict[str, list[tuple[float, float]]] = {}
    for fr in frame_results:
        if fr.detection is None:
            continue
        for d in fr.detection.detections:
            if d.label in WEAPON_CANDIDATE_LABELS:
                hits.setdefault(d.label, []).append((fr.timestamp_sec, d.confidence))

    promotions: dict[str, dict[str, Any]] = {}

    # 2) For each label, find longest qualifying run.
    for label, points in hits.items():
        if not points:
            continue
        points.sort(key=lambda t: t[0])
        run_start = 0
        best_run: tuple[int, int] = (0, 0)  # inclusive-exclusive
        for i in range(len(points)):
            # Window from points[run_start] to points[i]
            while (run_start < i
                   and points[i][0] - points[run_start][0] > max_frame_gap_sec):
                run_start += 1
            win_len = i - run_start + 1
            if win_len > best_run[1] - best_run[0]:
                best_run = (run_start, run_start + win_len)
        start, end = best_run
        length = end - start
        if length >= min_frames:
            avg_conf = sum(points[i][1] for i in range(start, end)) / length
            if avg_conf >= min_avg_confidence:
                promotions[label] = {
                    "frames": length,
                    "avg_confidence": round(avg_conf, 3),
                    "span_sec": round(points[end - 1][0] - points[start][0], 2),
                }
                _promote_frames(frame_results, label, points[start:end])

    return promotions


def _promote_frames(
    frame_results: list[FrameResult],
    label: str,
    points: list[tuple[float, float]],
) -> None:
    """
    For the supplied list of (timestamp, confidence) hits, walk the
    frames and re-label matching detections from ``candidate_weapon``
    to ``weapon`` and from ``"candidate"`` to ``"verified"``.

    Detection objects are immutable dataclasses, so we rebuild the
    DetectionResult with a new list.
    """
    ts_to_label = {round(ts, 2) for ts, _ in points}
    for fr in frame_results:
        if fr.detection is None:
            continue
        if round(fr.timestamp_sec, 2) not in ts_to_label:
            continue
        new_dets = []
        for d in fr.detection.detections:
            if d.label == label and (
                getattr(d, "weapon_status", "") in ("candidate", "")
                or d.label == label
            ):
                from dataclasses import replace
                new_dets.append(replace(d, label="weapon", weapon_status="verified"))
            else:
                new_dets.append(d)
        fr.detection = DetectionResult(
            source_name=fr.detection.source_name,
            timestamp=fr.detection.timestamp,
            detections=new_dets,
            raw_count=fr.detection.raw_count,
            annotated_image=fr.detection.annotated_image,
            model_name=fr.detection.model_name,
            source_tag=fr.detection.source_tag,
            models_used=list(fr.detection.models_used),
        )


# ----------------------------------------------------------------------
# Phase 51 — consolidated video events
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VideoEvent:
    """One coherent video event derived from per-frame detections.

    `start_sec` and `end_sec` are inclusive/exclusive. `frame_count`
    is the number of sampled frames in which the object appeared
    (de-duplicated from the raw per-frame detections).
    """

    label: str
    start_sec: float
    end_sec: float
    frame_count: int
    avg_confidence: float
    status: str            # "verified" | "candidate" | "non_weapon"


def consolidate_video_events(
    frame_results: list[FrameResult],
    max_frame_gap_sec: float = MAX_FRAME_GAP_SEC,
    min_frames: int = MIN_WEAPON_FRAMES,
    min_avg_confidence: float = MIN_AVERAGE_WEAPON_CONFIDENCE,
) -> list[VideoEvent]:
    """
    Walk every frame result and collapse per-frame detections into a
    small list of coherent `VideoEvent`s.

    An object that appears across N frames is reported as ONE event
    whose `[start_sec, end_sec)` spans the run. This is what the
    timeline, the report and the 3D scene should consume — never the
    raw per-frame list.

    Parameters
    ----------
    frame_results      : per-frame detections (sorted by timestamp).
    max_frame_gap_sec  : max gap between consecutive hits in the same
                         run. A larger gap splits a new event.
    min_frames         : minimum frame hits for the run to count as
                         an event at all (anything shorter is dropped).
    min_avg_confidence : mean confidence across the run must clear
                         this bar for the event to be verified.

    Returns
    -------
    list[VideoEvent], one entry per `(label, run)`.
    """
    if not frame_results:
        return []

    # Group (timestamp, confidence, status) by label.
    by_label: dict[str, list[tuple[float, float, str]]] = {}
    for fr in frame_results:
        if fr.detection is None:
            continue
        for d in fr.detection.detections:
            by_label.setdefault(d.label, []).append(
                (fr.timestamp_sec, d.confidence, getattr(d, "weapon_status", "") or "")
            )

    events: list[VideoEvent] = []
    for label, points in by_label.items():
        if not points:
            continue
        points.sort(key=lambda t: t[0])
        # Find runs: contiguous windows where consecutive timestamps
        # differ by no more than max_frame_gap_sec.
        runs: list[list[tuple[float, float, str]]] = [[points[0]]]
        for prev, cur in zip(points, points[1:]):
            if cur[0] - prev[0] <= max_frame_gap_sec:
                runs[-1].append(cur)
            else:
                runs.append([cur])
        for run in runs:
            if len(run) < min_frames:
                continue
            avg_conf = sum(p[1] for p in run) / len(run)
            status = "verified" if avg_conf >= min_avg_confidence else "candidate"
            events.append(
                VideoEvent(
                    label=label,
                    start_sec=round(run[0][0], 2),
                    end_sec=round(run[-1][0], 2),
                    frame_count=len(run),
                    avg_confidence=round(avg_conf, 3),
                    status=status,
                )
            )
    events.sort(key=lambda e: (e.start_sec, e.label))
    return events

