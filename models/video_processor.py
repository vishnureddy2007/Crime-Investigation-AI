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
    WEAPON_VERIFY_THRESHOLD,
    get_weights_sha256,
)
from models.schemas import DetectionResult, FrameResult, VideoAnalysisResult
from models.weapon_verifier import (
    apply_to_detections as _verify_detections,
)
from utils.frame_utils import cv2_to_pil


def _bbox_iou(a: Any, b: Any) -> float:
    """Intersection over union for two detection boxes."""
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


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

        m_name = getattr(self.detector, "model_name", "unknown")
        weights_map = {}
        if hasattr(self.detector, "general"):
            gen_name = getattr(self.detector.general, "model_name", "yolov8n.pt")
            weights_map[gen_name] = get_weights_sha256(gen_name)
        elif m_name != "unknown":
            weights_map[m_name] = get_weights_sha256(m_name)

        config_meta = {
            "frame_interval_sec": self.frame_interval_sec,
            "keyframe_count": self.keyframe_count,
        }

        return VideoAnalysisResult(
            source_name=source_name,
            timestamp=datetime.now(),
            metadata=metadata,
            frame_results=frame_results,
            keyframe_indices=keyframe_indices,
            model_name=m_name,
            model_version="video-yolov8n",
            model_weights=weights_map,
            config_metadata=config_meta,
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

    # 1) Build spatial tracks per label so separate objects with the same
    # label cannot verify each other merely by appearing in the same clip.
    # At most one observation per physical object is counted per sampled frame.
    hits: dict[str, list[list[tuple[float, float, Any, str]]]] = {}
    for fr in frame_results:
        if fr.detection is None:
            continue
        frame_weapons: list[Any] = [
            d for d in fr.detection.detections if d.label in WEAPON_CANDIDATE_LABELS
        ]
        # Sort descending by confidence so the best box matches first
        frame_weapons.sort(key=lambda item: item.confidence, reverse=True)

        for d in frame_weapons:
            tracks = hits.setdefault(d.label, [])
            best_track, best_iou = None, 0.0
            for idx, track in enumerate(tracks):
                last_ts, _, last_bbox, _ = track[-1]
                # A track can receive at most one box from this frame
                if last_ts == fr.timestamp_sec:
                    continue
                if fr.timestamp_sec - last_ts > max_frame_gap_sec:
                    continue
                overlap = _bbox_iou(d.bbox, last_bbox)
                if overlap > best_iou:
                    best_track, best_iou = idx, overlap

            status_str = getattr(d, "weapon_status", "") or ""
            if best_track is not None and best_iou >= 0.30:
                tracks[best_track].append((fr.timestamp_sec, d.confidence, d.bbox, status_str))
            else:
                tracks.append([(fr.timestamp_sec, d.confidence, d.bbox, status_str)])

    promotions: dict[str, dict[str, Any]] = {}

    # 2) For each label, find longest qualifying run.
    for label, tracks in hits.items():
        for points in tracks:
            if not points:
                continue
            run_start = 0
            best_run: tuple[int, int] = (0, 0)  # inclusive-exclusive
            for i in range(len(points)):
                while (run_start < i and points[i][0] - points[run_start][0] > max_frame_gap_sec):
                    run_start += 1
                win_len = i - run_start + 1
                if win_len > best_run[1] - best_run[0]:
                    best_run = (run_start, run_start + win_len)
            start, end = best_run
            length = end - start
            if length >= min_frames:
                avg_conf = sum(points[i][1] for i in range(start, end)) / length
                run_statuses = {points[i][3] for i in range(start, end)}

                # Requirement 4: Never verify a weapon from temporal persistence alone.
                # Must have frame-level visual evidence (conf >= WEAPON_VERIFY_THRESHOLD or status == "verified")
                # and must not contain rejected detections.
                has_frame_evidence = any(
                    points[i][1] >= WEAPON_VERIFY_THRESHOLD or points[i][3] == "verified"
                    for i in range(start, end)
                )
                is_rejected = any(points[i][3] == "rejected" for i in range(start, end))

                if avg_conf >= min_avg_confidence and has_frame_evidence and not is_rejected:
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
    points: list[tuple[float, float, Any, str]],
) -> None:
    """
    For the supplied list of (timestamp, confidence, bbox, status) hits,
    walk the frames and re-label matching detections from candidate to verified.
    """
    targets = [(round(p[0], 2), p[2]) for p in points]
    for fr in frame_results:
        if fr.detection is None:
            continue
        new_dets = []
        for d in fr.detection.detections:
            matching_target = next(
                (bbox for ts, bbox in targets
                 if ts == round(fr.timestamp_sec, 2)
                 and _bbox_iou(d.bbox, bbox) >= 0.30),
                None,
            )
            if d.label == label and matching_target is not None and (
                getattr(d, "weapon_status", "") in ("candidate", "")
                or d.label == label
            ):
                from dataclasses import replace
                new_dets.append(replace(d, label="weapon", weapon_status="verified"))
            else:
                new_dets.append(d)

        orig_d = fr.detection
        verified_w = [d for d in new_dets if getattr(d, "weapon_status", "") == "verified"]
        candidates_w = [d for d in new_dets if getattr(d, "weapon_status", "") == "candidate"]
        rejected_w = [d for d in new_dets if getattr(d, "weapon_status", "") == "rejected"]

        fr.detection = DetectionResult(
            source_name=orig_d.source_name,
            timestamp=orig_d.timestamp,
            detections=new_dets,
            raw_count=orig_d.raw_count,
            annotated_image=orig_d.annotated_image,
            model_name=orig_d.model_name,
            source_tag=orig_d.source_tag,
            models_used=list(orig_d.models_used),
            raw_detections=list(orig_d.raw_detections or new_dets),
            weapon_candidates=candidates_w,
            verified_weapons=verified_w,
            rejected_detections=rejected_w,
            uncertain_detections=candidates_w,
            model_version=orig_d.model_version,
            model_weights=dict(orig_d.model_weights),
            config_metadata=dict(orig_d.config_metadata),
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

    # Build short-lived tracks per label. Label-only grouping incorrectly
    # combined separate objects and counted duplicate boxes in a frame as
    # separate observations. Greedy IoU matching keeps one observation per
    # object per sampled frame without introducing an external tracker.
    by_label: dict[str, list[list[tuple[float, float, str, Any]]]] = {}
    for fr in frame_results:
        if fr.detection is None:
            continue
        frame_labels: dict[str, list[Any]] = {}
        for d in fr.detection.detections:
            frame_labels.setdefault(d.label, []).append(d)
        for label, detections in frame_labels.items():
            tracks = by_label.setdefault(label, [])
            # A track can receive at most one box from this frame.
            available = set(range(len(tracks)))
            for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
                best_track, best_iou = None, 0.0
                for track_idx in available:
                    last_ts, _, _, last_detection = tracks[track_idx][-1]
                    if fr.timestamp_sec - last_ts > max_frame_gap_sec:
                        continue
                    overlap = _bbox_iou(detection.bbox, last_detection.bbox)
                    if overlap > best_iou:
                        best_track, best_iou = track_idx, overlap
                point = (fr.timestamp_sec, detection.confidence,
                         getattr(detection, "weapon_status", "") or "", detection)
                if best_track is not None and best_iou >= 0.30:
                    tracks[best_track].append(point)
                    available.remove(best_track)
                else:
                    tracks.append([point])

    events: list[VideoEvent] = []
    for label, tracks in by_label.items():
        for run in tracks:
            if len(run) < min_frames:
                continue
            avg_conf = sum(p[1] for p in run) / len(run)
            statuses = {p[2] for p in run}
            # Only call weapon events verified if supported by frame-level evidence or verified detections.
            # Requirement 4: Persistence alone is never proof of a weapon.
            is_weapon = label.lower() in {name.lower() for name in WEAPON_CANDIDATE_LABELS}
            if is_weapon:
                has_frame_evidence = any(
                    p[1] >= WEAPON_VERIFY_THRESHOLD or p[2] == "verified"
                    for p in run
                )
                is_rejected = any(p[2] == "rejected" for p in run)
                status = (
                    "verified"
                    if (avg_conf >= min_avg_confidence and has_frame_evidence and not is_rejected)
                    else "candidate"
                )
            else:
                status = "verified" if avg_conf >= min_avg_confidence else "candidate"
            events.append(
                VideoEvent(
                    label=label,
                    start_sec=round(min(p[0] for p in run), 2),
                    end_sec=round(max(p[0] for p in run), 2),
                    frame_count=len(run),
                    avg_confidence=round(avg_conf, 3),
                    status=status,
                )
            )
    events.sort(key=lambda e: (e.start_sec, e.label))
    return events

