"""
Batch evidence processor.

Handles a list of mixed image / video uploads and runs the right
pipeline on each one. The result is a `BatchEvidenceResult` that
contains:

- one `FileEvidence` per file (with its own DetectionResult /
  VideoAnalysisResult, error if any, and per-file metrics)
- a combined `EvidenceAnalysis` for the whole batch (consumed by
  the report and reconstruction layers)

Design rules:
- One failed file MUST NOT crash the rest of the batch.
- Never invent detections. If a file fails, the failure is recorded
  and the rest of the batch still proceeds.
- Videos are decoded frame-by-frame via the existing VideoProcessor.
- The processor is plain Python (no Streamlit) so it can be tested
  with stub detectors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from models.analysis_engine import build_combined_analysis
from models.schemas import (
    AnalysisInput,
    DetectionResult,
    EvidenceAnalysis,
    VideoAnalysisResult,
)
from models.video_processor import VideoProcessor
from utils.image_io import ImageLoadError, load_image_from_bytes
from utils.video_io import VideoLoadError, validate_video_bytes


# ----------------------------------------------------------------------
# Supported file types
# ----------------------------------------------------------------------
IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def classify_file(filename: str) -> str:
    """Return 'image', 'video', or 'unsupported' for a filename."""
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unsupported"


# ----------------------------------------------------------------------
# File-level result
# ----------------------------------------------------------------------
@dataclass
class FileEvidence:
    """The result of processing a single uploaded file."""

    filename: str
    source_type: str                       # "image" | "video" | "unsupported"
    file_size_bytes: int
    detection: DetectionResult | None = None
    video: VideoAnalysisResult | None = None
    error: dict[str, str] | None = None    # {"message": str, "reason": str}
    metrics: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def succeeded(self) -> bool:
        return (self.detection is not None or self.video is not None) and self.error is None

    @property
    def person_count(self) -> int:
        return self.metrics.get("person_count", 0)

    @property
    def weapon_count(self) -> int:
        return self.metrics.get("weapon_count", 0)

    @property
    def vehicle_count(self) -> int:
        return self.metrics.get("vehicle_count", 0)

    @property
    def bag_count(self) -> int:
        return self.metrics.get("bag_count", 0)

    @property
    def detection_count(self) -> int:
        if self.detection is not None:
            return self.detection.count
        if self.video is not None:
            return self.video.total_detections()
        return 0


# ----------------------------------------------------------------------
# Aggregate batch result
# ----------------------------------------------------------------------
@dataclass
class BatchEvidenceResult:
    """The aggregated result of processing a batch of files."""

    files: list[FileEvidence] = field(default_factory=list)
    combined_analysis: EvidenceAnalysis | None = None
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def succeeded_files(self) -> int:
        return sum(1 for f in self.files if f.succeeded)

    @property
    def failed_files(self) -> int:
        return sum(1 for f in self.files if not f.succeeded)

    def totals(self) -> dict[str, int]:
        """Aggregate counts across every successful file."""
        out = {
            "person_count": 0,
            "weapon_count": 0,
            "vehicle_count": 0,
            "bag_count": 0,
            "bottle_count": 0,
            "knife_count": 0,
            "gun_count": 0,
            "detection_count": 0,
            "frame_count": 0,
            "image_count": 0,
            "video_count": 0,
        }
        for f in self.files:
            if not f.succeeded:
                continue
            for k in (
                "person_count", "weapon_count", "vehicle_count",
                "bag_count", "bottle_count", "knife_count", "gun_count",
            ):
                out[k] += f.metrics.get(k, 0)
            out["detection_count"] += f.detection_count
            if f.source_type == "video" and f.video is not None:
                out["frame_count"] += f.video.frame_count
                out["video_count"] += 1
            elif f.source_type == "image":
                out["image_count"] += 1
        return out

    def per_file_summary(self) -> list[dict[str, Any]]:
        """Return one row per file, ready for st.dataframe."""
        rows: list[dict[str, Any]] = []
        for f in self.files:
            rows.append({
                "filename":      f.filename,
                "source_type":   f.source_type,
                "status":        "OK" if f.succeeded else "FAILED",
                "error":         (f.error or {}).get("message", ""),
                "detections":    f.detection_count,
                "persons":       f.person_count,
                "weapons":       f.weapon_count,
                "vehicles":      f.vehicle_count,
                "bags":          f.bag_count,
                "frames":        (f.video.frame_count if f.video else 0),
                "models_used":   ", ".join(f.detection.models_used) if f.detection else "",
                "model_name":    f.detection.model_name if f.detection else "",
            })
        return rows


# ----------------------------------------------------------------------
# Processor
# ----------------------------------------------------------------------
SaveVideoFn = Callable[[bytes, str], Path]


class BatchEvidenceProcessor:
    """
    Run multi-source detection on every uploaded file.

    Parameters
    ----------
    detector:
        Any object exposing `detect_image(image, source_name, confidence, iou)`.
        `MultiSourceDetector` is the canonical choice.
    video_processor_factory:
        Callable that returns a `VideoProcessor` (or a duck-typed
        stand-in) given a detector. Defaults to `VideoProcessor`.
    save_video:
        Optional callback to save uploaded video bytes to disk.
        If absent, videos are decoded from the raw bytes by saving
        to a temp directory under `MODELS_DIR / "_tmp" / "<safe_name>"`.
    """

    def __init__(
        self,
        detector: Any,
        video_processor_factory: Callable[[Any], Any] | None = None,
        save_video: SaveVideoFn | None = None,
    ) -> None:
        self.detector = detector
        self.video_processor_factory = video_processor_factory or (
            lambda d: VideoProcessor(detector=d)
        )
        self.save_video = save_video

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_batch(self, *args, **kwargs) -> BatchEvidenceResult:
        """Backwards-compatible alias for ``process``.

        Older callers (and any cached Streamlit bytecode) may still
        invoke the legacy ``process_batch`` name. Forward to ``process``
        so the public surface stays stable across renames.
        """
        return self.process(*args, **kwargs)

    def process(
        self,
        files: list[tuple[str, bytes]],
        confidence: float | None = None,
        iou: float | None = None,
    ) -> BatchEvidenceResult:
        """
        Process every (filename, bytes) pair in `files`.

        Order is preserved: the i-th output `FileEvidence` corresponds
        to the i-th input. Files are processed sequentially to avoid
        hammering the GPU on shared weights.
        """
        batch = BatchEvidenceResult(files=[])
        batch.started_at = datetime.now()

        # First pass: walk every file, fail-soft.
        for filename, data in files:
            kind = classify_file(filename)
            entry = FileEvidence(
                filename=filename,
                source_type=kind,
                file_size_bytes=len(data) if data else 0,
            )
            if kind == "unsupported":
                entry.error = {
                    "message": f"Unsupported file type: {filename}",
                    "reason":  "extension",
                }
                batch.files.append(entry)
                continue
            if not data:
                entry.error = {
                    "message": f"Empty file: {filename}",
                    "reason":  "empty",
                }
                batch.files.append(entry)
                continue

            try:
                if kind == "image":
                    self._process_image(entry, data, confidence, iou)
                else:
                    self._process_video(entry, data, confidence, iou)
            except (ImageLoadError, VideoLoadError, RuntimeError, OSError, ValueError) as exc:
                entry.error = {
                    "message": f"Could not process {filename}: {exc}",
                    "reason":  exc.__class__.__name__,
                }
            except Exception as exc:  # pragma: no cover - defensive
                entry.error = {
                    "message": f"Unexpected error: {exc}",
                    "reason":  exc.__class__.__name__,
                }

            batch.files.append(entry)

        # Second pass: aggregate per-file analyses into one combined
        # analysis. This is what the report and reconstruction layers
        # consume. Done in a separate pass so a single failure doesn't
        # block the file-level results.
        try:
            batch.combined_analysis = self._build_combined_analysis(batch)
        except Exception:  # pragma: no cover - defensive
            batch.combined_analysis = None

        batch.finished_at = datetime.now()
        return batch

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process_image(
        self,
        entry: FileEvidence,
        data: bytes,
        confidence: float | None,
        iou: float | None,
    ) -> None:
        image = load_image_from_bytes(data, source_name=entry.filename)
        detection = self.detector.detect_image(
            image=image,
            source_name=entry.filename,
            confidence=confidence,
            iou=iou,
        )
        entry.detection = detection
        entry.metrics = _per_file_metrics(detection)

    def _process_video(
        self,
        entry: FileEvidence,
        data: bytes,
        confidence: float | None,
        iou: float | None,
    ) -> None:
        validate_video_bytes(data, filename=entry.filename)
        # Save to a temp path the VideoProcessor can open.
        from config import VIDEOS_DIR
        save_dir = Path(VIDEOS_DIR) / "_tmp"
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_name(entry.filename)
        target = save_dir / f"{int(time.time() * 1000)}_{safe_name}"
        target.write_bytes(data)
        try:
            if self.save_video is not None:
                saved_path = self.save_video(data, entry.filename)
            else:
                saved_path = target
            # Set the per-run confidence on the underlying detector
            # if it's the shared multi-source detector.
            original_conf = getattr(self.detector.general, "confidence", None)
            if original_conf is not None and confidence is not None:
                self.detector.general.confidence = float(confidence)
            try:
                processor = self.video_processor_factory(self.detector)
                result = processor.process(
                    video_path=saved_path,
                    source_name=entry.filename,
                    save_frames=False,
                )
            finally:
                if original_conf is not None and confidence is not None:
                    self.detector.general.confidence = original_conf
            entry.video = result
            entry.metrics = _per_file_video_metrics(result)
        finally:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass

    def _build_combined_analysis(
        self,
        batch: BatchEvidenceResult,
    ) -> EvidenceAnalysis | None:
        """Build a combined EvidenceAnalysis from the batch's
        per-file inputs. Returns None if there are no successful
        inputs."""
        inputs: list[AnalysisInput] = []
        for f in batch.files:
            if not f.succeeded:
                continue
            if f.detection is not None:
                inputs.append(AnalysisInput.from_image(f.detection))
            elif f.video is not None:
                inputs.append(AnalysisInput.from_video(f.video))
        if not inputs:
            return None
        return build_combined_analysis(inputs, total_files=batch.total_files)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """Sanitise a filename (avoid path traversal)."""
    from core.security import safe_filename
    return safe_filename(name, max_length=80)


def _per_file_metrics(detection: DetectionResult) -> dict[str, int]:
    """Per-file counts for the batch summary table.

    Phase 46: bottle is no longer a weapon. `weapon_count` reflects
    verified weapons only (knife, weapon, gun, pistol, rifle, handgun,
    firearm) — anything labelled ``candidate_weapon`` is surfaced
    separately as ``candidate_weapon_count`` for transparency.
    """
    counts = detection.counts_by_label()
    weapon_count = 0
    candidate_weapon_count = 0
    bottle_count = 0
    knife_count = 0
    gun_count = 0
    for d in detection.detections:
        lbl = d.label
        status = getattr(d, "weapon_status", "")
        if lbl == "knife":
            knife_count += 1
            weapon_count += 1
        elif lbl == "candidate_weapon":
            candidate_weapon_count += 1
        elif lbl == "bottle":
            bottle_count += 1
        elif lbl in ("weapon", "gun", "pistol", "rifle", "handgun", "firearm"):
            weapon_count += 1
            if lbl in ("gun", "pistol", "rifle", "handgun", "firearm"):
                gun_count += 1
    return {
        "person_count":  counts.get("person", 0),
        "weapon_count":  weapon_count,
        "candidate_weapon_count": candidate_weapon_count,
        "vehicle_count": counts.get("vehicle", 0),
        "bag_count":     counts.get("bag", 0),
        "bottle_count":  bottle_count,
        "knife_count":   knife_count,
        "gun_count":     gun_count,
        "detection_count": detection.count,
    }


def _per_file_video_metrics(video: VideoAnalysisResult) -> dict[str, int]:
    """Per-file counts for a video result (Phase 46 — same rules)."""
    counts = video.aggregate_counts_by_label()
    weapon_count = 0
    candidate_weapon_count = 0
    bottle_count = 0
    knife_count = 0
    gun_count = 0
    for f in video.frame_results:
        if f.detection is None:
            continue
        for d in f.detection.detections:
            lbl = d.label
            if lbl == "knife":
                knife_count += 1
                weapon_count += 1
            elif lbl == "candidate_weapon":
                candidate_weapon_count += 1
            elif lbl == "bottle":
                bottle_count += 1
            elif lbl in ("weapon", "gun", "pistol", "rifle", "handgun", "firearm"):
                weapon_count += 1
                if lbl in ("gun", "pistol", "rifle", "handgun", "firearm"):
                    gun_count += 1
    return {
        "person_count":  counts.get("person", 0),
        "weapon_count":  weapon_count,
        "candidate_weapon_count": candidate_weapon_count,
        "vehicle_count": counts.get("vehicle", 0),
        "bag_count":     counts.get("bag", 0),
        "bottle_count":  bottle_count,
        "knife_count":   knife_count,
        "gun_count":     gun_count,
        "detection_count": video.total_detections(),
        "frame_count":   video.frame_count,
    }
