"""
Shared Streamlit session helpers for pages.

Centralizes things that more than one page needs so we don't end
up with two cached copies of the same YOLO model, or two
inconsistent session-state keys.

Currently provides:
- `SessionKeys`: centralized constants for every session-state key the
  app uses. Pages should reference these instead of stringly-typing the
  keys so a typo in one page can't silently desync from a reader in
  another page (Phase 50).
- `get_yolo_detector()`: the single source of truth for the
  cached YOLODetector instance across all pages.
- `resolve_active_analysis()`, `resolve_batch_or_detection_image()`,
  `resolve_batch_or_video()`, `resolve_batch_or_detection()` —
  analysis-resolution helpers that prefer batch results over the
  single-file session keys.
"""

from __future__ import annotations

import streamlit as st

from models.schemas import EvidenceAnalysis
from models.yolo_detector import MultiSourceDetector, get_multi_source_detector


class SessionKeys:
    """Central registry of session-state keys used by the app.

    Pages should import this class and reference ``SessionKeys.LAST_ANALYSIS``
    instead of writing the bare string ``"last_analysis"`` in two
    different places. This keeps reads and writes in sync.
    """

    # ----- Theme / UI ------------------------------------------------
    UI_THEME = "ui_theme"

    # ----- Model singletons ------------------------------------------
    YOLO_DETECTOR = "yolo_detector_singleton"

    # ----- Latest evidence -------------------------------------------
    LAST_DETECTION       = "last_detection"       # single-file image
    LAST_VIDEO           = "last_video"           # single-file video
    LAST_ANALYSIS        = "last_analysis"        # single-file analysis
    LAST_BATCH_RESULT    = "last_batch_result"    # multi-file analysis

    # ----- Storyboard / reconstruction ------------------------------
    LAST_STORYBOARD               = "last_storyboard"
    LAST_STORYBOARD_IMAGE_PATHS   = "last_storyboard_image_paths"
    LAST_RECONSTRUCTION_VIDEO_PATH = "last_reconstruction_video_path"

    # ----- AI summary / chat ----------------------------------------
    LAST_SUMMARY = "last_summary"
    CHAT_HISTORY = "chat_history"


_YOLO_KEY_NAME = SessionKeys.YOLO_DETECTOR


def get_yolo_detector() -> MultiSourceDetector:
    """
    Return the singleton MultiSourceDetector cached for this Streamlit session.

    Combines general YOLOv8 detection with dedicated weapon detection.
    """
    existing = st.session_state.get(SessionKeys.YOLO_DETECTOR)
    if existing is not None:
        return existing

    with st.spinner("Loading Multi-Source YOLOv8 models (general + weapon)..."):
        detector = get_multi_source_detector()
    st.session_state[SessionKeys.YOLO_DETECTOR] = detector
    return detector


def reset_yolo_detector() -> None:
    """Drop the cached detector. Useful for tests and dev tools."""
    st.session_state.pop(SessionKeys.YOLO_DETECTOR, None)


# ----------------------------------------------------------------------
# Analysis resolution helper (used by every downstream page)
# ----------------------------------------------------------------------
def resolve_active_analysis() -> EvidenceAnalysis | None:
    """
    Return the most relevant EvidenceAnalysis for the current page render.

    Resolution order (first hit wins):
    1. The combined analysis produced by the multi-file Crime Scene
       Investigation page (`SessionKeys.LAST_BATCH_RESULT`
       .combined_analysis). This is fresher than any single-file
       analysis and reflects every successful file in the batch.
    2. The single-file analysis (`SessionKeys.LAST_ANALYSIS`)
       populated by the Evidence Analysis page after running
       Image Detection / Video Processing.

    Returns None if neither is set, in which case the caller should
    render an "evidence not generated yet" warning.
    """
    batch = st.session_state.get(SessionKeys.LAST_BATCH_RESULT)
    if batch is not None:
        combined = getattr(batch, "combined_analysis", None)
        if combined is not None:
            return combined
    return st.session_state.get(SessionKeys.LAST_ANALYSIS)


def resolve_batch_or_detection_image():
    """
    Return the best available annotated image for reconstruction /
    storyboard purposes.

    Resolution order:
    1. The first keyframe's annotated image from the latest video
       batch result, if any.
    2. The first file's annotated image from the latest batch result,
       if any.
    3. The annotated image from `last_detection` (single-file path).
    4. None — caller falls back to a clean placeholder panel.
    """
    batch = st.session_state.get(SessionKeys.LAST_BATCH_RESULT)
    if batch is not None:
        for f in batch.files:
            if getattr(f, "video", None) is not None and f.video.keyframes:
                for kf in f.video.keyframes:
                    if (
                        kf.detection is not None
                        and kf.detection.annotated_image is not None
                    ):
                        return kf.detection.annotated_image
            if getattr(f, "detection", None) is not None and f.detection.annotated_image is not None:
                return f.detection.annotated_image
    last_detection = st.session_state.get(SessionKeys.LAST_DETECTION)
    if last_detection is not None and last_detection.annotated_image is not None:
        return last_detection.annotated_image
    return None


def resolve_batch_or_video() -> "object | None":
    """
    Return the latest video analysis result, preferring the batch
    result's video entry over `last_video`. Used by pages that need
    per-frame details (e.g. Crime Timeline).
    """
    batch = st.session_state.get(SessionKeys.LAST_BATCH_RESULT)
    if batch is not None:
        for f in batch.files:
            if getattr(f, "video", None) is not None:
                return f.video
    return st.session_state.get(SessionKeys.LAST_VIDEO)


def resolve_batch_or_detection() -> "object | None":
    """
    Return the latest image-detection result, preferring the batch
    result's first image entry over `last_detection`.
    """
    batch = st.session_state.get(SessionKeys.LAST_BATCH_RESULT)
    if batch is not None:
        for f in batch.files:
            if getattr(f, "detection", None) is not None:
                return f.detection
    return st.session_state.get(SessionKeys.LAST_DETECTION)

