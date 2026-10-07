"""
Regression tests for `pages/_state.py` — the analysis-resolution helpers
that let downstream pages (Dashboard, AI Summary, Reconstruction, ...)
honour the multi-file Crime Scene Investigation batch result as well
as the single-file Evidence Analysis path.

These tests don't need Streamlit's runtime; we stub `streamlit` with
a minimal object that mimics the parts of `session_state` that
`_state.py` reads.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ----------------------------------------------------------------------
# Streamlit stub — only what `_state.py` touches.
# ----------------------------------------------------------------------
class _FakeSessionState(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def _install_streamlit_stub() -> None:
    if "streamlit" in sys.modules:
        return
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.session_state = _FakeSessionState()
    streamlit_stub.spinner = lambda *_a, **_kw: _NullContext()
    sys.modules["streamlit"] = streamlit_stub


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None


_install_streamlit_stub()


# ----------------------------------------------------------------------
# Lightweight stand-ins for the schema objects _state.py touches.
# ----------------------------------------------------------------------
@dataclass
class _FakeAnalysis:
    source_name: str = "x.png"
    source_type: str = "image"


@dataclass
class _FakeDetection:
    annotated_image: Any = "annotated-pil"


@dataclass
class _FakeVideoResult:
    keyframes: list = field(default_factory=list)
    annotated_image: Any = "annotated-pil"


@dataclass
class _FakeVideoKeyframe:
    detection: _FakeDetection | None = None


@dataclass
class _FakeFileEvidence:
    filename: str = "x.png"
    source_type: str = "image"
    detection: _FakeDetection | None = None
    video: _FakeVideoResult | None = None
    error: dict | None = None


@dataclass
class _FakeBatchResult:
    files: list = field(default_factory=list)
    combined_analysis: _FakeAnalysis | None = None


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def _reset_state() -> None:
    sys.modules["streamlit"].session_state = _FakeSessionState()


def test_resolve_active_analysis_returns_batch_combined_first() -> None:
    """When a batch result is present, its combined analysis wins."""
    _reset_state()
    batch_analysis = _FakeAnalysis(source_name="batch")
    single_analysis = _FakeAnalysis(source_name="single")
    sys.modules["streamlit"].session_state["last_batch_result"] = _FakeBatchResult(
        combined_analysis=batch_analysis
    )
    sys.modules["streamlit"].session_state["last_analysis"] = single_analysis

    from pages._state import resolve_active_analysis

    result = resolve_active_analysis()
    assert result is batch_analysis
    assert result.source_name == "batch"


def test_resolve_active_analysis_falls_back_to_single_when_no_batch() -> None:
    """Without a batch result, return the single-file analysis."""
    _reset_state()
    single = _FakeAnalysis(source_name="single")
    sys.modules["streamlit"].session_state["last_analysis"] = single

    from pages._state import resolve_active_analysis

    assert resolve_active_analysis() is single


def test_resolve_active_analysis_returns_none_when_nothing_set() -> None:
    """Empty session state → None."""
    _reset_state()
    from pages._state import resolve_active_analysis
    assert resolve_active_analysis() is None


def test_resolve_active_analysis_falls_back_when_batch_has_no_combined() -> None:
    """A batch result with `combined_analysis=None` must NOT mask the
    single-file analysis."""
    _reset_state()
    single = _FakeAnalysis(source_name="single")
    sys.modules["streamlit"].session_state["last_batch_result"] = _FakeBatchResult(
        combined_analysis=None
    )
    sys.modules["streamlit"].session_state["last_analysis"] = single

    from pages._state import resolve_active_analysis

    assert resolve_active_analysis() is single


def test_resolve_batch_or_detection_image_prefers_video_keyframe() -> None:
    """Annotated keyframes from a batch video result win over
    single-file `last_detection`."""
    _reset_state()
    kf_image = "kf-pil"
    detection_image = "single-pil"

    kf = _FakeVideoKeyframe(detection=_FakeDetection(annotated_image=kf_image))
    video = _FakeVideoResult(keyframes=[kf])
    f = _FakeFileEvidence(filename="vid.mp4", source_type="video", video=video)
    batch = _FakeBatchResult(files=[f])
    sys.modules["streamlit"].session_state["last_batch_result"] = batch
    sys.modules["streamlit"].session_state["last_detection"] = _FakeDetection(
        annotated_image=detection_image
    )

    from pages._state import resolve_batch_or_detection_image

    assert resolve_batch_or_detection_image() == kf_image


def test_resolve_batch_or_detection_image_falls_back_to_last_detection() -> None:
    """No batch image available → fall back to `last_detection`."""
    _reset_state()
    detection_image = "single-pil"
    sys.modules["streamlit"].session_state["last_detection"] = _FakeDetection(
        annotated_image=detection_image
    )

    from pages._state import resolve_batch_or_detection_image

    assert resolve_batch_or_detection_image() == detection_image


def test_resolve_batch_or_detection_image_returns_none_when_empty() -> None:
    """Nothing in session → None (caller falls back to placeholder)."""
    _reset_state()
    from pages._state import resolve_batch_or_detection_image
    assert resolve_batch_or_detection_image() is None


def test_resolve_batch_or_video_prefers_batch_video() -> None:
    """The first batch file's video result wins over `last_video`."""
    _reset_state()
    batch_video = _FakeVideoResult()
    f = _FakeFileEvidence(filename="vid.mp4", source_type="video", video=batch_video)
    batch = _FakeBatchResult(files=[f])
    single_video = _FakeVideoResult()

    sys.modules["streamlit"].session_state["last_batch_result"] = batch
    sys.modules["streamlit"].session_state["last_video"] = single_video

    from pages._state import resolve_batch_or_video

    assert resolve_batch_or_video() is batch_video


def test_resolve_batch_or_detection_prefers_batch_image() -> None:
    """The first batch file's detection wins over `last_detection`."""
    _reset_state()
    batch_detection = _FakeDetection(annotated_image="batch-pil")
    f = _FakeFileEvidence(
        filename="img.png", source_type="image", detection=batch_detection
    )
    batch = _FakeBatchResult(files=[f])
    single_detection = _FakeDetection(annotated_image="single-pil")

    sys.modules["streamlit"].session_state["last_batch_result"] = batch
    sys.modules["streamlit"].session_state["last_detection"] = single_detection

    from pages._state import resolve_batch_or_detection

    assert resolve_batch_or_detection() is batch_detection


def test_resolve_helpers_skip_failed_batch_files() -> None:
    """Failed batch entries (error != None) must NOT contribute
    detections or keyframes."""
    _reset_state()
    failed = _FakeFileEvidence(
        filename="bad.png", source_type="image", detection=None, error={"message": "x"}
    )
    good_detection = _FakeDetection(annotated_image="good-pil")
    good = _FakeFileEvidence(
        filename="good.png", source_type="image", detection=good_detection
    )
    batch = _FakeBatchResult(files=[failed, good])
    sys.modules["streamlit"].session_state["last_batch_result"] = batch

    from pages._state import resolve_batch_or_detection

    assert resolve_batch_or_detection() is good_detection


def test_resolve_helpers_skip_unsupported_batch_files() -> None:
    """Unsupported / empty entries (no detection, no video) must NOT
    contribute to the resolved image."""
    _reset_state()
    unsupported = _FakeFileEvidence(
        filename="notes.txt", source_type="unsupported", detection=None
    )
    good_detection = _FakeDetection(annotated_image="good-pil")
    good = _FakeFileEvidence(
        filename="good.png", source_type="image", detection=good_detection
    )
    batch = _FakeBatchResult(files=[unsupported, good])
    sys.modules["streamlit"].session_state["last_batch_result"] = batch

    from pages._state import resolve_batch_or_detection

    assert resolve_batch_or_detection() is good_detection


# ----------------------------------------------------------------------
# get_yolo_detector / reset_yolo_detector
# ----------------------------------------------------------------------
def test_get_yolo_detector_caches_instance() -> None:
    """Calling get_yolo_detector twice returns the same instance
    and only loads the model once (the spinner runs at most once)."""
    _reset_state()
    from unittest.mock import patch, MagicMock

    from pages._state import get_yolo_detector, reset_yolo_detector

    fake_detector = MagicMock(name="fake-multi-source-detector")
    with patch(
        "pages._state.get_multi_source_detector",
        return_value=fake_detector,
    ) as factory:
        d1 = get_yolo_detector()
        d2 = get_yolo_detector()
    # Same instance on second call.
    assert d1 is d2
    assert d1 is fake_detector
    # The factory should have been hit exactly once (cache hit on 2nd call).
    assert factory.call_count == 1

    # Clean up for the next test.
    reset_yolo_detector()


def test_reset_yolo_detector_clears_cache() -> None:
    """After reset_yolo_detector(), the next get_yolo_detector()
    call rebuilds the model (factory invoked again)."""
    _reset_state()
    from unittest.mock import patch

    from pages._state import get_yolo_detector, reset_yolo_detector

    fake_a = object()
    fake_b = object()
    with patch(
        "pages._state.get_multi_source_detector",
        side_effect=[fake_a, fake_b],
    ):
        d1 = get_yolo_detector()
        reset_yolo_detector()
        d2 = get_yolo_detector()
    assert d1 is fake_a
    assert d2 is fake_b
    assert d1 is not d2


def test_reset_yolo_detector_is_safe_when_empty() -> None:
    """reset_yolo_detector() must not raise when nothing is cached
    (idempotent — used in tests and dev tools)."""
    _reset_state()
    from pages._state import reset_yolo_detector

    # No prior cache → should silently no-op.
    reset_yolo_detector()
    reset_yolo_detector()  # idempotent