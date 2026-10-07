"""Tests for Phase 16 fixes — drift, sanitisation, OCR resilience."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from core.logging import configure_logging, reset_for_tests
from services.analytics import severity_band
from utils import ocr
from utils.video_io import validate_video_bytes


# ---- severity band drift --------------------------------------------------


def test_severity_band_agrees_with_config_thresholds() -> None:
    """The analytics bands must match config.SEVERITY_THRESHOLDS exactly."""
    from config import SEVERITY_THRESHOLDS

    sorted_thr = sorted(SEVERITY_THRESHOLDS, key=lambda t: t[0])
    # Build expected labels at every threshold inclusive and exclusive.
    for min_score, label in sorted_thr:
        assert severity_band(min_score) == label, (
            f"severity_band({min_score}) expected {label}, "
            f"got {severity_band(min_score)}"
        )
    # Score 100 must always be critical.
    assert severity_band(100) == "critical"


def test_severity_band_handles_high_boundary() -> None:
    assert severity_band(50) == "high"
    assert severity_band(75) == "critical"
    assert severity_band(74) == "high"


# ---- video_io tmp sanitisation -------------------------------------------


def test_validate_video_sanitises_tmp_filename(monkeypatch, tmp_path: Path) -> None:
    """A traversal-style filename must not escape the cwd when written."""
    monkeypatch.chdir(tmp_path)
    fake_bytes = b"\x00\x00\x00\x00fake mp4 content"
    import utils.video_io as video_io

    captured: dict[str, Path] = {}

    real_unlink = Path.unlink

    def tracking_unlink(self, *a, **kw):  # type: ignore[no-untyped-def]
        captured["path"] = Path(self)
        return real_unlink(self, *a, **kw)

    with patch.object(video_io.cv2, "imdecode", return_value=None), \
         patch.object(video_io.cv2, "VideoCapture") as mock_capture, \
         patch.object(Path, "unlink", tracking_unlink):
        mock_cap = mock_capture.return_value
        mock_cap.isOpened.return_value = True
        validate_video_bytes(fake_bytes, filename="../../../etc/passwd")

    assert "path" in captured, "expected a tmp file path to be captured"
    leaked = captured["path"]
    # The leaked path must be inside tmp_path (the cwd) and the
    # filename must not contain traversal components.
    leaked.resolve().relative_to(tmp_path.resolve())
    assert ".." not in leaked.name
    assert "/" not in leaked.name
    assert "\\" not in leaked.name


# ---- ocr stub handles numpy inputs ---------------------------------------


def test_ocr_stub_handles_numpy_array() -> None:
    """The stub fallback must accept both PIL images and numpy arrays."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    result = ocr._stub_extract(arr)
    assert 0.0 <= result.confidence <= 1.0
    assert result.engine == "stub"
    assert result.text == ""


def test_ocr_stub_handles_pil_image() -> None:
    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    result = ocr._stub_extract(img)
    assert 0.0 <= result.confidence <= 1.0
    assert result.engine == "stub"


def test_ocr_stub_handles_garbage_input() -> None:
    """An object with neither `.tobytes()` nor `convert()` must not crash."""
    class _Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    result = ocr._stub_extract(_Opaque())
    assert 0.0 <= result.confidence <= 1.0


# ---- core.logging LOG_FILE convergence -----------------------------------


def test_configure_logging_uses_config_log_file(monkeypatch, tmp_path: Path) -> None:
    """If `config.LOG_FILE` is set, configure_logging should honour it."""
    monkeypatch.delenv("LOG_FILE", raising=False)
    reset_for_tests()
    monkeypatch.setattr("config.LOG_FILE", tmp_path / "from_config.log", raising=False)
    try:
        out = configure_logging()
        assert out == tmp_path / "from_config.log"
    finally:
        reset_for_tests()


def test_configure_logging_falls_back_to_env(monkeypatch, tmp_path: Path) -> None:
    """If config import fails, fall back to the LOG_FILE env var."""
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "from_env.log"))
    reset_for_tests()
    # Force the config import to fail.
    import sys

    sys.modules.pop("config", None)
    with patch.dict(sys.modules, {"config": None}):
        # `from config import LOG_FILE` raises ImportError — handled.
        from core.logging import _resolve_log_file
        out = _resolve_log_file(None)
    assert out == tmp_path / "from_env.log"