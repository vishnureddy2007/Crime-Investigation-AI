"""
Tests for video I/O validation and metadata.

The tests use a small synthetic .mp4 generated into tests/fixtures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.video_io import (
    ALLOWED_VIDEO_EXTS,
    VideoLoadError,
    get_video_metadata,
    is_allowed_video,
    save_uploaded_video,
    validate_video_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_MP4 = FIXTURES / "sample.mp4"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def ensure_sample_video() -> None:
    """Create a small 2-second 8x8 test video if not present."""
    if SAMPLE_MP4.exists() and SAMPLE_MP4.stat().st_size > 0:
        return

    FIXTURES.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(SAMPLE_MP4), fourcc, 10.0, (8, 8))
    for i in range(20):  # 2 seconds at 10 fps
        frame = np.full((8, 8, 3), (i * 12 % 255,), dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_is_allowed_video() -> None:
    assert is_allowed_video("clip.mp4")
    assert is_allowed_video("clip.MP4")
    assert is_allowed_video("clip.avi")
    assert is_allowed_video("clip.mov")
    assert not is_allowed_video("clip.txt")
    assert not is_allowed_video("image.jpg")
    assert "mp4" not in ALLOWED_VIDEO_EXTS
    assert ".mp4" in ALLOWED_VIDEO_EXTS


def test_validate_video_bytes_accepts_real_video(ensure_sample_video) -> None:
    data = SAMPLE_MP4.read_bytes()
    # Should not raise
    validate_video_bytes(data, filename="sample.mp4")


def test_validate_video_bytes_rejects_empty() -> None:
    with pytest.raises(VideoLoadError):
        validate_video_bytes(b"", filename="x.mp4")


def test_validate_video_bytes_rejects_garbage() -> None:
    with pytest.raises(VideoLoadError):
        validate_video_bytes(b"not a video at all", filename="x.mp4")


def test_get_video_metadata(ensure_sample_video) -> None:
    meta = get_video_metadata(SAMPLE_MP4)
    assert meta["width"] == 8
    assert meta["height"] == 8
    assert meta["fps"] == 10.0
    assert meta["frame_count"] == 20
    assert 1.5 < meta["duration_sec"] < 2.5


def test_get_video_metadata_missing_file() -> None:
    with pytest.raises(VideoLoadError):
        get_video_metadata("does_not_exist.mp4")


# ----------------------------------------------------------------------
# save_uploaded_video — security
# ----------------------------------------------------------------------
def test_save_uploaded_video_sanitises_filename(tmp_path: Path) -> None:
    out = save_uploaded_video(b"data", tmp_path, "../../etc/passwd.mp4")
    # Filename must live under tmp_path and not contain traversal
    assert str(out).startswith(str(tmp_path.resolve()))
    assert ".." not in out.parts


def test_save_uploaded_video_keeps_extension(tmp_path: Path) -> None:
    out = save_uploaded_video(b"data", tmp_path, "my clip 123.MP4")
    assert out.suffix.lower() == ".mp4"


def test_save_uploaded_video_rejects_empty_filename(tmp_path: Path) -> None:
    import pytest as _pytest
    from core.security import UnsafeFilenameError
    with _pytest.raises(UnsafeFilenameError):
        save_uploaded_video(b"data", tmp_path, "@#$%")


# ----------------------------------------------------------------------
# Phase 24 — defensive coverage
# ----------------------------------------------------------------------


def test_validate_video_bytes_oversize_raises(monkeypatch) -> None:
    """A buffer exceeding MAX_VIDEO_SIZE_MB raises VideoLoadError.
    Line 41 of utils/video_io.py."""
    import utils.video_io as vio
    monkeypatch.setattr(vio, "MAX_VIDEO_SIZE_MB", 0)  # every buffer is "too big"
    import pytest as _pt
    with _pt.raises(VideoLoadError, match="exceeds limit"):
        validate_video_bytes(b"\x00" * 1024, "huge.mp4")


def test_get_video_metadata_rejects_missing_file(tmp_path: Path) -> None:
    """get_video_metadata raises VideoLoadError for a non-existent path.
    Line 79."""
    import pytest as _pt
    with _pt.raises(VideoLoadError, match="File not found"):
        get_video_metadata(tmp_path / "does_not_exist.mp4")


def test_get_video_metadata_rejects_non_video(tmp_path: Path) -> None:
    """get_video_metadata raises VideoLoadError when cv2 cannot open
    the file. Line 84."""
    fake = tmp_path / "not_a_video.txt"
    fake.write_bytes(b"definitely not a video\n")
    import pytest as _pt
    with _pt.raises(VideoLoadError, match="Cannot open video"):
        get_video_metadata(fake)


def test_validate_video_bytes_empty_raises() -> None:
    """An empty byte buffer raises VideoLoadError immediately at line 37."""
    import pytest as _pt
    with _pt.raises(VideoLoadError, match="file is empty"):
        validate_video_bytes(b"", "empty.mp4")
