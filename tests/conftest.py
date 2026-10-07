"""
Shared pytest fixtures for the project.

Keeps the cross-cutting infrastructure in one place so individual
test files stay focused on their own concerns.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"
SAMPLE_JPG:    Path = FIXTURES_DIR / "sample.jpg"
SAMPLE_MP4:    Path = FIXTURES_DIR / "sample.mp4"


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the project root (cached)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to the tests/fixtures directory (auto-created)."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_image_path(fixtures_dir: Path) -> Path:
    """Tiny 16x16 PNG-as-JPG image; auto-created if missing."""
    if not SAMPLE_JPG.exists() or SAMPLE_JPG.stat().st_size == 0:
        img = Image.new("RGB", (16, 16), color=(200, 50, 50))
        SAMPLE_JPG.write_bytes(_png_bytes(img))
    return SAMPLE_JPG


@pytest.fixture(scope="session")
def sample_video_path(fixtures_dir: Path) -> Path:
    """Tiny 2-second 8x8 MP4; auto-created if missing."""
    if not SAMPLE_MP4.exists() or SAMPLE_MP4.stat().st_size == 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(SAMPLE_MP4), fourcc, 10.0, (8, 8))
        try:
            for i in range(20):
                frame = np.full((8, 8, 3), (i * 12 % 255,), dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
    return SAMPLE_MP4


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Per-test SQLite path inside pytest's tmp_path."""
    return tmp_path / "test.db"
