"""
Tests for frame conversion utilities (OpenCV <-> PIL).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from utils.frame_utils import cv2_to_pil, pil_to_cv2


def test_cv2_to_pil_basic() -> None:
    # BGR image: red pixel in OpenCV is (0, 0, 255) -> RGB (255, 0, 0)
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[..., 0] = 0
    bgr[..., 1] = 0
    bgr[..., 2] = 255
    pil = cv2_to_pil(bgr)
    assert isinstance(pil, Image.Image)
    assert pil.size == (10, 10)
    r, g, b = pil.getpixel((5, 5))
    assert (r, g, b) == (255, 0, 0)


def test_pil_to_cv2_roundtrip() -> None:
    pil = Image.new("RGB", (16, 16), color=(10, 20, 30))
    bgr = pil_to_cv2(pil)
    assert bgr.shape == (16, 16, 3)
    pil2 = cv2_to_pil(bgr)
    assert pil2.getpixel((5, 5)) == (10, 20, 30)


def test_cv2_to_pil_none_raises() -> None:
    try:
        cv2_to_pil(None)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("Expected ValueError for None frame")
