"""
Frame conversion helpers between OpenCV (BGR, ndarray) and PIL (RGB, Image).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def cv2_to_pil(frame: np.ndarray[Any, np.dtype]) -> Image.Image:
    """
    Convert an OpenCV BGR frame to an RGB PIL.Image.

    Parameters
    ----------
    frame : np.ndarray
        BGR image as returned by `cv2.VideoCapture.read()`.

    Returns
    -------
    PIL.Image in RGB mode.
    """
    if frame is None:
        raise ValueError("Cannot convert None frame.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv2(image: Image.Image) -> np.ndarray[Any, np.dtype]:
    """Convert a PIL RGB image to an OpenCV BGR ndarray."""
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
