"""Lightweight face detection helper.

This module wraps OpenCV's Haar cascade (`cv2.CascadeClassifier`) —
it ships with `opencv-python` so no extra dependency is required.
If the cascade XML isn't available on the host we return an empty
list rather than crashing. The output is a list of `FaceBox`.

Public API:
    FaceBox(x1, y1, x2, y2, confidence)
    detect_faces(image) -> list[FaceBox]
    is_face_detection_available() -> bool
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FaceBox:
    """A detected face rectangle with a confidence in [0, 1]."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "x1": round(self.x1, 2), "y1": round(self.y1, 2),
            "x2": round(self.x2, 2), "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
        }


# Haar cascades ship with opencv-python under data/haarcascades/.
_CASCADE_NAME = "haarcascade_frontalface_default.xml"


def _cascade_path() -> str | None:
    """Locate the bundled Haar cascade XML; return None if missing."""
    try:
        import cv2  # type: ignore[import-not-found]
        root = Path(cv2.data.haarcascades)  # type: ignore[attr-defined]
        candidate = root / _CASCADE_NAME
        if candidate.exists():
            return str(candidate)
    except (OSError, RuntimeError):
        pass
    return None


def is_face_detection_available() -> bool:
    return _cascade_path() is not None


def detect_faces(image: Any) -> list[FaceBox]:
    """Detect faces in `image` (PIL or numpy array) and return boxes."""
    cascade_file = _cascade_path()
    if cascade_file is None:
        return []

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        # Accept either PIL.Image or a numpy array.
        if hasattr(image, "convert"):
            arr = np.array(image.convert("RGB"))
        else:
            arr = image

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cascade_file)
        rects = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
        )
        out: list[FaceBox] = []
        for (x, y, w, h) in rects:
            out.append(
                FaceBox(
                    x1=float(x), y1=float(y),
                    x2=float(x + w), y2=float(y + h),
                    confidence=0.85,
                )
            )
        return out
    except (OSError, AttributeError, RuntimeError):
        return []