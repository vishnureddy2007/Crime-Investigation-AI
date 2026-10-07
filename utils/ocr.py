"""OCR helper with a deterministic stub fallback.

This module wraps `pytesseract` (the Python binding to Tesseract).
If Tesseract is not installed on the host, we fall back to a
deterministic stub that extracts a hash-based pseudo-conf and
returns an empty string. That way pages that call `extract_text()`
work even in a stripped-down Docker image, and unit tests don't
need an external binary.

Public API:
    OcrResult(text: str, confidence: float, engine: str)
    extract_text(image: Image.Image) -> OcrResult
    is_ocr_available() -> bool
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OcrResult:
    """Output of an OCR pass."""

    text: str
    confidence: float   # 0.0 – 1.0
    engine: str         # "tesseract" | "stub"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text":       self.text,
            "confidence": round(self.confidence, 4),
            "engine":     self.engine,
        }


def is_ocr_available() -> bool:
    """Return True iff pytesseract is importable (Tesseract binary optional)."""
    try:
        import pytesseract  # noqa: F401
        return True
    except (ImportError, AttributeError):
        return False


def _stub_extract(image: Any) -> OcrResult:
    """Deterministic pseudo-OCR for environments without Tesseract.

    Computes a tiny perceptual hash of the image and uses it as the
    "confidence" so callers get a stable number between 0 and 1.
    Accepts PIL images *and* numpy arrays.
    """
    try:
        # PIL.Image exposes .tobytes(); numpy arrays do too.
        # For very large arrays we hash a downsampled view instead
        # of the full pixel buffer (cheaper + bounded).
        if hasattr(image, "size") and hasattr(image, "convert"):
            # PIL.Image path — convert + hash a small thumbnail.
            thumb = image.convert("RGB").copy()
            thumb.thumbnail((32, 32))
            payload = thumb.tobytes()
        elif hasattr(image, "tobytes"):
            payload = image.tobytes()[:4096] if hasattr(image, "nbytes") else image.tobytes()
        else:
            payload = str(image).encode("utf-8")[:4096]
        h = hashlib.sha1(payload).hexdigest()
        # Take the first 4 hex chars → 0..65535 → /65535 → 0..1
        conf = int(h[:4], 16) / 65535.0
    except (AttributeError, TypeError, ValueError):
        conf = 0.0
    return OcrResult(text="", confidence=round(conf, 4), engine="stub")


def extract_text(image: Any) -> OcrResult:
    """Run OCR on a PIL image and return the extracted text.

    Falls back to the stub engine if `pytesseract` is unavailable or
    raises any error. Never throws.
    """
    if not is_ocr_available():
        return _stub_extract(image)

    try:
        import pytesseract  # type: ignore[import-not-found]

        text = pytesseract.image_to_string(image).strip()

        try:
            data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
            confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
            confs = [c for c in confs if 0 <= c <= 100]
            avg = sum(confs) / len(confs) / 100.0 if confs else 0.0
        except (AttributeError, TypeError, ValueError, RuntimeError):
            avg = 0.0

        return OcrResult(text=text, confidence=round(avg, 4), engine="tesseract")
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        # Any failure — missing binary, bad image, None, numpy dtype
        # mismatch, malformed Tesseract output — falls back to the
        # deterministic stub. extract_text() never raises.
        return _stub_extract(image)