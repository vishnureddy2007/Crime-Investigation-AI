"""
Image I/O helpers.

Centralizes validation and conversion so every module that takes
an image uses the same rules.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# File extensions we accept from the uploader
ALLOWED_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageLoadError(Exception):
    """Raised when an uploaded file cannot be loaded as an image."""


def sha256_bytes(data: bytes) -> str:
    """Stable hex SHA-256 hash of file bytes. Same content → same hash."""
    return hashlib.sha256(data).hexdigest()


def evidence_status_from_error(error: dict | None) -> str:
    """Map a freeform FileEvidence.error dict to a formal status string.

    Returns one of: PENDING / VERIFIED / REJECTED / ERROR. This is the
    single source of truth used by the UI and the audit log.
    """
    if error is None:
        return "VERIFIED"
    reason = (error.get("reason") or "").lower()
    if reason in {"extension", "empty", "corrupt", "format", "size"}:
        return "REJECTED"
    return "ERROR"


def load_image_from_bytes(data: bytes, source_name: str = "image") -> Image.Image:
    """
    Decode raw bytes into an RGB PIL.Image.

    Raises
    ------
    ImageLoadError
        If the bytes cannot be decoded.
    """
    if not data:
        raise ImageLoadError(f"{source_name}: file is empty.")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError as exc:
        raise ImageLoadError(
            f"{source_name}: not a recognized image format."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ImageLoadError(
            f"{source_name}: failed to decode ({exc})."
        ) from exc

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_image_from_path(path: Path | str) -> Image.Image:
    """Load an image from a filesystem path, raising ImageLoadError on failure."""
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"File not found: {p}")
    if p.suffix.lower() not in ALLOWED_IMAGE_EXTS:
        raise ImageLoadError(
            f"Unsupported extension '{p.suffix}'. "
            f"Allowed: {sorted(ALLOWED_IMAGE_EXTS)}"
        )
    return load_image_from_bytes(p.read_bytes(), source_name=p.name)


def is_allowed_image(filename: str) -> bool:
    """Return True if the filename's extension is in ALLOWED_IMAGE_EXTS."""
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTS
