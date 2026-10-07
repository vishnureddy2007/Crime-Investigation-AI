"""
Video I/O utilities: validation, metadata extraction, file saving.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import MAX_VIDEO_DURATION_SEC, MAX_VIDEO_SIZE_MB
from core.security import UnsafeFilenameError, safe_filename, safe_join

ALLOWED_VIDEO_EXTS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoLoadError(Exception):
    """Raised when a video cannot be loaded or fails validation."""


def is_allowed_video(filename: str) -> bool:
    """Return True if the filename's extension is in ALLOWED_VIDEO_EXTS."""
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTS


def validate_video_bytes(data: bytes, filename: str = "video") -> None:
    """
    Validate uploaded video bytes against size + basic OpenCV-readability.

    Raises
    ------
    VideoLoadError
        If the file is too large or cannot be opened by OpenCV.
    """
    if not data:
        raise VideoLoadError(f"{filename}: file is empty.")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise VideoLoadError(
            f"{filename}: file is {size_mb:.1f} MB, "
            f"exceeds limit of {MAX_VIDEO_SIZE_MB} MB."
        )

    # Try to read with OpenCV (uses a temporary file in memory)
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            # Fall back: try writing to a temp file and opening.
            # Sanitise `filename` so a malicious caller cannot escape
            # the cwd by supplying `../../etc/passwd`.
            from core.security import safe_filename
            safe_name = safe_filename(filename, max_length=64)
            tmp = Path.cwd() / f"_tmp_validate_{safe_name}"
            tmp.write_bytes(data)
            cap = cv2.VideoCapture(str(tmp))
            opened = cap.isOpened()
            cap.release()
            tmp.unlink(missing_ok=True)
            if not opened:
                raise VideoLoadError(f"{filename}: cannot decode video.")
    except VideoLoadError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise VideoLoadError(f"{filename}: cannot decode video ({exc}).") from exc


def get_video_metadata(path: Path | str) -> dict[str, float | int | str]:
    """
    Return basic metadata about a video file.

    Returns a dict with: path, fps, frame_count, width, height,
    duration_sec, size_mb.
    """
    p = Path(path)
    if not p.exists():
        raise VideoLoadError(f"File not found: {p}")

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoLoadError(f"Cannot open video: {p}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = (frame_count / fps) if fps > 0 else 0.0
    finally:
        cap.release()

    size_mb = p.stat().st_size / (1024 * 1024)

    if duration_sec > MAX_VIDEO_DURATION_SEC:
        raise VideoLoadError(
            f"Video is {duration_sec:.1f}s, "
            f"exceeds limit of {MAX_VIDEO_DURATION_SEC}s."
        )

    return {
        "path":         str(p),
        "fps":          round(fps, 2),
        "frame_count":  frame_count,
        "width":        width,
        "height":       height,
        "duration_sec": round(duration_sec, 2),
        "size_mb":      round(size_mb, 2),
    }


def save_uploaded_video(data: bytes, output_dir: Path, filename: str) -> Path:
    """Save uploaded video bytes to disk and return the path.

    The supplied filename is sanitised through `core.security.safe_filename`
    (defending against path-traversal separators, control characters,
    and empty-after-cleanup filenames) and the final destination is
    validated by `safe_join` to guarantee it lives inside `output_dir`.
    """
    safe_name = safe_filename(filename)
    out_path = safe_join(output_dir, safe_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path
