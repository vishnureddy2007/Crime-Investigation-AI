"""
Reconstruction video synthesis.

Given a Storyboard, produce:
1. Captioned storyboard images (PIL, saved to disk).
2. A short MP4 reconstruction video via OpenCV with cross-fade
   transitions between scenes.

This is the visual "movie" of the crime scene reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    STORYBOARD_FADE_SEC,
    STORYBOARD_PANEL_SIZE,
    STORYBOARD_VIDEO_CODEC,
    STORYBOARD_VIDEO_FPS,
)
from models.schemas import Storyboard


# ----------------------------------------------------------------------
# Caption rendering
# ----------------------------------------------------------------------
def draw_caption_on_image(
    image: Any,
    title: str,
    caption: str,
    bar_ratio: float = 0.28,
) -> Any:
    """
    Draw a semi-transparent black bar at the bottom of the image with
    the title and caption. Returns a NEW image.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = image.convert("RGB").copy()
    w, h = img.size
    bar_h = int(h * bar_ratio)

    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0))

    try:
        title_font = ImageFont.truetype("arial.ttf", max(14, int(h * 0.06)))
        cap_font = ImageFont.truetype("arial.ttf", max(10, int(h * 0.035)))
    except OSError:
        title_font = ImageFont.load_default()
        cap_font = ImageFont.load_default()

    draw.text((14, h - bar_h + 8), title, fill=(255, 255, 255), font=title_font)

    wrapped = _wrap_text(caption, max_chars=max(30, int(w / 18)))
    for i, line in enumerate(wrapped[:2]):
        y = h - bar_h + 8 + max(18, int(h * 0.07)) + i * max(14, int(h * 0.05))
        draw.text((14, y), line, fill=(220, 220, 230), font=cap_font)

    return img


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ----------------------------------------------------------------------
# Video synthesis (OpenCV)
# ----------------------------------------------------------------------
def _pil_to_cv2(img: Any) -> np.ndarray[Any, np.dtype]:
    arr = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def render_storyboard_images(
    storyboard: Storyboard,
    output_dir: Path,
) -> list[Path]:
    """
    Save each storyboard scene as a captioned JPEG. Returns the list
    of saved paths in scene order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for scene in storyboard.scenes:
        if scene.image is None:
            continue
        captioned = draw_caption_on_image(scene.image, scene.title, scene.caption)
        out = output_dir / f"scene_{scene.index + 1:02d}.jpg"
        captioned.save(out, format="JPEG", quality=90)
        paths.append(out)
    return paths


def synthesize_reconstruction_video(
    storyboard: Storyboard,
    output_path: Path,
    fps: int = STORYBOARD_VIDEO_FPS,
    fade_sec: float = STORYBOARD_FADE_SEC,
    codec: str = STORYBOARD_VIDEO_CODEC,
) -> Path:
    """
    Build a short MP4 slideshow with cross-fade transitions.

    Each scene is held for `scene.duration_sec` seconds. Between
    adjacent scenes, frames cross-fade for `fade_sec` seconds.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = STORYBOARD_PANEL_SIZE
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer at {output_path} (codec={codec})."
        )

    held: list[np.ndarray[Any, np.dtype]] = []
    for scene in storyboard.scenes:
        if scene.image is None:
            continue
        captioned = draw_caption_on_image(scene.image, scene.title, scene.caption)
        held.append(_pil_to_cv2(captioned))

    if not held:
        writer.release()
        raise RuntimeError("Storyboard has no scenes with images; cannot render video.")

    # Hold the first scene
    frames_first = int(round(fps * storyboard.scenes[0].duration_sec))
    for _ in range(max(1, frames_first)):
        writer.write(held[0])

    # Cross-fade + hold for each subsequent scene
    for i in range(1, len(held)):
        prev = held[i - 1]
        curr = held[i]
        fade_frames = int(round(fps * fade_sec))
        for f in range(fade_frames):
            alpha = (f + 1) / max(1, fade_frames)
            blended = cv2.addWeighted(prev, 1.0 - alpha, curr, alpha, 0)
            writer.write(blended)
        hold_frames = int(round(fps * storyboard.scenes[i].duration_sec))
        for _ in range(max(1, hold_frames)):
            writer.write(curr)

    writer.release()
    return output_path
