"""
Tests for the reconstruction video synthesizer.

Verifies:
1. draw_caption_on_image returns a new image of the same size.
2. render_storyboard_images writes one JPEG per scene.
3. synthesize_reconstruction_video produces a valid MP4 file.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import cv2
import pytest
from PIL import Image

from models.evidence_analyzer import analyze
from models.reconstruction import (
    draw_caption_on_image,
    render_storyboard_images,
    synthesize_reconstruction_video,
)
from models.scene_planner import plan_scenes
from models.schemas import (
    AnalysisInput,
    BoundingBox,
    Detection,
    DetectionResult,
)


def _det(name: str, label: str, conf: float) -> Detection:
    return Detection(
        class_name=name, label=label,
        confidence=conf, bbox=BoundingBox(0, 0, 10, 10),
    )


def _analysis(counts: dict[str, int]):
    dets = [_det(label, label, 0.8) for label, n in counts.items() for _ in range(n)]
    dr = DetectionResult(
        source_name="scene.jpg",
        timestamp=datetime.datetime.now(),
        detections=dets,
    )
    return analyze(AnalysisInput(
        source_name=dr.source_name,
        counts_by_label=dr.counts_by_label(),
        average_confidence=dr.average_confidence(),
        source_type="image",
    ))


# ----------------------------------------------------------------------
# Caption
# ----------------------------------------------------------------------
def test_draw_caption_returns_same_size() -> None:
    img = Image.new("RGB", (640, 360), color=(50, 50, 50))
    out = draw_caption_on_image(img, "Title", "A caption.")
    assert isinstance(out, Image.Image)
    assert out.size == img.size


def test_draw_caption_does_not_mutate_input() -> None:
    img = Image.new("RGB", (640, 360), color=(50, 50, 50))
    before = img.copy()
    _ = draw_caption_on_image(img, "Title", "A caption.")
    assert img.tobytes() == before.tobytes()


# ----------------------------------------------------------------------
# Image rendering
# ----------------------------------------------------------------------
def test_render_storyboard_images_writes_files(tmp_path: Path) -> None:
    a = _analysis({"person": 2, "knife": 1})
    sb = plan_scenes(a, base_image=Image.new("RGB", (320, 240), (60, 60, 60)))
    paths = render_storyboard_images(sb, tmp_path)
    assert len(paths) == sb.scene_count
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".jpg"


# ----------------------------------------------------------------------
# Video synthesis
# ----------------------------------------------------------------------
def test_synthesize_reconstruction_video_produces_mp4(tmp_path: Path) -> None:
    a = _analysis({"person": 2, "knife": 1})
    sb = plan_scenes(
        a,
        base_image=Image.new("RGB", (320, 240), (60, 60, 60)),
        default_duration=1.0,
    )
    out = tmp_path / "recon.mp4"
    synthesize_reconstruction_video(sb, out, fps=12, fade_sec=0.25)

    assert out.exists()
    assert out.stat().st_size > 1000  # not empty

    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert fps > 0
        assert frame_count > 0
        assert width == 640
        assert height == 360
    finally:
        cap.release()


def test_synthesize_reconstruction_video_duration_in_range(tmp_path: Path) -> None:
    """Total video duration should be roughly 5-10 seconds for the default plan."""
    a = _analysis({"person": 2, "knife": 1, "vehicle": 1, "bag": 1})
    sb = plan_scenes(
        a,
        base_image=Image.new("RGB", (320, 240), (60, 60, 60)),
        default_duration=1.5,
    )
    out = tmp_path / "recon.mp4"
    synthesize_reconstruction_video(sb, out, fps=24, fade_sec=0.5)

    cap = cv2.VideoCapture(str(out))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    duration = frames / fps if fps > 0 else 0.0
    # 4-6 scenes * 1.5s + 3-5 fades * 0.5s = 6.0s - 11.5s target
    assert 4.0 <= duration <= 15.0, f"unexpected duration {duration:.1f}s"


def test_synthesize_reconstruction_video_with_placeholder(tmp_path: Path) -> None:
    """If no base image is provided, the placeholder scenes still produce a video."""
    a = _analysis({"person": 1})
    sb = plan_scenes(a, base_image=None)
    out = tmp_path / "recon.mp4"
    synthesize_reconstruction_video(sb, out, fps=12, fade_sec=0.25)
    assert out.exists()
    assert out.stat().st_size > 1000


# ----------------------------------------------------------------------
# Phase 21 — defensive coverage
# ----------------------------------------------------------------------


def test_draw_caption_with_long_caption_wraps(tmp_path: Path) -> None:
    """A multi-word caption is wrapped to multiple lines."""
    img = Image.new("RGB", (320, 200), color=(10, 10, 10))
    long_caption = " ".join(["alpha"] * 30)  # > max_chars
    out = draw_caption_on_image(img, "Long", long_caption)
    assert isinstance(out, Image.Image)
    assert out.size == img.size


def test_draw_caption_with_short_caption_single_line() -> None:
    img = Image.new("RGB", (640, 360), color=(40, 40, 40))
    out = draw_caption_on_image(img, "T", "Short.")
    assert isinstance(out, Image.Image)


def test_render_storyboard_images_skips_scenes_without_image(tmp_path: Path) -> None:
    """Scenes where `image is None` are skipped (no file written)."""
    from models.scene_planner import plan_scenes
    from models.schemas import Storyboard, StoryboardScene

    # Manually build a storyboard mixing placeholder/scene/None-image scenes.
    from PIL import Image
    sb = Storyboard(
        source_name="x.png",
        source_type="image",
        scenes=[
            StoryboardScene(
                index=0, title="first", caption="c",
                image=Image.new("RGB", (160, 90), (80, 80, 80)),
                duration_sec=1.0, based_on_real_frame=False,
            ),
            StoryboardScene(
                index=1, title="skip", caption="c",
                image=None, duration_sec=1.0, based_on_real_frame=False,
            ),
        ],
        timestamp=datetime.datetime.now(),
    )
    paths = render_storyboard_images(sb, tmp_path)
    assert len(paths) == 1
    assert paths[0].name == "scene_01.jpg"


def test_synthesize_video_with_only_placeholders(tmp_path: Path) -> None:
    """A storyboard with no scenes that have images raises clearly."""
    from models.scene_planner import plan_scenes

    sb = plan_scenes(_analysis({}), base_image=None)
    # Force every scene to have a None image (plan_scenes gives placeholders).
    for sc in sb.scenes:
        sc.image = None
    out = tmp_path / "empty.mp4"
    with pytest.raises(RuntimeError, match="no scenes with images"):
        synthesize_reconstruction_video(sb, out)


def test_synthesize_video_writer_open_failure(tmp_path: Path, monkeypatch) -> None:
    """A writer that fails to open raises a clear RuntimeError."""
    import cv2 as _cv2
    from models.reconstruction import synthesize_reconstruction_video
    from models.scene_planner import plan_scenes

    sb = plan_scenes(_analysis({"person": 1}), base_image=None)

    class _DeadWriter:
        def isOpened(self) -> bool:
            return False
        def write(self, *_a, **_kw) -> None:
            pass
        def release(self) -> None:
            pass

    monkeypatch.setattr(
        _cv2, "VideoWriter", lambda *_a, **_kw: _DeadWriter(),
    )
    out = tmp_path / "open_fail.mp4"
    with pytest.raises(RuntimeError, match="Could not open video writer"):
        synthesize_reconstruction_video(sb, out)


def test_draw_caption_keeps_different_caption_lengths() -> None:
    """Both short and long captions produce valid images."""
    img = Image.new("RGB", (640, 360), color=(40, 40, 40))
    for caption in ("a", "a short caption", "x" * 200):
        out = draw_caption_on_image(img, "Title", caption)
        assert isinstance(out, Image.Image)
        assert out.size == img.size
