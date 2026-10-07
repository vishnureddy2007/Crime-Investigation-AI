"""
Tests for the storyboard scene planner.
"""

from __future__ import annotations

import datetime

import pytest
from PIL import Image

from models.evidence_analyzer import analyze
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


def _analysis(counts: dict[str, int], conf: float = 0.8, source_type: str = "image"):
    dets = [_det(label, label, conf) for label, n in counts.items() for _ in range(n)]
    dr = DetectionResult(
        source_name="scene.jpg",
        timestamp=datetime.datetime.now(),
        detections=dets,
    )
    return analyze(AnalysisInput(
        source_name=dr.source_name,
        counts_by_label=dr.counts_by_label(),
        average_confidence=dr.average_confidence(),
        source_type=source_type,
    ))


def test_plan_scenes_empty_returns_min_scenes() -> None:
    a = _analysis({})
    sb = plan_scenes(a)
    assert sb.scene_count >= 4
    assert sb.scene_count <= 6
    # Scenes 1 and 6 (location + outcome) are always present
    titles = [s.title for s in sb.scenes]
    assert any("Location" in t for t in titles)
    assert any("Outcome" in t for t in titles)


def test_plan_scenes_includes_threat_when_weapon_present() -> None:
    a = _analysis({"person": 2, "knife": 1})
    sb = plan_scenes(a)
    titles = [s.title for s in sb.scenes]
    assert any("Threat" in t for t in titles)
    assert any("Persons" in t for t in titles)


def test_plan_scenes_includes_vehicle_scene() -> None:
    a = _analysis({"vehicle": 1})
    sb = plan_scenes(a)
    assert any("Vehicle" in s.title for s in sb.scenes)


def test_plan_scenes_includes_bag_scene() -> None:
    a = _analysis({"person": 1, "bag": 1})
    sb = plan_scenes(a)
    assert any("Property" in s.title or "Bag" in s.title for s in sb.scenes)


def test_plan_scenes_respects_max_scenes() -> None:
    a = _analysis({"person": 5, "knife": 3, "vehicle": 2, "bag": 4})
    sb = plan_scenes(a, max_scenes=4)
    assert sb.scene_count == 4


def test_plan_scenes_uses_real_base_image_when_provided() -> None:
    a = _analysis({"person": 1})
    base = Image.new("RGB", (320, 240), color=(120, 80, 40))
    sb = plan_scenes(a, base_image=base)
    for scene in sb.scenes:
        assert scene.based_on_real_frame is True
        assert scene.image is not None
        # All scenes should be the same panel size
        assert scene.image.size == (640, 360)


def test_plan_scenes_uses_placeholder_when_no_image() -> None:
    a = _analysis({"person": 1})
    sb = plan_scenes(a, base_image=None)
    for scene in sb.scenes:
        assert scene.based_on_real_frame is False
        assert scene.image is not None
        assert scene.image.size == (640, 360)


def test_plan_scenes_total_duration_is_sum() -> None:
    a = _analysis({"person": 1})
    sb = plan_scenes(a, default_duration=1.5)
    expected = sum(s.duration_sec for s in sb.scenes)
    assert abs(sb.total_duration_sec - expected) < 0.01


# ----------------------------------------------------------------------
# Phase 21 — defensive coverage
# ----------------------------------------------------------------------


def test_plan_scenes_pads_when_evidence_is_sparse() -> None:
    """When min_scenes > matching candidates, the planner pads with
    'not observed in this evidence' placeholders."""
    a = _analysis({})  # no matches
    sb = plan_scenes(a, min_scenes=4, max_scenes=6)
    captions = [s.caption for s in sb.scenes]
    assert any("not observed" in c for c in captions)


def test_plan_scenes_exits_padding_when_candidates_exhausted() -> None:
    """When the padding loop exhausts all False candidates before reaching
    min_scenes, the planner stops adding scenes. The defensive break fires
    when min_scenes > len(candidates) or when all candidates have been
    consumed (Phase 23 note: line 201 was actually unreachable in any input,
    and has been removed)."""
    a = _analysis({})
    # Force a higher min_scenes than there are candidates (6).
    sb = plan_scenes(a, min_scenes=10, max_scenes=8)
    # The planner returns at most max_scenes scenes and never repeats.
    assert sb.scene_count <= 8
    # No duplicate indices.
    indices = [s.index for s in sb.scenes]
    assert len(indices) == len(set(indices))


def test_panel_placeholder_handles_long_caption() -> None:
    """The placeholder renderer wraps long captions over multiple lines."""
    from models.scene_planner import _panel_placeholder

    caption = " ".join(["bravo"] * 30)  # > 46 char wrap
    img = _panel_placeholder("Long", caption)
    assert img.size == (640, 360)


def test_wrap_text_empty_string() -> None:
    from models.scene_planner import _wrap_text

    assert _wrap_text("") == []


def test_wrap_text_single_word_longer_than_max() -> None:
    """A single word longer than max_chars is returned as a single line."""
    from models.scene_planner import _wrap_text

    out = _wrap_text("superlongword", max_chars=4)
    assert out == ["superlongword"]


def test_tint_image_skips_black_tint() -> None:
    """A black tint returns the base image unchanged (no blending)."""
    from models.scene_planner import _tint_image

    base = Image.new("RGB", (100, 100), (200, 100, 50))
    out = _tint_image(base, (0, 0, 0))
    assert out.tobytes() == base.convert("RGB").tobytes()


def test_resize_to_panel_pads_to_aspect() -> None:
    """A non-square input is letterboxed into the standard panel."""
    from models.scene_planner import _resize_to_panel

    wide = Image.new("RGB", (200, 100), (10, 10, 10))
    out = _resize_to_panel(wide)
    assert out.size == (640, 360)


def test_panel_placeholder_font_arial_missing(monkeypatch) -> None:
    """When arial.ttf is missing, _panel_placeholder falls back to load_default()."""
    from PIL import ImageFont
    from models.scene_planner import _panel_placeholder

    real_truetype = ImageFont.truetype
    real_load_default = ImageFont.load_default

    def fake_truetype(font_path, *args, **kwargs):
        # Only fail for the arial path; let load_default's eventual
        # recursive call (with different args) succeed.
        if "arial" in str(font_path).lower():
            raise OSError("no arial here")
        return real_truetype(font_path, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", fake_truetype)
    # load_default on Windows + Pillow 14 internally calls truetype with
    # platform-specific paths — pass through to the real one for safety.
    monkeypatch.setattr(ImageFont, "load_default", real_load_default)

    img = _panel_placeholder("T", "C")
    assert img.size == (640, 360)


def test_panel_placeholder_textlength_missing(monkeypatch) -> None:
    """When ImageDraw.textlength is missing (very old Pillow), the helper
    falls back to len(text)*N. Both the title and the caption paths must
    still produce a valid image."""
    from PIL import Image, ImageDraw
    from models.scene_planner import _panel_placeholder

    def fake_textlength(self, text, font=None):
        raise AttributeError("no textlength")

    monkeypatch.setattr(ImageDraw.ImageDraw, "textlength", fake_textlength)

    img = _panel_placeholder("Title", "multi line caption that wraps")
    assert img.size == (640, 360)


def test_plan_scenes_padding_loop_breaks_when_no_remaining_candidates() -> None:
    """Phase 23 — exercise a config that drives the padding loop to
    actually iterate past one match."""
    from models.evidence_analyzer import analyze
    from models.schemas import AnalysisInput

    a = analyze(AnalysisInput(
        source_name="everything.png",
        counts_by_label={"person": 1, "knife": 1, "vehicle": 1, "bag": 1},
        average_confidence=0.9,
        source_type="image",
    ))
    # min_scenes=4, max_scenes=6: all 6 candidates match, so 6 scenes are
    # returned (Location + Persons + Threat + Vehicle + Property + Outcome).
    sb = plan_scenes(a, min_scenes=4, max_scenes=6)
    assert sb.scene_count == 6
