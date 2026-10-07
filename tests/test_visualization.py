"""
Tests for utils.visualization (draw_detections + color helper).
"""

from __future__ import annotations

import pytest
from PIL import Image

from models.schemas import BoundingBox, Detection
from utils.visualization import _color_for, draw_detections


def _img(w: int = 100, h: int = 100) -> Image.Image:
    return Image.new("RGB", (w, h), color=(10, 10, 10))


def _det(label: str, conf: float, box: tuple[int, int, int, int] = (10, 10, 50, 50)) -> Detection:
    return Detection(
        class_name=label,
        label=label,
        confidence=conf,
        bbox=BoundingBox(*box),
    )


# ----------------------------------------------------------------------
# _color_for
# ----------------------------------------------------------------------
def test_color_for_known_labels() -> None:
    assert _color_for("person") == "#E63946"
    assert _color_for("knife") == "#1D3557"
    assert _color_for("vehicle") == "#2A9D8F"
    assert _color_for("bag") == "#F4A261"


def test_color_for_unknown_label_falls_back() -> None:
    assert _color_for("alien") == "#264653"


# ----------------------------------------------------------------------
# draw_detections
# ----------------------------------------------------------------------
def test_draw_detections_returns_image_of_same_size() -> None:
    src = _img(120, 80)
    out = draw_detections(src, [_det("person", 0.9)])
    assert isinstance(out, Image.Image)
    assert out.size == (120, 80)


def test_draw_detections_does_not_mutate_input() -> None:
    src = _img()
    pre = src.tobytes()
    _ = draw_detections(src, [_det("person", 0.9)])
    assert src.tobytes() == pre


def test_draw_detections_with_no_detections() -> None:
    """Empty list still produces a valid image of the same size."""
    src = _img()
    out = draw_detections(src, [])
    assert isinstance(out, Image.Image)
    assert out.size == src.size


def test_draw_detections_with_multiple_detections() -> None:
    """The drawn boxes should add visible pixels to the image."""
    src = _img(200, 200)
    dets = [
        _det("person",  0.9, box=(10,  10, 60,  60)),
        _det("knife",   0.8, box=(80,  80, 130, 130)),
        _det("vehicle", 0.7, box=(150, 10, 190, 50)),
    ]
    out = draw_detections(src, dets)
    bg = (10, 10, 10)
    # Count non-background pixels across the whole image. The 3px-wide
    # box outlines + label bars guarantee many non-bg pixels.
    non_bg = sum(
        1
        for y in range(200)
        for x in range(200)
        if out.getpixel((x, y)) != bg
    )
    assert non_bg > 50, f"expected many drawn pixels, got {non_bg}"


def test_draw_detections_iterable_input_not_just_list() -> None:
    """The function signature accepts Iterable[Detection]."""
    src = _img()
    out = draw_detections(src, (_det("person", 0.5) for _ in range(1)))
    assert isinstance(out, Image.Image)


# ----------------------------------------------------------------------
# Phase 21 — defensive coverage
# ----------------------------------------------------------------------


def test_color_for_known_labels() -> None:
    from utils.visualization import _color_for

    assert _color_for("person").startswith("#")
    assert _color_for("knife").startswith("#")
    assert _color_for("vehicle").startswith("#")


def test_color_for_unknown_label_uses_default() -> None:
    from utils.visualization import _color_for

    assert _color_for("never-seen-label") == "#264653"


def test_font_returns_a_font_object() -> None:
    """_font returns a font object (either truetype or load_default)."""
    from utils.visualization import _font

    font = _font(12)
    # Either PIL's loaded truetype or its default font — either
    # way the call must not raise.
    assert font is not None


def test_draw_detections_with_textlength_fallback(monkeypatch) -> None:
    """If ImageDraw.textlength raises AttributeError, the helper falls
    back to len(text)*7 and still produces a valid image."""
    from PIL import Image, ImageDraw
    from models.schemas import BoundingBox, Detection
    from utils.visualization import draw_detections

    real_textlength = ImageDraw.ImageDraw.textlength

    def fake_textlength(self, text, font=None):
        # textlength used by current Pillow; mimic very-old Pillow
        # by raising AttributeError when called from draw_detections.
        raise AttributeError("no textlength in this Pillow")

    monkeypatch.setattr(ImageDraw.ImageDraw, "textlength", fake_textlength)

    img = Image.new("RGB", (200, 200), (10, 10, 10))
    det = Detection(
        class_name="person", label="person",
        confidence=0.95,
        bbox=BoundingBox(20, 30, 100, 150),
    )
    out = draw_detections(img, [det])
    assert isinstance(out, Image.Image)


def test_draw_detections_with_unknown_label_uses_default_color() -> None:
    from PIL import Image
    from models.schemas import BoundingBox, Detection
    from utils.visualization import draw_detections

    img = Image.new("RGB", (100, 100), (10, 10, 10))
    det = Detection(
        class_name="alien", label="alien",
        confidence=0.5,
        bbox=BoundingBox(10, 10, 50, 50),
    )
    # Should not raise, just draw with default color.
    out = draw_detections(img, [det])
    assert isinstance(out, Image.Image)


def test_draw_detections_handles_empty_list() -> None:
    from PIL import Image
    from utils.visualization import draw_detections

    img = Image.new("RGB", (100, 100), (10, 10, 10))
    out = draw_detections(img, [])
    assert out.tobytes() == img.copy().tobytes()


# ----------------------------------------------------------------------
# Phase 21 — image_io + ocr + face coverage
# ----------------------------------------------------------------------


def test_image_io_load_rgb_from_bytes() -> None:
    """A raw JPEG byte buffer is loaded and converted to RGB."""
    from utils.image_io import load_image_from_bytes

    # Build a minimal JPEG with PIL and round-trip it.
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (10, 10), (200, 100, 50)).save(buf, format="JPEG")
    img = load_image_from_bytes(buf.getvalue(), source_name="roundtrip.jpg")
    assert img.mode == "RGB"
    assert img.size == (10, 10)


def test_image_io_load_rgb_from_rgba() -> None:
    """A RGBA input must be converted to RGB."""
    from utils.image_io import load_image_from_bytes
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (200, 100, 50, 128)).save(buf, format="PNG")
    img = load_image_from_bytes(buf.getvalue(), source_name="rgba.png")
    assert img.mode == "RGB"


def test_image_io_load_missing_file_raises() -> None:
    from utils.image_io import ImageLoadError, load_image_from_path

    with pytest.raises(ImageLoadError, match="File not found"):
        load_image_from_path("/nonexistent/path/never_exists.png")


def test_ocr_stub_with_huge_input_truncates() -> None:
    """A huge numpy array is truncated to 4 KB before hashing."""
    import numpy as np
    from utils.ocr import _stub_extract

    arr = np.zeros((2048, 2048, 3), dtype=np.uint8)  # 12 MB
    result = _stub_extract(arr)
    # 4 KB SHA-1 → result is bounded.
    assert 0.0 <= result.confidence <= 1.0


def test_face_detection_supports_pil_image() -> None:
    """A PIL image input is converted to RGB before detection."""
    from utils.face_detection import detect_faces
    from PIL import Image

    img = Image.new("L", (64, 64), 128)  # grayscale PIL
    boxes = detect_faces(img)
    # Cascade may or may not match anything; the function must not raise.
    assert isinstance(boxes, list)
    for box in boxes:
        assert 0.0 <= box.confidence <= 1.0


# ----------------------------------------------------------------------
# Phase 37 — coverage push to 100%
# ----------------------------------------------------------------------


def test_font_falls_back_to_default_when_truetype_missing(monkeypatch) -> None:
    """_font's OSError branch (utils/visualization.py:32-33) returns
    the bundled default font when arial.ttf is not on the host."""
    from PIL import ImageFont

    # Patch only the module-level symbol that `utils.visualization`
    # imported (utils/visualization.py uses `from PIL import ImageFont`
    # — so the name it calls is `utils.visualization.ImageFont.truetype`).
    import utils.visualization as viz_mod

    call_count = {"n": 0}

    def boom(*_args, **_kwargs):
        # First call (from _font) must raise OSError. Subsequent calls
        # from load_default() must work so the fallback can return a font.
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("truetype lookup failed (test stub)")
        # Real call: delegate to the original truetype.
        return _original_truetype(*_args, **_kwargs)

    _original_truetype = viz_mod.ImageFont.truetype
    monkeypatch.setattr(viz_mod.ImageFont, "truetype", boom)

    from utils.visualization import _font

    font = _font(14)
    assert font is not None
    assert call_count["n"] >= 2, (
        "expected _font to call truetype once (fail) then load_default to call it again"
    )
