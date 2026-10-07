"""
Tests for the path-based and save-side helpers in utils.image_io and
utils.video_io that aren't covered by the original test files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from utils.image_io import ImageLoadError, is_allowed_image, load_image_from_path
from utils.video_io import is_allowed_video, save_uploaded_video


# ----------------------------------------------------------------------
# load_image_from_path
# ----------------------------------------------------------------------
def test_load_image_from_path(sample_image_path: Path) -> None:
    img = load_image_from_path(sample_image_path)
    assert img.mode == "RGB"
    # The conftest fixture auto-creates a 16x16 image; assert mode/size type.
    assert img.size[0] > 0 and img.size[1] > 0


def test_load_image_from_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_image_from_path(tmp_path / "nope.png")


def test_load_image_from_path_accepts_string(sample_image_path: Path) -> None:
    """Path | str union — pass a string and confirm it still works."""
    img = load_image_from_path(str(sample_image_path))
    assert isinstance(img, Image.Image)


# ----------------------------------------------------------------------
# is_allowed_image / is_allowed_video edge cases
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("a.JPG",   True),
    ("a.jpeg",  True),
    ("a.png",   True),
    ("a.webp",  True),
    ("a.bmp",   True),
    ("a.gif",   False),     # gif is NOT in the project's allowed list
    ("a.txt",   False),
    ("no_ext",  False),
    ("a.mp4",   False),
])
def test_is_allowed_image_extensions(name: str, expected: bool) -> None:
    assert is_allowed_image(name) is expected


def test_is_allowed_video_extensions() -> None:
    assert is_allowed_video("a.mp4") is True
    assert is_allowed_video("a.MOV") is True
    assert is_allowed_video("a.avi") is True
    assert is_allowed_video("a.txt") is False


# ----------------------------------------------------------------------
# save_uploaded_video
# ----------------------------------------------------------------------
def test_save_uploaded_video_writes_file(tmp_path: Path) -> None:
    data = b"\x00\x01\x02 fake mp4 bytes" * 100
    out = save_uploaded_video(data, tmp_path, "clip.mp4")
    assert out.exists()
    assert out.read_bytes() == data
    assert out.name == "clip.mp4"


def test_save_uploaded_video_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested"
    out = save_uploaded_video(b"abc", nested, "c.mp4")
    assert out.exists()
    assert nested.is_dir()


def test_save_uploaded_video_filename_collision_overwrites(tmp_path: Path) -> None:
    save_uploaded_video(b"first",  tmp_path, "x.mp4")
    out = save_uploaded_video(b"second", tmp_path, "x.mp4")
    assert out.read_bytes() == b"second"
    assert len(list(tmp_path.glob("x*.mp4"))) == 1
