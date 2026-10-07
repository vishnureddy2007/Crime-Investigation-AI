"""Tests for the image I/O utilities."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from utils.image_io import (
    ImageLoadError,
    evidence_status_from_error,
    is_allowed_image,
    load_image_from_bytes,
    sha256_bytes,
)


def _png_bytes(size: tuple[int, int] = (32, 32), color: str = "red") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_from_bytes_rgb() -> None:
    img = load_image_from_bytes(_png_bytes(), source_name="test.png")
    assert img.mode == "RGB"
    assert img.size == (32, 32)


def test_load_image_from_bytes_empty_raises() -> None:
    try:
        load_image_from_bytes(b"", source_name="empty.png")
    except ImageLoadError:
        return
    raise AssertionError("Expected ImageLoadError for empty input")


def test_load_image_from_bytes_garbage_raises() -> None:
    try:
        load_image_from_bytes(b"not an image", source_name="bad.png")
    except ImageLoadError:
        return
    raise AssertionError("Expected ImageLoadError for garbage input")


def test_is_allowed_image_extensions() -> None:
    assert is_allowed_image("a.jpg")
    assert is_allowed_image("a.JPG")
    assert is_allowed_image("a.png")
    assert not is_allowed_image("a.gif")
    assert not is_allowed_image("a.txt")


def test_sha256_bytes_is_deterministic() -> None:
    """Same bytes → same SHA-256 hash; used as evidence fingerprint."""
    a = sha256_bytes(b"hello world")
    b = sha256_bytes(b"hello world")
    assert a == b
    assert a == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_sha256_bytes_distinguishes_content() -> None:
    """Different bytes → different hashes."""
    assert sha256_bytes(b"a") != sha256_bytes(b"b")


def test_evidence_status_verified_when_no_error() -> None:
    assert evidence_status_from_error(None) == "VERIFIED"


def test_evidence_status_rejected_for_bad_input() -> None:
    """extension/empty/corrupt/format/size → REJECTED (user-facing)."""
    for reason in ("extension", "empty", "corrupt", "format", "size"):
        assert evidence_status_from_error({"reason": reason}) == "REJECTED"


def test_evidence_status_error_for_unknown_reason() -> None:
    """Anything we don't recognise → ERROR (not VERIFIED, not REJECTED)."""
    assert evidence_status_from_error({"reason": "weird_thing"}) == "ERROR"
    assert evidence_status_from_error({"reason": ""}) == "ERROR"
