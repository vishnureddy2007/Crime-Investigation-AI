"""Unit tests for `core/security.py`."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.security import (
    PathTraversalError,
    UnsafeFilenameError,
    is_allowed_extension,
    safe_filename,
    safe_join,
    validate_email,
    validate_text_input,
)


class TestSafeFilename:
    def test_basic(self) -> None:
        assert safe_filename("hello.txt") == "hello.txt"

    def test_spaces_replaced(self) -> None:
        assert safe_filename("my file.png") == "my_file.png"

    def test_unicode_replaced(self) -> None:
        result = safe_filename("héllo.jpg")
        assert "héllo" not in result
        assert result.endswith(".jpg")

    def test_empty_raises(self) -> None:
        with pytest.raises(UnsafeFilenameError):
            safe_filename("")

    def test_all_unsafe_raises(self) -> None:
        with pytest.raises(UnsafeFilenameError):
            safe_filename("@#$%^&")

    def test_strips_path_components(self) -> None:
        assert safe_filename("/etc/passwd") == "passwd"
        assert safe_filename("../../etc/passwd") == "passwd"

    def test_truncation_keeps_extension(self) -> None:
        long_stem = "a" * 200
        out = safe_filename(f"{long_stem}.png", max_length=20)
        assert out.endswith(".png")
        assert len(out) <= 20

    def test_none_raises(self) -> None:
        with pytest.raises(UnsafeFilenameError):
            safe_filename(None)  # type: ignore[arg-type]


class TestIsAllowedExtension:
    def test_basic(self) -> None:
        assert is_allowed_extension("a.jpg", {"jpg", "png"})
        assert not is_allowed_extension("a.gif", {"jpg", "png"})

    def test_case_insensitive(self) -> None:
        assert is_allowed_extension("A.JPG", {"jpg"})
        assert is_allowed_extension("A.Jpg", {"JPG"})

    def test_no_extension(self) -> None:
        assert not is_allowed_extension("a", {"jpg"})

    def test_empty(self) -> None:
        assert not is_allowed_extension("", {"jpg"})
        assert not is_allowed_extension("a.jpg", set())


class TestSafeJoin:
    def test_inside_base(self, tmp_path: Path) -> None:
        base = tmp_path
        result = safe_join(base, "subdir", "file.txt")
        assert str(result).startswith(str(base.resolve()))

    def test_traversal_blocked(self, tmp_path: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_join(tmp_path, "..", "etc", "passwd")

    def test_absolute_path_blocked(self, tmp_path: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_join(tmp_path, "/etc/passwd")

    def test_no_parts_returns_base(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path) == tmp_path.resolve()


class TestValidateTextInput:
    def test_basic(self) -> None:
        assert validate_text_input("hello", field="x") == "hello"

    def test_strips_whitespace(self) -> None:
        assert validate_text_input("  hello  ", field="x") == "hello"

    def test_strips_newlines_when_disallowed(self) -> None:
        out = validate_text_input("a\nb", field="x", allow_newlines=False)
        assert "\n" not in out

    def test_too_long(self) -> None:
        with pytest.raises(ValueError):
            validate_text_input("x" * 11, field="x", max_length=10)

    def test_too_short(self) -> None:
        with pytest.raises(ValueError):
            validate_text_input("ab", field="x", min_length=5)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_text_input(None, field="x")  # type: ignore[arg-type]

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_text_input(123, field="x")  # type: ignore[arg-type]


class TestValidateEmail:
    def test_basic(self) -> None:
        assert validate_email("Foo@Bar.COM") == "foo@bar.com"

    def test_no_at_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_email("foobar")

    def test_no_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_email("foo@bar")

    def test_too_long_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_email(f"{'a' * 250}@b.com")