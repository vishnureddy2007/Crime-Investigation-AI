"""Unit tests for `core/theming.py`."""
from __future__ import annotations

import pytest

from core.theming import (
    InvalidThemeError,
    available_themes,
    get_theme_css,
    render_theme_markdown,
)


class TestAvailableThemes:
    def test_contains_light_and_dark(self) -> None:
        themes = available_themes()
        assert "light" in themes
        assert "dark" in themes


class TestGetThemeCss:
    def test_light_returns_nonempty(self) -> None:
        css = get_theme_css("light")
        assert isinstance(css, str)
        assert len(css) > 50
        assert "--primary" in css
        assert "#" in css  # at least one hex color

    def test_dark_returns_nonempty(self) -> None:
        css = get_theme_css("dark")
        assert len(css) > 50
        assert "--primary" in css
        assert "#" in css

    def test_themes_differ(self) -> None:
        assert get_theme_css("light") != get_theme_css("dark")

    def test_invalid_theme_raises(self) -> None:
        with pytest.raises(InvalidThemeError):
            get_theme_css("neon")


class TestRenderThemeMarkdown:
    def test_wraps_in_style_tag(self) -> None:
        md = render_theme_markdown("dark")
        assert md.startswith("<style>")
        assert md.endswith("</style>")

    def test_invalid_raises(self) -> None:
        with pytest.raises(InvalidThemeError):
            render_theme_markdown("bogus")


class TestAccessibility:
    """The accessibility chunk must be present in every theme."""

    def test_light_has_skip_link(self) -> None:
        css = get_theme_css("light")
        assert ".skip-link" in css
        assert "focus" in css

    def test_dark_has_skip_link(self) -> None:
        css = get_theme_css("dark")
        assert ".skip-link" in css

    def test_both_have_reduced_motion_guard(self) -> None:
        for theme in ("light", "dark"):
            css = get_theme_css(theme)
            assert "prefers-reduced-motion" in css

    def test_both_have_focus_ring_variable(self) -> None:
        for theme in ("light", "dark"):
            css = get_theme_css(theme)
            assert "--focus-ring" in css

    def test_both_have_sr_only_utility(self) -> None:
        for theme in ("light", "dark"):
            css = get_theme_css(theme)
            assert ".sr-only" in css