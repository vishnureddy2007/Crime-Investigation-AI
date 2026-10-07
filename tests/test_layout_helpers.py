"""Tests for the new layout helpers added in the Professional UI polish phase.

These tests use a per-test `_FakeStreamlit` instance and the
``monkeypatch.setattr`` fixture to swap it in for the bound ``st``
module inside ``pages._layout`` only — we do **not** touch
``sys.modules['streamlit']`` so subsequent tests (e.g.
``test_smoke_full``) keep importing the real Streamlit.
"""
from __future__ import annotations

from unittest.mock import MagicMock


def _make_fake_streamlit() -> MagicMock:
    """Build a MagicMock with the attributes `_layout` actually touches."""
    fake = MagicMock(name="fake_streamlit")
    fake.markdown = MagicMock()
    fake.button = MagicMock(return_value=False)
    fake.link_button = MagicMock(return_value=False)
    fake.error = MagicMock()
    fake.info = MagicMock()
    fake.warning = MagicMock()
    fake.success = MagicMock()
    fake.caption = MagicMock()
    return fake


# ----------------------------------------------------------------------
# core/icons.py
# ----------------------------------------------------------------------
def test_icons_module_exports_required_constants() -> None:
    """Every page + section + status label is exported and non-empty."""
    from core import icons

    for name in (
        "BRAND_NAME", "BRAND_TAGLINE",
        "ICON_HOME", "ICON_DASHBOARD", "ICON_INVESTIGATION",
        "ICON_IMAGE_DETECTION", "ICON_VIDEO_PROCESSING",
        "ICON_EVIDENCE_ANALYSIS", "ICON_AI_SUMMARY", "ICON_CHAT",
        "ICON_TIMELINE", "ICON_PREDICTION", "ICON_REPORT",
        "ICON_RECONSTRUCTION", "ICON_CASE_HISTORY", "ICON_ANALYTICS",
        "ICON_SETTINGS", "ICON_HELP", "ICON_CONTACT", "ICON_ABOUT",
        "ICON_SYSTEM_STATUS",
        "SECTION_KPIS", "SECTION_PIPELINE", "SECTION_DETECTIONS",
        "SUMMARY_EVIDENCE", "SECTION_OBSERVATIONS", "SECTION_DIAGNOSTICS",
        "SECTION_PER_FILE", "SECTION_COMBINED", "SECTION_DOWNLOADS",
        "SECTION_SEVERITY",
        "STATUS_OK", "STATUS_PENDING", "STATUS_FAILED", "STATUS_PROCESSING",
        "STATUS_NOT_GENERATED",
        "ACTION_ANALYZE", "ACTION_DOWNLOAD_PDF", "ACTION_DOWNLOAD_DOCX",
        "ACTION_RETRY", "ACTION_NEW_CASE", "ACTION_VIEW_REPORTS",
        "ACTION_VIEW_RECONSTRUCTION",
    ):
        assert hasattr(icons, name), f"missing icon constant: {name}"
        value = getattr(icons, name)
        assert isinstance(value, str) and value, f"empty icon: {name}"


def test_icons_have_no_emoji_glyphs() -> None:
    """No core/ icon constant may contain a Unicode emoji character
    (U+1F000..U+1FFFF or the pictograph/symbol ranges) — this is
    the whole point of the module."""
    from core import icons

    for name in dir(icons):
        if name.startswith("_"):
            continue
        value = getattr(icons, name)
        if not isinstance(value, str):
            continue
        for ch in value:
            cp = ord(ch)
            assert not (0x1F000 <= cp <= 0x1FFFF), (
                f"icon {name!r} contains emoji codepoint U+{cp:04X}"
            )
            assert not (0x2600 <= cp <= 0x27BF), (
                f"icon {name!r} contains dingbat codepoint U+{cp:04X}"
            )


# ----------------------------------------------------------------------
# pages/_layout.empty_state / friendly_error / status_pill
# ----------------------------------------------------------------------
def test_empty_state_renders_card_html() -> None:
    """empty_state writes the cv-empty card via st.markdown."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.empty_state("No cases found.", "Run a case to populate this dashboard.")
    fake.markdown.assert_called()
    body = fake.markdown.call_args[0][0]
    assert "cv-empty" in body
    assert "No cases found." in body
    assert "Run a case to populate this dashboard." in body


def test_empty_state_with_link_button() -> None:
    """When action_target is given, a link_button is rendered with that target."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.empty_state(
        "X", "Y", action_label="Open", action_target="Crime Scene Investigation",
    )
    fake.link_button.assert_called_once()
    args = fake.link_button.call_args[0]
    assert args[0] == "Open"
    assert args[1] == "Crime Scene Investigation"


def test_empty_state_with_plain_button() -> None:
    """When only action_label is given (no target), a plain button is rendered."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.empty_state("X", "Y", action_label="Reload")
    fake.button.assert_called_once()
    fake.link_button.assert_not_called()


def test_empty_state_no_action() -> None:
    """When no action is given, no buttons are rendered."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.empty_state("X", "Y")
    fake.button.assert_not_called()
    fake.link_button.assert_not_called()


def test_friendly_error_import_error() -> None:
    """ImportError → 'Required component is not installed.'"""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.friendly_error(ImportError("missing reportlab"),
                           fallback_title="Cannot render PDF report.")
    fake.error.assert_called()
    text = fake.error.call_args[0][0]
    assert "Cannot render PDF report." in text
    assert "Required component is not installed." in text


def test_friendly_error_timeout_error() -> None:
    """TimeoutError → 'Operation timed out.'"""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.friendly_error(TimeoutError("slow"),
                           fallback_title="AI summary timed out.")
    fake.error.assert_called()
    assert "Operation timed out." in fake.error.call_args[0][0]


def test_friendly_error_runtime_error() -> None:
    """RuntimeError → 'Processing failed.'"""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.friendly_error(RuntimeError("boom"),
                           fallback_title="Reconstruction failed.")
    fake.error.assert_called()
    assert "Processing failed." in fake.error.call_args[0][0]


def test_friendly_error_value_error() -> None:
    """ValueError → 'Invalid input.'"""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.friendly_error(ValueError("bad"),
                           fallback_title="Invalid input.")
    fake.error.assert_called()
    text = fake.error.call_args[0][0]
    assert "Invalid input." in text


def test_friendly_error_unknown_error() -> None:
    """Any other exception → 'Unexpected error.'"""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.friendly_error(ArithmeticError("x"),
                           fallback_title="X failed.")
    fake.error.assert_called()
    text = fake.error.call_args[0][0]
    assert "Unexpected error." in text


def test_friendly_error_detail_mode_includes_class() -> None:
    """When detail=True, the technical class name + message is included."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.friendly_error(
        RuntimeError("boom"), fallback_title="X failed.", detail=True,
    )
    text = fake.error.call_args[0][0]
    assert "RuntimeError" in text
    assert "boom" in text


def test_friendly_error_renders_retry_button() -> None:
    """friendly_error always appends a Try Again button."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.friendly_error(RuntimeError("x"), fallback_title="Failed.")
    fake.button.assert_called()
    label = fake.button.call_args[0][0]
    assert "Try Again" in label


def test_status_pill_renders_html() -> None:
    """status_pill writes a cv-pill span (no emoji)."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.status_pill("Completed", kind="ok")
    fake.markdown.assert_called()
    html = fake.markdown.call_args[0][0]
    assert "cv-pill" in html
    assert "cv-pill--ok" in html
    assert "Completed" in html
    for ch in html:
        cp = ord(ch)
        assert not (0x1F000 <= cp <= 0x1FFFF)


def test_status_pill_kind_falls_back_to_ok() -> None:
    """An unknown kind is silently coerced to 'ok'."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.status_pill("X", kind="bogus")
    html = fake.markdown.call_args[0][0]
    assert "cv-pill--ok" in html


def test_status_pill_warn() -> None:
    """warn kind emits cv-pill--warn."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.status_pill("X", kind="warn")
    html = fake.markdown.call_args[0][0]
    assert "cv-pill--warn" in html


def test_status_pill_danger() -> None:
    """danger kind emits cv-pill--danger."""
    fake = _make_fake_streamlit()
    from pages import _layout

    _layout.st = fake
    _layout.status_pill("X", kind="danger")
    html = fake.markdown.call_args[0][0]
    assert "cv-pill--danger" in html