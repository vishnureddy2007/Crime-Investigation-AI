"""Tests for ``core.notifications``."""

from __future__ import annotations

from core.notifications import (
    Notification,
    Severity,
    format_message,
)


def test_format_message_includes_icon_and_title() -> None:
    rendered = format_message(Severity.SUCCESS, "Saved", "row 42")
    assert "✅" in rendered
    assert "Saved" in rendered
    assert "row 42" in rendered


def test_format_message_without_body() -> None:
    rendered = format_message(Severity.ERROR, "Boom")
    assert "❌" in rendered
    assert "Boom" in rendered


def test_severity_icons_match_expectations() -> None:
    assert "✅" in format_message(Severity.SUCCESS, "ok")
    assert "❌" in format_message(Severity.ERROR, "bad")
    assert "⚠️" in format_message(Severity.WARNING, "warn")
    assert "ℹ️" in format_message(Severity.INFO, "fyi")


def test_notification_dataclass_renders() -> None:
    n = Notification(severity=Severity.WARNING, title="Disk", body="90% full")
    assert "Disk" in n.render()
    assert "90% full" in n.render()


def test_notification_is_immutable() -> None:
    n = Notification(severity=Severity.INFO, title="x")
    try:
        n.title = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Notification should be frozen")

# ----------------------------------------------------------------------
# Phase 17 — Streamlit-aware renderers (mocked)
# ----------------------------------------------------------------------


class _MockStreamlit:
    """Capture every st.success / st.error / st.warning / st.info / st.toast call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []  # (fn, payload, icon)
        self.toast_supported = True

    def success(self, payload: str) -> None:
        self.calls.append(("success", payload, ""))

    def error(self, payload: str) -> None:
        self.calls.append(("error", payload, ""))

    def warning(self, payload: str) -> None:
        self.calls.append(("warning", payload, ""))

    def info(self, payload: str) -> None:
        self.calls.append(("info", payload, ""))

    def toast(self, payload: str, icon: str = "") -> None:
        if not self.toast_supported:
            raise AttributeError("toast unsupported")
        self.calls.append(("toast", payload, icon))


def test_show_success_renders_to_st(monkeypatch) -> None:
    mock = _MockStreamlit()
    import sys
    monkeypatch.setitem(sys.modules, "streamlit", mock)

    from core.notifications import show_success

    n = show_success("Saved")
    assert n.severity.value == "success"
    assert ("success", "**✅ Saved**", "") in mock.calls


def test_show_error_renders_to_st(monkeypatch) -> None:
    mock = _MockStreamlit()
    import sys
    monkeypatch.setitem(sys.modules, "streamlit", mock)

    from core.notifications import show_error

    show_error("Boom", "stack overflow")
    assert any(
        fn == "error" and "Boom" in payload and "stack overflow" in payload
        for fn, payload, _ in mock.calls
    )


def test_show_warning_with_body(monkeypatch) -> None:
    mock = _MockStreamlit()
    import sys
    monkeypatch.setitem(sys.modules, "streamlit", mock)

    from core.notifications import show_warning

    show_warning("Disk", "90% full")
    assert ("warning", "**⚠️ Disk** — 90% full", "") in mock.calls


def test_show_info_toast_path(monkeypatch) -> None:
    mock = _MockStreamlit()
    import sys
    monkeypatch.setitem(sys.modules, "streamlit", mock)

    from core.notifications import show_info

    show_info("FYI", toast=True)
    # Toast path takes precedence over st.info.
    assert any(fn == "toast" and "FYI" in payload for fn, payload, _ in mock.calls)


def test_show_info_falls_back_when_toast_unsupported(monkeypatch) -> None:
    mock = _MockStreamlit()
    mock.toast_supported = False
    import sys
    monkeypatch.setitem(sys.modules, "streamlit", mock)

    from core.notifications import show_info

    show_info("FYI", toast=True)
    # Falls back to st.info.
    assert ("info", "**ℹ️ FYI**", "") in mock.calls


def test_notify_returns_notification(monkeypatch) -> None:
    mock = _MockStreamlit()
    import sys
    monkeypatch.setitem(sys.modules, "streamlit", mock)

    from core.notifications import Severity, notify

    n = notify(Severity.SUCCESS, "title", "body")
    assert n.severity == Severity.SUCCESS
    assert n.title == "title"
    assert n.body == "body"
