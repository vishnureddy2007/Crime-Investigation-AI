"""
UI notification helper.

Centralises the project's success / error / warning / info message
formats so every page renders consistent feedback. All helpers are
pure Python — they return a string (or HTML block) for the caller to
pass into ``st.success`` / ``st.error`` / ``st.toast``.

Why a separate module? Streamlit's built-in ``st.toast`` only shows
briefly, and the page-level helpers don't accept HTML payloads. This
wrapper gives us:

- Stable icon prefixes (✅ / ❌ / ⚠️ / ℹ️)
- A consistent toast envelope
- A render helper that picks the right ``st.*`` call by severity
- A :func:`format_message` pure helper for tests

The module is Streamlit-aware (it imports ``streamlit`` at function
call time, not at module load) so unit tests can run without
Streamlit bootstrapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class Severity(str, Enum):
    """Notification severity."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


_ICON: dict[Severity, str] = {
    Severity.SUCCESS: "✅",
    Severity.ERROR:   "❌",
    Severity.WARNING: "⚠️",
    Severity.INFO:    "ℹ️",
}


@dataclass(frozen=True)
class Notification:
    """A single notification envelope."""

    severity: Severity
    title: str
    body: str = ""

    def render(self) -> str:
        """Return the rendered HTML block for the notification."""
        icon = _ICON[self.severity]
        if self.body:
            return f"**{icon} {self.title}** — {self.body}"
        return f"**{icon} {self.title}**"


def format_message(severity: Severity, title: str, body: str = "") -> str:
    """Pure helper: format a notification without touching Streamlit."""
    return Notification(severity=severity, title=title, body=body).render()


# ---- Streamlit-aware render helpers ---------------------------------------


def _streamlit_render(notification: Notification, *, toast: bool = False) -> None:
    """Render a notification into the running Streamlit app."""
    import streamlit as st  # local import keeps tests Streamlit-free

    fn: dict[Severity, Callable[..., None]] = {
        Severity.SUCCESS: st.success,
        Severity.ERROR:   st.error,
        Severity.WARNING: st.warning,
        Severity.INFO:    st.info,
    }
    payload = notification.render()
    if toast:
        try:
            st.toast(payload, icon=_ICON[notification.severity])
            return
        except (AttributeError, RuntimeError):
            # st.toast was added in Streamlit 1.27; fall back gracefully.
            pass
    fn[notification.severity](payload)


def notify(severity: Severity, title: str, body: str = "", *, toast: bool = False) -> Notification:
    """Build + render a notification, returning the underlying object."""
    notif = Notification(severity=severity, title=title, body=body)
    _streamlit_render(notif, toast=toast)
    return notif


def show_success(title: str, body: str = "", *, toast: bool = False) -> Notification:
    return notify(Severity.SUCCESS, title, body, toast=toast)


def show_error(title: str, body: str = "", *, toast: bool = False) -> Notification:
    return notify(Severity.ERROR, title, body, toast=toast)


def show_warning(title: str, body: str = "", *, toast: bool = False) -> Notification:
    return notify(Severity.WARNING, title, body, toast=toast)


def show_info(title: str, body: str = "", *, toast: bool = False) -> Notification:
    return notify(Severity.INFO, title, body, toast=toast)