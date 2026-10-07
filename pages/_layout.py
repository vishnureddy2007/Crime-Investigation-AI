"""Shared branding layout helper for Streamlit pages.

Centralises the sidebar header, footer, and any project-branded
markup so every page looks consistent. Pages can either call the
helper functions individually or wrap their render in
``branded_section(...)``.

Why a separate module? Pages often want to render different
sections in different orders; a layout helper that returns
strings (rather than side-effects) is easy to reuse and test.
"""
from __future__ import annotations

import platform
from contextlib import contextmanager
from typing import Iterator

import streamlit as st

from config import APP_NAME, APP_VERSION


SIDEBAR_HEADER_HTML = """
<div class="brand-header">
  <span class="brand-mark">CV</span>
  <span>
    <span class="brand-name">{name}</span>
    <span class="brand-tag">v{version} &middot; {os_label}</span>
  </span>
</div>
"""

SIDEBAR_FOOTER_HTML = """
<div class="brand-footer">
  &copy; 2026 &middot; Final-Year B.Tech Project<br>
  Built with Streamlit &middot; YOLOv8 &middot; FLAN-T5
</div>
"""


def _os_label() -> str:
    sysname = platform.system().lower()
    if sysname == "windows":
        return "Windows"
    if sysname == "darwin":
        return "macOS"
    return "Linux"


def render_sidebar_header() -> None:
    """Render the branded header at the top of the sidebar."""
    html = SIDEBAR_HEADER_HTML.format(
        name=APP_NAME,
        version=APP_VERSION,
        os_label=_os_label(),
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


def render_sidebar_footer() -> None:
    """Render the branded footer at the bottom of the sidebar."""
    st.sidebar.markdown(SIDEBAR_FOOTER_HTML, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str | None = None) -> None:
    """Standard page header used by every page after the title."""
    st.markdown(
        f"<h1 style='margin:0 0 0.25rem 0'>{title}</h1>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


@contextmanager
def branded_section() -> Iterator[None]:
    """Context manager that draws the standard sidebar decorations.

    Usage::

        from pages._layout import branded_section

        def render():
            with branded_section():
                st.title("...")
                ...
    """
    render_sidebar_header()
    try:
        yield None
    finally:
        render_sidebar_footer()


# ----------------------------------------------------------------------
# Empty / error / status helpers (Phase A)
# ----------------------------------------------------------------------
def empty_state(
    title: str,
    body: str,
    action_label: str | None = None,
    action_target: str | None = None,
) -> None:
    """Render a consistent, professional empty-state card.

    Pages call this when there is no data to show yet (no cases in
    the database, no analysis yet, no report generated, etc.).

    Parameters
    ----------
    title : str
        Headline of the empty state, e.g. "No cases found."
    body : str
        One short paragraph explaining what the user should do next.
    action_label : str | None
        Optional label for a button rendered below the body.
    action_target : str | None
        If provided, the action button is a ``link_button`` to the
        given page name. Otherwise it is a plain ``button`` (the
        caller is responsible for handling its click).
    """
    st.markdown(
        f"""
<div class="cv-empty">
  <p class="cv-empty-title">{title}</p>
  <p class="cv-empty-body">{body}</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    if action_label:
        if action_target:
            st.link_button(action_label, action_target, use_container_width=False)
        else:
            st.button(action_label, use_container_width=False)


def friendly_error(
    exc: BaseException,
    *,
    fallback_title: str,
    detail: bool = False,
) -> None:
    """Render a user-friendly error block instead of a raw exception.

    The brief requires error states like::

        "Unable to generate the investigation summary.
         Reason: AI service unavailable.
         [Try Again]"

    This helper does exactly that — translates the exception class
    into a short professional phrase, and includes the technical
    detail only if ``detail=True`` (developer/system_status views).
    """
    from core import icons as _icons

    if isinstance(exc, ImportError):
        reason = "Required component is not installed."
    elif isinstance(exc, TimeoutError):
        reason = "Operation timed out."
    elif isinstance(exc, (ConnectionError, OSError)):
        reason = "External service is unreachable."
    elif isinstance(exc, PermissionError):
        reason = "Permission denied."
    elif isinstance(exc, FileNotFoundError):
        reason = "Required file was not found."
    elif isinstance(exc, RuntimeError):
        reason = "Processing failed."
    elif isinstance(exc, ValueError):
        reason = "Invalid input."
    else:
        reason = "Unexpected error."

    msg = f"**{fallback_title}**  \nReason: {reason}"
    if detail:
        msg += f"  \n\n`{exc.__class__.__name__}: {exc}`"
    st.error(msg)
    st.button(_icons.ACTION_RETRY, key=f"retry_{id(exc)}", use_container_width=False)


def status_pill(text: str, *, kind: str = "ok") -> None:
    """Render a small uppercase status pill (Completed / Pending / Failed).

    Replaces emoji-based status indicators (✅ ⚠ ❌) with a
    typographic-only badge so the UI looks forensic, not playful.
    """
    safe_kind = kind if kind in {"ok", "warn", "danger"} else "ok"
    st.markdown(
        f"<span class='cv-pill cv-pill--{safe_kind}'>{text}</span>",
        unsafe_allow_html=True,
    )


def environment_snapshot() -> dict[str, str]:
    """Return a dict describing the deployment environment (read-only)."""
    from config import (
        APP_ADDRESS,
        APP_HEADLESS,
        APP_PORT,
        APP_THEME,
        DATABASE_PATH,
        HF_HOME,
    )
    return {
        "APP_PORT":      str(APP_PORT),
        "APP_ADDRESS":   APP_ADDRESS,
        "APP_HEADLESS":  str(APP_HEADLESS),
        "APP_THEME":     APP_THEME,
        "HF_HOME":       str(HF_HOME),
        "DATABASE_PATH": str(DATABASE_PATH),
        "OS":            f"{platform.system()} {platform.release()}",
        "Python":        platform.python_version(),
    }


def reset_ui_session() -> None:
    """Drop every top-level ``last_*`` and ``*_db_id`` session key.

    Bound to a button in Settings; the actual button + confirm
    dialog lives in pages/settings.py.
    """
    for key in list(st.session_state.keys()):
        if key.startswith("last_") or key.endswith("_db_id") or key.endswith("_case_id"):
            try:
                del st.session_state[key]
            except (KeyError, RuntimeError):
                st.session_state.pop(key, None)
