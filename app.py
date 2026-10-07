"""
AI-Based Crime Investigation Assistant
=====================================

Production-grade Streamlit entry point (Phase 13).

This file is intentionally thin: it sets up Streamlit's page config,
injects the chosen theme, lists the available pages, and routes the
selection to the matching `pages/*` module.

All business logic lives in `pages/*`, `models/`, `services/`,
`core/`, and `database/`. Nothing in here touches I/O or the
business rules.
"""

from __future__ import annotations

import streamlit as st

from core.theming import (
    InvalidThemeError,
    available_themes,
    render_theme_markdown,
)
from core.logging import configure_logging, get_logger
from config import DATABASE_PATH
from database.db import init_db


# ----------------------------------------------------------------------
# Logging — initialised once at app boot
# ----------------------------------------------------------------------
configure_logging()

# ----------------------------------------------------------------------
# Page configuration - MUST be the first Streamlit command
# ----------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="AI Crime Investigation Assistant",
        page_icon="CI",  # text-only — no emoji chrome (Phase 50)
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass


# ----------------------------------------------------------------------
# Theme + branding
# ----------------------------------------------------------------------
_THEME_KEY = "ui_theme"


def _resolve_theme() -> str:
    if _THEME_KEY in st.session_state:
        candidate = st.session_state[_THEME_KEY]
        if candidate in available_themes():
            return candidate
    from config import APP_THEME
    if APP_THEME in available_themes():
        return APP_THEME
    return "light"


def _inject_theme() -> None:
    try:
        st.markdown(render_theme_markdown(_resolve_theme()), unsafe_allow_html=True)
    except Exception:
        # Prevent top-level import crashes during tests outside a active Streamlit runner
        pass


_inject_theme()


# ----------------------------------------------------------------------
# Page list
# ----------------------------------------------------------------------
PAGES: list[str] = [
    "Home",
    "Investigation",
    "AI Insights",
    "Reports",
    "Cases",
    "Link Analysis",
    "Reconstruction",
    "Settings & System",
]


def _render_page(page: str) -> None:
    """Dispatch to the selected page module."""
    log = get_logger(__name__)

    route = {
        "Home":              ("pages.home",           "render"),
        "Investigation":     ("pages.investigation",  "render"),
        "AI Insights":       ("pages.ai_insights",    "render"),
        "Reports":           ("pages.report",         "render"),
        "Cases":             ("pages.case_history",   "render"),
        "Link Analysis":      ("pages.link_analysis",  "render"),
        "Reconstruction":    ("pages.reconstruction", "render"),
        "Settings & System": ("pages.settings",       "render"),
    }

    entry = route.get(page)
    if entry is None:
        st.error(f"Unknown page: {page}")
        return

    module_name, attr = entry
    log.info("rendering page=%s", page)
    import importlib
    try:
        module = importlib.import_module(module_name)
        getattr(module, attr)()
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("page %s crashed", page)
        st.error(f"❌ Failed to render '{page}': {exc}")


def main() -> None:
    """Application entry point with branded sidebar + page routing."""
    from pages._layout import render_sidebar_header, render_sidebar_footer
    from config import APP_VERSION

    # Ensure DB is initialized (schemas, migrations) at boot
    try:
        init_db(DATABASE_PATH)
    except Exception as exc:
        st.error(f"❌ Database initialization failed: {exc}")

    render_sidebar_header()

    # Initialize default page in session state if not set
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = PAGES[0]

    # Use a simple radio without a 'key' to avoid session_state conflicts during programmatic navigation.
    # Instead, we use the 'index' parameter to sync with our session state.
    try:
        current_idx = PAGES.index(st.session_state["nav_selection"])
    except (ValueError, KeyError):
        current_idx = 0

    page = st.sidebar.radio(
        "Navigate",
        options=PAGES,
        index=current_idx,
        label_visibility="collapsed",
    )

    # Update session state when the user manually changes the radio button
    st.session_state["nav_selection"] = page

    _render_page(page)

    render_sidebar_footer()
    st.sidebar.caption(f"Version {APP_VERSION}")


if __name__ == "__main__":
    main()
