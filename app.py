"""
AI-Based Crime Investigation Assistant
=====================================

Production-grade Streamlit entry point with secure session authentication,
role-based authorization, dark forensic theme, and dynamic page routing.
"""

from __future__ import annotations

import streamlit as st

from config import APP_VERSION, DATABASE_PATH
from core.logging import configure_logging, get_logger
from core.theming import available_themes, render_theme_markdown
from database.db import init_db
from database.repository import (
    count_users,
    create_user,
    get_user_by_username,
    hash_password,
    update_user_last_login,
    verify_password,
)

# ----------------------------------------------------------------------
# Logging — initialised once at app boot
# ----------------------------------------------------------------------
configure_logging()

# ----------------------------------------------------------------------
# Page configuration - MUST be the first Streamlit command
# ----------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Crime Investigation AI",
        page_icon="⚖️",
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
    return "dark"


def _inject_theme() -> None:
    try:
        st.markdown(render_theme_markdown(_resolve_theme()), unsafe_allow_html=True)
    except Exception:
        pass


_inject_theme()


# ----------------------------------------------------------------------
# Page List & Routing
# ----------------------------------------------------------------------
PAGES: list[str] = [
    "Dashboard",
    "New Investigation",
    "Cases",
    "Evidence",
    "AI Analysis",
    "Reports",
    "Investigation Video",
    "AI Assistant",
    "Settings & System",
]


def _render_login_page() -> None:
    """Render the forensic login portal for unauthenticated users."""
    st.markdown(
        """
        <style>
        .login-card {
            max-width: 450px;
            margin: 60px auto;
            padding: 40px;
            background: #161C2E;
            border: 1px solid #2A344B;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            text-align: center;
        }
        .login-title {
            font-size: 26px;
            font-weight: 700;
            color: #00F2FE;
            margin-bottom: 8px;
        }
        .login-sub {
            font-size: 14px;
            color: #94A3B8;
            margin-bottom: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='login-title'>CRIME INVESTIGATION AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='login-sub'>Authorized Forensic Access Only</div>", unsafe_allow_html=True)

        with st.form("login_form"):
            username_input = st.text_input("Username")
            password_input = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Sign In to Portal", type="primary", use_container_width=True)

            if submit_login:
                if not username_input or not password_input:
                    st.error("Please provide both username and password.")
                else:
                    user = get_user_by_username(DATABASE_PATH, username_input)
                    if user and verify_password(user["password_hash"], password_input):
                        if not bool(user.get("is_active", 1)):
                            st.error("Account is inactive. Contact an administrator.")
                        else:
                            st.session_state["authenticated"] = True
                            st.session_state["user_id"] = user["id"]
                            st.session_state["username"] = user["username"]
                            st.session_state["role"] = user["role"]
                            update_user_last_login(DATABASE_PATH, user["id"])
                            st.success(f"Welcome, {user['username']}!")
                            st.rerun()
                    else:
                        st.error("Invalid username or password.")


def _render_page(page: str) -> None:
    """Dispatch to the selected page module."""
    log = get_logger(__name__)

    route = {
        "Dashboard":           ("pages.home",                 "render"),
        "Home":                ("pages.home",                 "render"),
        "New Investigation":   ("pages.investigation",        "render"),
        "Investigation":       ("pages.investigation",        "render"),
        "Cases":               ("pages.case_history",         "render"),
        "Evidence":            ("pages.evidence",             "render"),
        "AI Analysis":         ("pages.ai_insights",          "render"),
        "AI Insights":         ("pages.ai_insights",          "render"),
        "Reports":             ("pages.report",               "render"),
        "Investigation Video": ("pages.investigation_video",  "render"),
        "Reconstruction":      ("pages.investigation_video",  "render"),
        "AI Assistant":        ("pages.ai_assistant",         "render"),
        "Link Analysis":       ("pages.link_analysis",        "render"),
        "Settings & System":   ("pages.settings",             "render"),
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
    except Exception as exc:
        log.exception("page %s crashed", page)
        st.error(f"❌ Failed to render '{page}': {exc}")


def main() -> None:
    """Application entry point with authentication gate & role-based sidebar."""
    from pages._layout import render_sidebar_footer, render_sidebar_header

    # Ensure DB is initialized (schemas, migrations) at boot
    try:
        init_db(DATABASE_PATH)
        # Ensure at least one admin exists
        if count_users(DATABASE_PATH) == 0:
            def_hash = hash_password("admin123")
            create_user(DATABASE_PATH, "admin", def_hash, role="ADMIN", is_active=True)
    except Exception as exc:
        st.error(f"❌ Database initialization failed: {exc}")

    # Authentication Check
    if not st.session_state.get("authenticated", False):
        _render_login_page()
        return

    # Authenticated User Layout
    render_sidebar_header()

    # User Profile badge in sidebar
    st.sidebar.markdown(
        f"""
        <div style="background:#161C2E; padding:10px 14px; border-radius:8px; border:1px solid #2A344B; margin-bottom:15px;">
            <div style="font-size:12px; color:#94A3B8;">LOGGED IN AS</div>
            <div style="font-size:15px; font-weight:700; color:#00F2FE;">{st.session_state.get('username', 'User')}</div>
            <div style="font-size:11px; color:#34D399; font-weight:600;">ROLE: {st.session_state.get('role', 'INVESTIGATOR')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize default page in session state if not set
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = PAGES[0]

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

    st.session_state["nav_selection"] = page

    # Logout Button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.rerun()

    _render_page(page)

    render_sidebar_footer()
    st.sidebar.caption(f"Version {APP_VERSION}")


if __name__ == "__main__":
    main()
