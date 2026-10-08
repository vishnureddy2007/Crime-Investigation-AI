"""
Settings & System Diagnostics — Crime Investigation AI.

Consolidates Model configuration thresholds, UI Theme settings, SQLite management,
Host System Status & Health diagnostics, and Help/About.
"""

from __future__ import annotations

import platform
import streamlit as st

from config import (
    APP_NAME,
    APP_VERSION,
    DATABASE_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
)
from core.icons import ICON_SETTINGS, ICON_SYSTEM_STATUS
from core.theming import available_themes
from database.db import get_connection
from pages._layout import environment_snapshot, render_page_header, reset_ui_session


def _render_thresholds_tab() -> None:
    """Tab 1: Detection & Model Configuration."""
    st.markdown("### Detection & AI Configuration")
    st.caption("Adjust default confidence and IoU thresholds for object and weapon detection models.")

    conf = st.slider(
        "Default YOLO Confidence Threshold",
        min_value=0.10, max_value=0.90,
        value=YOLO_CONFIDENCE_THRESHOLD, step=0.05,
        key="set_conf_slider",
    )
    iou = st.slider(
        "Default YOLO IoU Threshold",
        min_value=0.10, max_value=0.90,
        value=YOLO_IOU_THRESHOLD, step=0.05,
        key="set_iou_slider",
    )

    st.markdown("#### Loaded Model Status")
    st.markdown("- **General Object Detector**: `yolov8n.pt` (YOLOv8 Small)")
    st.markdown("- **Weapon Detector**: `models/weapon.pt` (Multi-source weapon model)")
    st.markdown("- **Threat Weapon Model**: `models/threat_weapon.pt` (Expanded threat model)")
    from config import OLLAMA_MODEL
    st.markdown(f"- **NLP Narrative Model**: `{OLLAMA_MODEL}` (Local AI)")


def _render_theme_tab() -> None:
    """Tab 2: UI Theme & Session Management."""
    st.markdown("### Theme & Session Management")

    current_theme = st.session_state.get("ui_theme", "light")
    theme_choice = st.selectbox(
        "Select UI Theme",
        options=available_themes(),
        index=0 if current_theme == "light" else 1,
    )
    if theme_choice != current_theme:
        st.session_state["ui_theme"] = theme_choice
        st.rerun()

    st.markdown("---")
    st.markdown("#### Session State Reset")
    st.caption("Clear cached in-memory analysis and reloaded session objects.")
    if st.button("Reset Active Session Data", type="secondary"):
        reset_ui_session()
        st.success("Session state cleared successfully.")


def _render_system_status_tab() -> None:
    """Tab 3: System Status & Host Diagnostics."""
    st.markdown("### System Status & Host Diagnostics")

    env = environment_snapshot()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Deployment Environment")
        for k, v in env.items():
            st.markdown(f"- **{k}**: `{v}`")

    with col2:
        st.markdown("#### Database Connection")
        if DATABASE_PATH.exists():
            st.success(f"Database connected: `{DATABASE_PATH.name}`")
            try:
                with get_connection(DATABASE_PATH) as conn:
                    cases_cnt = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                st.markdown(f"- Total Recorded Cases: `{cases_cnt}`")
            except Exception as exc:
                st.error(f"DB Read Error: {exc}")
        else:
            st.warning("Database file not created yet.")


def _render_help_about_tab() -> None:
    """Tab 4: Help & About."""
    st.markdown(f"### About {APP_NAME}")
    st.write(
        f"**{APP_NAME}** (v{APP_VERSION}) is a production-grade AI platform designed to assist "
        "investigators with computer vision, NLP narrative summaries, and 2D/3D crime scene reconstructions."
    )

    st.markdown("#### Technologies Used")
    st.markdown("- **Core Language & UI**: Python 3.11+ & Streamlit")
    st.markdown("- **Computer Vision**: OpenCV & Ultralytics YOLOv8")
    from config import OLLAMA_MODEL
    st.markdown(f"- **Natural Language Processing**: Ollama & {OLLAMA_MODEL.replace(':', ' ')} (Local AI)")
    st.markdown("- **Report Generation**: ReportLab (PDF) & python-docx (DOCX)")
    st.markdown("- **Database Repository**: SQLite3")


def _render_user_management_tab() -> None:
    """Tab 5: User Management (Admin Only)."""
    st.markdown("### User Management & Role Authorization")
    st.caption("Manage user accounts, credentials, and access roles.")

    from database.repository import (
        create_user,
        delete_user,
        hash_password,
        list_all_users,
        update_user_status,
    )

    # User List Table
    users = list_all_users(DATABASE_PATH)
    if users:
        st.dataframe(users, use_container_width=True)
    else:
        st.info("No user accounts found.")

    st.markdown("---")
    st.markdown("#### Create New User")
    with st.form("create_user_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_username = st.text_input("Username")
        with col2:
            new_password = st.text_input("Password", type="password")
        with col3:
            new_role = st.selectbox("Role", ["INVESTIGATOR", "ADMIN", "VIEWER"])

        submit_user = st.form_submit_button("Create User Account", type="primary")
        if submit_user:
            if not new_username or not new_password:
                st.error("Username and password cannot be empty.")
            elif len(new_password) < 4:
                st.error("Password must be at least 4 characters.")
            else:
                try:
                    pwd_hash = hash_password(new_password)
                    create_user(DATABASE_PATH, new_username, pwd_hash, role=new_role, is_active=True)
                    st.success(f"User '{new_username}' created successfully!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to create user: {exc}")

    st.markdown("---")
    st.markdown("#### User Actions")
    if users:
        user_opts = {f"{u['username']} (ID: {u['id']}, Role: {u['role']})": u for u in users}
        sel_u_label = st.selectbox("Select User for Actions", list(user_opts.keys()))
        target_u = user_opts[sel_u_label]

        ac1, ac2 = st.columns(2)
        with ac1:
            curr_act = bool(target_u.get("is_active", 1))
            lbl = "Disable Account" if curr_act else "Enable Account"
            if st.button(lbl, type="secondary"):
                update_user_status(DATABASE_PATH, target_u["id"], not curr_act)
                st.success(f"Updated status for '{target_u['username']}'.")
                st.rerun()

        with ac2:
            if target_u["username"] != "admin":
                if st.button("Delete User Account", type="secondary"):
                    delete_user(DATABASE_PATH, target_u["id"])
                    st.success(f"Deleted user '{target_u['username']}'.")
                    st.rerun()


def render() -> None:
    """Render the Settings & System page."""
    render_page_header(
        ICON_SETTINGS,
        subtitle="Settings & System Diagnostics — Model configuration, UI theme controls, host diagnostics, and user management.",
    )

    current_role = st.session_state.get("role", "INVESTIGATOR")
    tabs_list = [
        "⚙️ Detection Settings",
        "🎨 Theme & Session",
        "🖥️ System Status",
        "ℹ️ Help & About",
    ]
    if current_role == "ADMIN" or st.session_state.get("username") == "admin":
        tabs_list.append("👥 User Management")

    tabs = st.tabs(tabs_list)

    with tabs[0]:
        _render_thresholds_tab()

    with tabs[1]:
        _render_theme_tab()

    with tabs[2]:
        _render_system_status_tab()

    with tabs[3]:
        _render_help_about_tab()

    if len(tabs) > 4:
        with tabs[4]:
            _render_user_management_tab()

