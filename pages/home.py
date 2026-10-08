"""
Home / Landing Page — Crime Investigation AI.

Provides an executive hero section, real DB-backed KPIs, primary workflow CTAs,
5-step user workflow visualization, and real capability cards.
Zero development milestones or software progress tables.
"""

from __future__ import annotations

import streamlit as st

from config import APP_NAME, APP_VERSION, DATABASE_PATH
from core.icons import (
    ACTION_NEW_CASE,
    ACTION_VIEW_REPORTS,
    ACTION_VIEW_RECONSTRUCTION,
    ICON_DASHBOARD,
)
from pages._layout import render_page_header
from database.repository import list_cases


def _render_hero() -> None:
    """Render the primary executive hero banner."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #161C2E 100%);
            border-radius: 12px;
            padding: 2rem 2.5rem;
            color: #F8FAFC;
            border: 1px solid #2A344B;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            margin-bottom: 1.5rem;
        ">
            <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #00F2FE; margin-bottom: 0.5rem;">
                AI-Assisted Digital Forensic System
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; margin: 0 0 0.5rem 0; color: #ffffff; line-height: 1.2;">
                CRIME INVESTIGATION AI
            </h1>
            <p style="font-size: 1.05rem; color: #94A3B8; max-width: 850px; line-height: 1.5; margin-bottom: 1rem;">
                AI-assisted evidence analysis, weapon verification, case management, and automated explanation video generation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_db_kpis() -> None:
    """Render live KPI metric cards from SQLite repository."""
    from database.db import get_connection
    cases = list_cases(DATABASE_PATH)
    total_cases = len(cases)

    total_weapons = 0
    total_videos = 0
    total_reports = 0

    try:
        with get_connection(DATABASE_PATH) as conn:
            row_w = conn.execute("SELECT COUNT(*) FROM human_reviews WHERE decision = 'CONFIRMED'").fetchone()
            if row_w:
                total_weapons = row_w[0]
            row_v = conn.execute("SELECT COUNT(*) FROM investigation_videos").fetchone()
            if row_v:
                total_videos = row_v[0]
            row_r = conn.execute("SELECT COUNT(*) FROM reports").fetchone()
            if row_r:
                total_reports = row_r[0]
    except Exception:
        pass

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Cases", total_cases)
    with c2:
        st.metric("Active Investigations", total_cases)
    with c3:
        st.metric("Verified Weapons", total_weapons)
    with c4:
        st.metric("Generated Reports", total_reports)
    with c5:
        st.metric("Generated Videos", total_videos)


def _render_primary_actions() -> None:
    """Render prominent workflow action buttons."""
    st.markdown("---")
    st.subheader("Quick Actions")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("➕ New Investigation", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "New Investigation"
            st.rerun()

    with col2:
        if st.button("📁 Upload Evidence", use_container_width=True):
            st.session_state["nav_selection"] = "New Investigation"
            st.rerun()

    with col3:
        if st.button("🔬 Cases Directory", use_container_width=True):
            st.session_state["nav_selection"] = "Cases"
            st.rerun()

    with col4:
        if st.button("📄 View Reports", use_container_width=True):
            st.session_state["nav_selection"] = "Reports"
            st.rerun()

    with col5:
        if st.button("🎥 Generate Video", use_container_width=True):
            st.session_state["nav_selection"] = "Investigation Video"
            st.rerun()


def _render_user_workflow() -> None:
    """Render the 5-step user workflow."""
    st.markdown("---")
    st.subheader("Investigation Workflow")

    steps = [
        ("1. Upload Evidence", "Upload CCTV footage or crime scene images into the case workspace."),
        ("2. AI Detection", "YOLOv8 detects objects and highlights candidate threat weapons."),
        ("3. Weapon Verification", "Automatic weapon verification validates detection confidence."),
        ("4. AI Analysis", "Qwen3 14B / local engine synthesizes structured risk narratives."),
        ("5. Video & Report", "Generate 2D forensic explanation videos and official PDF/DOCX reports."),
    ]

    cols = st.columns(5)
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: #161C2E;
                    border: 1px solid #2A344B;
                    border-radius: 8px;
                    padding: 1rem;
                    height: 100%;
                ">
                    <div style="font-weight: 700; color: #00F2FE; margin-bottom: 0.4rem; font-size: 0.95rem;">{title}</div>
                    <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.4;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render() -> None:
    """Render the Home page."""
    _render_hero()
    _render_db_kpis()
    _render_primary_actions()
    _render_user_workflow()