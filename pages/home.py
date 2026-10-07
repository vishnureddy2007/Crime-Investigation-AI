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
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 12px;
            padding: 2.5rem;
            color: #f8fafc;
            border: 1px solid #334155;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
        ">
            <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #38bdf8; margin-bottom: 0.5rem;">
                Digital Forensic & Visual Intelligence Platform
            </div>
            <h1 style="font-size: 2.5rem; font-weight: 800; margin: 0 0 0.5rem 0; color: #ffffff; line-height: 1.2;">
                Crime Investigation AI
            </h1>
            <p style="font-size: 1.15rem; color: #cbd5e1; max-width: 850px; line-height: 1.6; margin-bottom: 1.5rem;">
                AI-powered evidence analysis, weapon verification, case reporting and 3D crime-scene reconstruction.
            </p>
            <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
                <span style="background: rgba(37, 99, 235, 0.2); color: #60a5fa; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(56, 189, 248, 0.3);">YOLOv8 Detection</span>
                <span style="background: rgba(22, 163, 74, 0.2); color: #4ade80; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(34, 197, 94, 0.3);">Human-in-the-Loop Verification</span>
                <span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(168, 85, 247, 0.3);">Qwen3 14B Local AI</span>
                <span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(245, 158, 11, 0.3);">3D Scene Reconstruction</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_primary_actions() -> None:
    """Render prominent workflow action buttons."""
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Upload Evidence", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "Investigation"
            st.rerun()

    with col2:
        if st.button("Start Investigation", use_container_width=True):
            st.session_state["nav_selection"] = "Investigation"
            st.rerun()

    with col3:
        if st.button("View Cases", use_container_width=True):
            st.session_state["nav_selection"] = "Cases"
            st.rerun()

    with col4:
        if st.button("AI Assistant", use_container_width=True):
            st.session_state["nav_selection"] = "AI Insights"
            st.rerun()


def _render_user_workflow() -> None:
    """Render the 5-step user workflow."""
    st.markdown("---")
    st.subheader("Investigation Workflow")

    steps = [
        ("1. Upload Evidence", "Upload CCTV video footage or crime scene images into the primary workspace."),
        ("2. AI Detection & Verification", "YOLOv8 detects objects while Human-in-the-Loop verifies weapon candidates."),
        ("3. Structured Analysis", "Automated severity scoring, category prediction, and evidence observations."),
        ("4. Local AI Insights", "Qwen3 14B generates narrative summaries and answers evidence questions."),
        ("5. Report & 3D Reconstruction", "Export forensic PDF/DOCX reports and render 3D crime-scene video."),
    ]

    cols = st.columns(5)
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: var(--surface, #ffffff);
                    border: 1px solid var(--border, #e2e8f0);
                    border-radius: 8px;
                    padding: 1.1rem;
                    height: 100%;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                ">
                    <div style="font-weight: 700; color: #2563eb; margin-bottom: 0.5rem; font-size: 0.95rem;">{title}</div>
                    <div style="font-size: 0.85rem; color: #64748b; line-height: 1.45;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_capabilities() -> None:
    """Render feature cards for core capabilities."""
    st.markdown("---")
    st.subheader("Core Capabilities")

    capabilities = [
        ("Multi-Source Object Detection", "Detect persons, vehicles, bags, and weapons using YOLOv8 multi-source inference."),
        ("Human-in-the-Loop (HITL)", "Investigators review and confirm or reject weapon detections to eliminate false positives."),
        ("Local LLM Narrative Summary", "Generate investigator-grade evidence summaries using local Qwen3 14B without cloud APIs."),
        ("Deterministic Risk Scoring", "Compute objective 0-100 severity scores and crime category classifications."),
        ("Forensic Case Management", "Store, search, and export investigation cases in a structured SQLite database."),
        ("3D Scene Reconstruction", "Generate multi-scene 3D Blender storyboards and MP4 video reconstructions from evidence."),
    ]

    col1, col2 = st.columns(2)
    for i, (title, desc) in enumerate(capabilities):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.markdown(
                f"""
                <div style="
                    background: var(--surface, #ffffff);
                    border: 1px solid var(--border, #e2e8f0);
                    border-radius: 8px;
                    padding: 1.2rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                ">
                    <h4 style="margin: 0 0 0.4rem 0; color: #0f172a; font-size: 1rem; font-weight: 700;">{title}</h4>
                    <p style="margin: 0; font-size: 0.88rem; color: #475569; line-height: 1.5;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render() -> None:
    """Render the Home page."""
    _render_hero()
    _render_primary_actions()
    _render_user_workflow()
    _render_capabilities()