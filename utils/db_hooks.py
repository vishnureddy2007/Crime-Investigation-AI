"""
Auto-save hooks: best-effort database persistence from Streamlit pages.

Each `auto_save_last_*` function reads the matching `last_*` session
key, ensures a parent `case_id` exists, saves the artifact, and
remembers the new id under a `last_*_db_id` key so subsequent
page re-renders don't create duplicate rows.

Auto-save failures are logged via `st.warning` and swallowed — DB
problems must never break the pipeline page.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import DATABASE_PATH
from database.repository import (
    latest_case_for_source,
    save_analysis,
    save_case,
    save_report,
    save_storyboard,
    save_summary,
)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _resolve_path(db_path: Path | None) -> Path:
    return db_path if db_path is not None else DATABASE_PATH


def _resolve_case_id(db_path: Path, source_name: str, source_type: str) -> int:
    """Reuse the latest case for this source, or create a new one."""
    existing = latest_case_for_source(db_path, source_name)
    if existing is not None:
        return existing
    return save_case(db_path, source_name, source_type)


def _warn(message: str) -> None:
    """Surface a soft warning to the UI; never raise."""
    try:
        st.warning(message)
    except (ImportError, AttributeError, RuntimeError):
        # If Streamlit itself is unavailable, silently swallow.
        pass


# ----------------------------------------------------------------------
# Public auto-save hooks
# ----------------------------------------------------------------------
def auto_save_last_analysis(db_path: Path | None = None) -> int | None:
    """Save `last_analysis` to the DB if it exists and hasn't been saved yet."""
    try:
        analysis = st.session_state.get("last_analysis")
        if analysis is None:
            return None
        existing_id = st.session_state.get("last_analysis_db_id")
        if existing_id is not None:
            return int(existing_id)

        path = _resolve_path(db_path)
        case_id = _resolve_case_id(
            path, analysis.source_name, analysis.source_type,
        )
        new_id = save_analysis(path, case_id, analysis)
        st.session_state["last_analysis_db_id"] = new_id
        st.session_state["last_analysis_case_id"] = case_id
        return new_id
    except Exception as exc:
        _warn(f"⚠️ Database auto-save failed for analysis: {exc}")
        return None


def auto_save_last_summary(db_path: Path | None = None) -> int | None:
    try:
        summary = st.session_state.get("last_summary")
        if summary is None:
            return None
        existing_id = st.session_state.get("last_summary_db_id")
        if existing_id is not None:
            return int(existing_id)

        path = _resolve_path(db_path)
        case_id = _resolve_case_id(
            path, summary.source_name, summary.source_type,
        )
        new_id = save_summary(path, case_id, summary)
        st.session_state["last_summary_db_id"] = new_id
        st.session_state["last_summary_case_id"] = case_id
        return new_id
    except Exception as exc:
        _warn(f"⚠️ Database auto-save failed for summary: {exc}")
        return None


def auto_save_last_report(db_path: Path | None = None) -> int | None:
    try:
        report = st.session_state.get("last_report_data")
        if report is None:
            return None
        existing_id = st.session_state.get("last_report_db_id")
        if existing_id is not None:
            return int(existing_id)

        path = _resolve_path(db_path)
        case_id = _resolve_case_id(
            path, report.source_name, report.source_type,
        )
        new_id = save_report(path, case_id, report)
        st.session_state["last_report_db_id"] = new_id
        st.session_state["last_report_case_id"] = case_id
        return new_id
    except Exception as exc:
        _warn(f"⚠️ Database auto-save failed for report: {exc}")
        return None


def auto_save_last_storyboard(db_path: Path | None = None) -> int | None:
    try:
        storyboard = st.session_state.get("last_storyboard")
        if storyboard is None:
            return None
        existing_id = st.session_state.get("last_storyboard_db_id")
        if existing_id is not None:
            return int(existing_id)

        path = _resolve_path(db_path)
        case_id = _resolve_case_id(
            path, storyboard.source_name, storyboard.source_type,
        )
        new_id = save_storyboard(path, case_id, storyboard)
        st.session_state["last_storyboard_db_id"] = new_id
        st.session_state["last_storyboard_case_id"] = case_id
        return new_id
    except Exception as exc:
        _warn(f"⚠️ Database auto-save failed for storyboard: {exc}")
        return None
