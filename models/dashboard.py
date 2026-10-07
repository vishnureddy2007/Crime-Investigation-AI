"""
Dashboard aggregation helpers.

Pure functions that turn an EvidenceAnalysis (and a snapshot of the
last_* session-state keys) into ready-to-render values for the
Dashboard page. No Streamlit in the public API, no I/O.

Keeping this module Streamlit-free makes it trivially unit-testable
and lets the page layer be a thin presentation wrapper.
"""

from __future__ import annotations

from typing import Any

from config import SEVERITY_LEVEL_COLORS, SEVERITY_THRESHOLDS
from models.schemas import EvidenceAnalysis


# ----------------------------------------------------------------------
# Session-state keys the dashboard reports on
# ----------------------------------------------------------------------
_PIPELINE_KEYS: tuple[str, ...] = (
    "last_detection",
    "last_video",
    "last_analysis",
    "last_summary",
    "last_storyboard",
    "last_reconstruction_video_path",
)


# ----------------------------------------------------------------------
# Severity
# ----------------------------------------------------------------------
def compute_severity_band(score: int) -> str:
    """
    Return the severity level for a 0-100 score.

    Walks `SEVERITY_THRESHOLDS` in declared order (highest threshold
    first) and returns the first level whose threshold the score
    meets or exceeds. Returns "low" for an empty threshold list
    (defensive fallback).
    """
    for threshold, level in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


def severity_color(level: str) -> str:
    """
    Return the hex color associated with a severity level.

    Falls back to neutral gray for unknown levels so the UI never
    breaks on bad data.
    """
    return SEVERITY_LEVEL_COLORS.get(level, "#888888")


def severity_band_progress(score: int) -> tuple[int, int, str]:
    """
    Return `(current, max, label)` ready for `st.progress` + caption.

    `current` is clamped to [0, 100]; `max` is always 100; `label`
    is the severity band the score falls into.
    """
    clamped = max(0, min(100, int(score)))
    return clamped, 100, compute_severity_band(clamped)


# ----------------------------------------------------------------------
# Per-analysis aggregations
# ----------------------------------------------------------------------
def compute_kpis(analysis: EvidenceAnalysis) -> dict[str, Any]:
    """
    Flat dict of headline KPIs pulled directly from the analysis.

    No recomputation — every value already lives on the
    `EvidenceAnalysis` dataclass. `severity_color` is added so the
    page can style a metric or badge without an extra lookup.
    """
    return {
        "source_name":        analysis.source_name,
        "source_type":        analysis.source_type,
        "severity_score":     analysis.severity_score,
        "severity_level":     analysis.severity_level,
        "severity_color":     severity_color(analysis.severity_level),
        "total_objects":      analysis.total_objects,
        "person_count":       analysis.person_count,
        "weapon_count":       analysis.weapon_count,
        "vehicle_count":      analysis.vehicle_count,
        "bag_count":          analysis.bag_count,
        "has_threat":         analysis.has_threat,
        "suggested_category": analysis.suggested_category,
    }


def label_counts_series(analysis: EvidenceAnalysis) -> list[dict[str, Any]]:
    """
    Return label counts as a list of `{"label", "count"}` dicts,
    sorted by count descending, omitting zero-count labels.

    This shape is what `st.bar_chart(data=..., x="label", y="count")`
    consumes directly.
    """
    rows: list[dict[str, Any]] = [
        {"label": label, "count": count}
        for label, count in analysis.counts_by_label.items()
        if count > 0
    ]
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


# ----------------------------------------------------------------------
# Session-state readiness
# ----------------------------------------------------------------------
def summarize_session_state(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Return a dict of `{key: bool}` for every pipeline output key, plus
    an aggregate `pipeline_complete` flag.

    If `snapshot` is `None`, this function tries to read from
    `st.session_state`. The Streamlit import is wrapped in a
    broad `try/except` so the helper can be unit-tested without
    Streamlit being importable and so a missing session still
    produces a valid (all-False) summary.
    """
    if snapshot is None:
        try:
            import streamlit as st
            snapshot = dict(st.session_state)
        except (ImportError, AttributeError):
            snapshot = {}

    flags: dict[str, Any] = {k: bool(snapshot.get(k)) for k in _PIPELINE_KEYS}
    flags["pipeline_complete"] = all(flags[k] for k in _PIPELINE_KEYS)
    return flags
