"""
Tests for the dashboard aggregation helpers.

The page layer is intentionally NOT tested here — Streamlit UI is
covered by manual smoke-testing the app. These tests pin the
behavior of the pure functions in `models/dashboard.py` so the
visual outputs can never silently disagree with the analyzer.
"""

from __future__ import annotations

import datetime

import pytest

from config import SEVERITY_LEVEL_COLORS, SEVERITY_THRESHOLDS
from models.dashboard import (
    compute_kpis,
    compute_severity_band,
    label_counts_series,
    severity_band_progress,
    severity_color,
    summarize_session_state,
)
from models.evidence_analyzer import analyze
from models.schemas import AnalysisInput, BoundingBox, Detection, DetectionResult


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _det(name: str, label: str, conf: float = 0.8) -> Detection:
    return Detection(
        class_name=name,
        label=label,
        confidence=conf,
        bbox=BoundingBox(0, 0, 10, 10),
    )


def _analysis(counts: dict[str, int], conf: float = 0.8, source_type: str = "image"):
    dets = [d for label, n in counts.items() for d in [_det(label, label, conf)] * n]
    dr = DetectionResult("x.jpg", datetime.datetime.now(), dets)
    return analyze(AnalysisInput(
        source_name=dr.source_name,
        counts_by_label=dr.counts_by_label(),
        average_confidence=dr.average_confidence(),
        source_type=source_type,
    ))


# ----------------------------------------------------------------------
# compute_severity_band
# ----------------------------------------------------------------------
def test_compute_severity_band_critical() -> None:
    assert compute_severity_band(80) == "critical"


def test_compute_severity_band_high() -> None:
    assert compute_severity_band(60) == "high"


def test_compute_severity_band_moderate() -> None:
    assert compute_severity_band(30) == "moderate"


def test_compute_severity_band_low() -> None:
    assert compute_severity_band(10) == "low"


@pytest.mark.parametrize("score,expected", [
    (0,   "low"),
    (24,  "low"),
    (25,  "moderate"),
    (49,  "moderate"),
    (50,  "high"),
    (74,  "high"),
    (75,  "critical"),
    (100, "critical"),
])
def test_compute_severity_band_boundaries(score: int, expected: str) -> None:
    assert compute_severity_band(score) == expected


def test_severity_band_config_order_matches_levels() -> None:
    """Config must keep the highest-threshold-first ordering."""
    levels = [lvl for _, lvl in SEVERITY_THRESHOLDS]
    assert levels == ["critical", "high", "moderate", "low"]


# ----------------------------------------------------------------------
# severity_color
# ----------------------------------------------------------------------
def test_severity_color_known_levels() -> None:
    for level, color in SEVERITY_LEVEL_COLORS.items():
        assert severity_color(level) == color


def test_severity_color_unknown_level_falls_back_to_gray() -> None:
    assert severity_color("nonexistent") == "#888888"


# ----------------------------------------------------------------------
# compute_kpis
# ----------------------------------------------------------------------
def test_compute_kpis_fields_and_counts() -> None:
    a = _analysis({"person": 2, "knife": 1, "bag": 1})
    kpis = compute_kpis(a)

    required = {
        "source_name", "source_type", "severity_score", "severity_level",
        "severity_color", "total_objects", "person_count", "weapon_count",
        "vehicle_count", "bag_count", "has_threat", "suggested_category",
    }
    assert required.issubset(kpis.keys())

    assert kpis["source_name"] == "x.jpg"
    assert kpis["source_type"] == "image"
    assert kpis["person_count"] == 2
    assert kpis["weapon_count"] == 1
    assert kpis["bag_count"] == 1
    assert kpis["total_objects"] == 4
    assert kpis["has_threat"] is True
    assert kpis["severity_color"] == severity_color(kpis["severity_level"])


def test_compute_kpis_no_threat_when_no_weapon() -> None:
    a = _analysis({"person": 1})
    kpis = compute_kpis(a)
    assert kpis["has_threat"] is False
    assert kpis["weapon_count"] == 0


# ----------------------------------------------------------------------
# label_counts_series
# ----------------------------------------------------------------------
def test_label_counts_series_sorted_desc_and_correct_total() -> None:
    a = _analysis({"person": 1, "knife": 3, "bag": 2})
    rows = label_counts_series(a)

    assert rows == [
        {"label": "knife", "count": 3},
        {"label": "bag", "count": 2},
        {"label": "person", "count": 1},
    ]
    assert sum(r["count"] for r in rows) == a.total_objects


def test_label_counts_series_omits_zero_count_labels() -> None:
    a = _analysis({"person": 2, "knife": 0, "bag": 1})  # knife absent from totals
    rows = label_counts_series(a)
    labels = [r["label"] for r in rows]
    assert "knife" not in labels
    assert "person" in labels
    assert "bag" in labels


def test_label_counts_series_empty_when_nothing_detected() -> None:
    a = _analysis({})
    assert label_counts_series(a) == []


# ----------------------------------------------------------------------
# severity_band_progress
# ----------------------------------------------------------------------
@pytest.mark.parametrize("score,expected_label", [
    (10,  "low"),
    (30,  "moderate"),
    (60,  "high"),
    (90,  "critical"),
])
def test_severity_band_progress_shape(score: int, expected_label: str) -> None:
    current, mx, label = severity_band_progress(score)
    assert current == score
    assert mx == 100
    assert label == expected_label


def test_severity_band_progress_clamps_above_100() -> None:
    current, mx, label = severity_band_progress(150)
    assert current == 100
    assert mx == 100
    assert label == "critical"


def test_severity_band_progress_clamps_below_zero() -> None:
    current, mx, label = severity_band_progress(-5)
    assert current == 0
    assert mx == 100
    assert label == "low"


# ----------------------------------------------------------------------
# summarize_session_state
# ----------------------------------------------------------------------
def test_summarize_session_state_with_full_snapshot() -> None:
    snap = {
        "last_detection":                  object(),
        "last_video":                      object(),
        "last_analysis":                   object(),
        "last_summary":                    object(),
        "last_storyboard":                 object(),
        "last_reconstruction_video_path":  "some/path.mp4",
    }
    out = summarize_session_state(snap)
    for k in (
        "last_detection", "last_video", "last_analysis",
        "last_summary", "last_storyboard",
        "last_reconstruction_video_path",
    ):
        assert out[k] is True
    assert out["pipeline_complete"] is True


def test_summarize_session_state_with_empty_snapshot() -> None:
    out = summarize_session_state({})
    for k in (
        "last_detection", "last_video", "last_analysis",
        "last_summary", "last_storyboard",
        "last_reconstruction_video_path",
    ):
        assert out[k] is False
    assert out["pipeline_complete"] is False


def test_summarize_session_state_with_partial_snapshot() -> None:
    out = summarize_session_state({"last_analysis": object()})
    assert out["last_analysis"] is True
    assert out["last_summary"] is False
    assert out["last_storyboard"] is False
    assert out["pipeline_complete"] is False


# ----------------------------------------------------------------------
# Phase 19 — defensive coverage
# ----------------------------------------------------------------------


def test_compute_severity_band_empty_thresholds_returns_low(monkeypatch) -> None:
    """If `SEVERITY_THRESHOLDS` is empty, the defensive fallback returns 'low'."""
    # dashboard.py captured the import at module load — patch the
    # reference inside the module, not the source `config`.
    monkeypatch.setattr("models.dashboard.SEVERITY_THRESHOLDS", [], raising=False)
    assert compute_severity_band(50) == "low"
    assert compute_severity_band(0) == "low"


def test_summarize_session_state_reads_streamlit_when_snapshot_none(monkeypatch) -> None:
    """When `snapshot=None`, the helper reads from streamlit.session_state."""
    import sys
    import types

    fake_state = {"last_analysis": object(), "last_summary": object()}
    fake_st = types.ModuleType("streamlit")
    fake_st.session_state = fake_state
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    out = summarize_session_state(None)
    assert out["last_analysis"] is True
    assert out["last_summary"] is True
    # The other keys are still False.
    assert out["last_detection"] is False
    assert out["pipeline_complete"] is False


def test_summarize_session_state_handles_missing_streamlit(monkeypatch) -> None:
    """When streamlit isn't importable, the helper falls back to {}."""
    import sys

    # Force the streamlit import inside summarize_session_state to fail.
    monkeypatch.setitem(sys.modules, "streamlit", None)
    out = summarize_session_state(None)
    assert out["pipeline_complete"] is False
    for k in (
        "last_detection", "last_video", "last_analysis",
        "last_summary", "last_storyboard",
        "last_reconstruction_video_path",
    ):
        assert out[k] is False
