"""Tests for ``services.analytics``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from database import repository as repo
from database.db import init_db
from models.schemas import EvidenceAnalysis, InvestigationSummary, ReportData
from services.analytics import (
    compute_snapshot,
    severity_band,
    worst_recent,
)


def _make_analysis(name: str, score: int, level: str, category: str, threat: bool) -> EvidenceAnalysis:
    return EvidenceAnalysis(
        source_name=name,
        source_type="image",
        counts_by_label={},
        total_objects=0,
        unique_labels=[],
        average_confidence=0.5,
        person_count=0,
        verified_weapon_count=1 if threat else 0,
        candidate_weapon_count=0,
        weapon_count=1 if threat else 0,
        vehicle_count=0,
        bag_count=0,
        severity_score=score,
        severity_level=level,
        suggested_category=category,
        key_observations=[],
        has_threat=threat,
        frame_count=1,
    )


def _seed_db(tmp_path: Path) -> Path:
    db = tmp_path / "db.sqlite"
    init_db(db)
    case = repo.save_case(db, "evidence_01.jpg", "image")
    repo.save_analysis(db, case, _make_analysis("evidence_01.jpg", 60, "high", "assault", True))
    case2 = repo.save_case(db, "parking_lot.mp4", "video")
    repo.save_analysis(db, case2, _make_analysis("parking_lot.mp4", 20, "low", "suspicious_activity", False))
    return db


def test_severity_band_labels() -> None:
    # Bands must agree with config.SEVERITY_THRESHOLDS.
    assert severity_band(0) == "low"
    assert severity_band(24) == "low"
    assert severity_band(25) == "moderate"
    assert severity_band(49) == "moderate"
    assert severity_band(50) == "high"
    assert severity_band(74) == "high"
    assert severity_band(75) == "critical"
    assert severity_band(100) == "critical"


def test_compute_snapshot_counts(tmp_path: Path) -> None:
    db = _seed_db(tmp_path)
    snap = compute_snapshot(db)
    assert snap.total_cases == 2
    assert snap.total_analyses == 2
    assert snap.total_reports == 0
    assert snap.total_feedback == 0
    assert snap.threats == 1
    assert snap.category_counts["assault"] == 1
    assert snap.category_counts["suspicious_activity"] == 1
    assert snap.severity_counts["high"] == 1
    assert snap.severity_counts["low"] == 1
    assert len(snap.recent_cases) == 2


def test_compute_snapshot_handles_missing_db(tmp_path: Path) -> None:
    snap = compute_snapshot(tmp_path / "missing.sqlite")
    assert snap.total_cases == 0
    assert snap.recent_cases == []


def test_compute_snapshot_avg_severity(tmp_path: Path) -> None:
    db = _seed_db(tmp_path)
    snap = compute_snapshot(db)
    assert snap.avg_severity == 40.0  # (60 + 20) / 2


def test_worst_recent_returns_highest_severity(tmp_path: Path) -> None:
    db = _seed_db(tmp_path)
    rows = worst_recent(db, limit=1)
    assert len(rows) == 1
    assert rows[0]["severity_score"] == 60


def test_worst_recent_handles_missing_db(tmp_path: Path) -> None:
    assert worst_recent(tmp_path / "missing.sqlite") == []


def test_compute_snapshot_reports_and_feedback(tmp_path: Path) -> None:
    db = _seed_db(tmp_path)
    case = repo.save_case(db, "x.jpg", "image")
    summary = InvestigationSummary(
        source_name="x.jpg",
        source_type="image",
        template_text="a summary",
        ai_text="a summary",
        used_ai=False,
        model_name="template",
        generation_time_sec=0.0,
        evidence_snapshot={},
    )
    repo.save_summary(db, case, summary)
    repo.save_report(
        db,
        case,
        ReportData(
            title="Test Report",
            report_id="r-001",
            generated_at=datetime.now(),
            source_name="x.jpg",
            source_type="image",
            model_name="template",
            summary_text="a summary",
            severity_score=10,
            severity_level="low",
            suggested_category="suspicious_activity",
            has_threat=False,
            counts_by_label={"person": 1},
            total_objects=1,
            person_count=1,
            weapon_count=0,
            vehicle_count=0,
            bag_count=0,
            key_observations=[],
            average_confidence=0.5,
            remarks="",
            app_version="1.0.0",
        ),
    )
    snap = compute_snapshot(db)
    assert snap.total_reports == 1
    assert snap.total_cases == 3

# ----------------------------------------------------------------------
# Phase 20 — moderate-bucket regression
# ----------------------------------------------------------------------


def test_severity_counts_use_moderate_label(tmp_path: Path) -> None:
    """An analysis with severity_score=30 must land in 'moderate', not 'medium'.

    Prior to Phase 20 the analytics snapshot hardcoded the dict
    ``{"low": 0, "medium": 0, "high": 0, "critical": 0}``, which put
    'moderate' counts under 'medium' — a silent label drift. The dict
    is now derived from SEVERITY_BANDS so it stays in sync with the
    canonical labels in config.SEVERITY_THRESHOLDS.
    """
    db = tmp_path / "moderate.sqlite"
    init_db(db)
    case = repo.save_case(db, "mid.png", "image")
    repo.save_analysis(db, case, _make_analysis("mid.png", 30, "moderate", "general", False))
    snap = compute_snapshot(db)
    assert snap.severity_counts.get("moderate") == 1
    # The legacy "medium" key must NOT be present (no drift allowed).
    assert "medium" not in snap.severity_counts
    # All four canonical labels must be initialised to 0 even when absent.
    for label in ("low", "moderate", "high", "critical"):
        assert label in snap.severity_counts


def test_severity_counts_all_buckets_present_when_empty(tmp_path: Path) -> None:
    """An empty-but-present DB returns 0 counts for every canonical band."""
    db = tmp_path / "empty.sqlite"
    init_db(db)
    snap = compute_snapshot(db)
    # No analyses yet, but the dict must include every canonical label.
    assert set(snap.severity_counts.keys()) == {"low", "moderate", "high", "critical"}
    assert all(v == 0 for v in snap.severity_counts.values())


def test_severity_counts_aggregate_across_cases(tmp_path: Path) -> None:
    """Counts from multiple analyses aggregate into the right buckets."""
    db = tmp_path / "multi.sqlite"
    init_db(db)
    cases = [
        ("low_01.png", 10, "low", False),
        ("moderate_01.png", 30, "moderate", False),
        ("moderate_02.png", 45, "moderate", True),
        ("high_01.png", 60, "high", True),
        ("critical_01.png", 90, "critical", True),
    ]
    for name, score, level, threat in cases:
        case = repo.save_case(db, name, "image")
        repo.save_analysis(db, case, _make_analysis(name, score, level, "general", threat))
    snap = compute_snapshot(db)
    assert snap.severity_counts["low"] == 1
    assert snap.severity_counts["moderate"] == 2
    assert snap.severity_counts["high"] == 1
    assert snap.severity_counts["critical"] == 1
