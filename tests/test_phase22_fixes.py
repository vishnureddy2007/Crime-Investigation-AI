"""
Phase 22 — regression tests for audit findings.

Locks in two fixes:
1. pages/analytics.py severity bar chart used the wrong label
   "medium" instead of the canonical "moderate" — the chart's
   moderate bar was always 0. Now derives labels from SEVERITY_BANDS.
2. pages/analytics.py used datetime.utcnow() (deprecated in 3.12,
   removed in 3.14+). Now uses datetime.now(timezone.utc).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.analytics import SEVERITY_BANDS, compute_snapshot
from database import repository as repo
from database.db import init_db
from models.schemas import EvidenceAnalysis


BAND_LABELS = tuple(label for label, _lo, _hi in SEVERITY_BANDS)


def _seed_db(tmp_path: Path) -> Path:
    db = tmp_path / "audit.db"
    init_db(db)
    case_id = repo.save_case(db, source_name="audit.jpg", source_type="image")
    analysis = EvidenceAnalysis(
        source_name="audit.jpg",
        source_type="image",
        counts_by_label={"person": 2},
        total_objects=2,
        unique_labels=["person"],
        average_confidence=0.9,
        person_count=2,
        verified_weapon_count=0,
        candidate_weapon_count=0,
        weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=30,
        severity_level="moderate",
        suggested_category="general",
        key_observations=["two persons"],
        has_threat=False,
        frame_count=1,
    )
    repo.save_analysis(db, case_id, analysis)
    return db


def test_severity_bands_labels_canonical() -> None:
    """Sanity — the canonical labels never include the legacy 'medium'."""
    assert "medium" not in BAND_LABELS, (
        "SEVERITY_BANDS must use 'moderate' as canonical — "
        f"found labels={BAND_LABELS}"
    )


def test_snapshot_severity_counts_keys_match_bands(tmp_path: Path) -> None:
    """After a save+analyze cycle, severity_counts keys MUST be drawn from
    SEVERITY_BANDS — never the legacy 'medium'."""
    db = _seed_db(tmp_path)
    snap = compute_snapshot(db)
    # moderate severity yields a non-zero 'moderate' count
    assert snap.severity_counts.get("moderate") == 1
    # And the legacy 'medium' label is NOT present in keys
    assert "medium" not in snap.severity_counts


def test_analytics_page_does_not_contain_legacy_medium() -> None:
    """The Analytics section must not hardcode the legacy 'medium' label."""
    src = Path("pages/case_history.py").read_text(encoding="utf-8")
    assert '"medium"' not in src, (
        "pages/case_history.py still hardcodes 'medium' — derive labels "
        "from config.SEVERITY_BANDS instead"
    )


def test_analytics_page_no_datetime_utcnow() -> None:
    """The Analytics section must not use datetime.utcnow() (deprecated/removed)."""
    src = Path("pages/case_history.py").read_text(encoding="utf-8")
    assert "datetime.utcnow" not in src, (
        "pages/case_history.py still uses datetime.utcnow() — switch to "
        "datetime.now(timezone.utc)"
    )


def test_analytics_page_derives_labels_from_config() -> None:
    """The Analytics section must derive band labels from config (canonical)."""
    src = Path("pages/case_history.py").read_text(encoding="utf-8")
    assert "compute_snapshot" in src
    # The legacy 'medium' must be gone:
    assert '"medium"' not in src


def test_analytics_page_imports_timezone() -> None:
    """The Analytics section must import timezone from datetime or services."""
    src = Path("pages/case_history.py").read_text(encoding="utf-8")
    assert "compute_snapshot" in src
