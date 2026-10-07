"""
Tests for Milestone 9 — SQLite schema bootstrap + repository CRUD.

No Streamlit imports anywhere — every test drives the repository
directly with a `tmp_path` so the data layer stays fast and
deterministic.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

from database.db import SCHEMA_SQL, get_connection, init_db
from database.repository import (
    case_exists,
    latest_case_for_source,
    list_analyses_for_case,
    list_cases,
    list_reports_for_case,
    list_storyboards_for_case,
    list_summaries_for_case,
    load_analysis,
    load_report,
    load_storyboard,
    load_summary,
    save_analysis,
    save_case,
    save_report,
    save_storyboard,
    save_summary,
)
from models.schemas import (
    EvidenceAnalysis,
    InvestigationSummary,
    ReportData,
    Storyboard,
)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


def _make_analysis() -> EvidenceAnalysis:
    return EvidenceAnalysis(
        source_name="scene.jpg", source_type="image",
        counts_by_label={"person": 2, "knife": 1},
        total_objects=3, unique_labels=["person", "knife"],
        average_confidence=0.82,
        person_count=2, verified_weapon_count=1,
        candidate_weapon_count=0, weapon_count=1,
        vehicle_count=0, bag_count=0,
        severity_score=55, severity_level="high",
        suggested_category="assault",
        key_observations=["2 persons detected", "Weapon present: 1 knife."],
        has_threat=True, frame_count=1, timestamp=datetime.now(),
    )


def _make_summary() -> InvestigationSummary:
    return InvestigationSummary(
        source_name="scene.jpg", source_type="image",
        template_text="Template summary.",
        ai_text="AI summary.",
        used_ai=True, model_name="google/flan-t5-base",
        generation_time_sec=0.42,
        evidence_snapshot={"person_count": 2, "weapon_count": 1},
        timestamp=datetime.now(),
    )


def _make_report() -> ReportData:
    return ReportData(
        title="AI Crime Investigation Report",
        report_id="RPT-20260101-120000",
        generated_at=datetime.now(),
        source_name="scene.jpg", source_type="image",
        model_name="google/flan-t5-base",
        summary_text="Summary text.",
        severity_score=55, severity_level="high",
        suggested_category="assault", has_threat=True,
        counts_by_label={"person": 2, "knife": 1},
        total_objects=3, person_count=2, weapon_count=1,
        vehicle_count=0, bag_count=0,
        key_observations=["obs1", "obs2"],
        average_confidence=0.82,
        remarks="For academic use only.",
        app_version="0.9.0",
    )


def _make_storyboard() -> Storyboard:
    return Storyboard(
        source_name="scene.jpg", source_type="image",
        scenes=[], timestamp=datetime.now(),
    )


# ----------------------------------------------------------------------
# Schema bootstrap
# ----------------------------------------------------------------------
def test_init_db_creates_tables(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert {"cases", "analyses", "summaries", "reports", "storyboards"}.issubset(names)


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "idem.db"
    init_db(db)
    init_db(db)  # must not raise
    assert db.exists()


def test_get_connection_enables_foreign_keys(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    with get_connection(db) as conn:
        cur = conn.execute("PRAGMA foreign_keys")
        assert cur.fetchone()[0] == 1


def test_schema_sql_has_all_tables() -> None:
    """Pin the SCHEMA_SQL string so missing-table regressions break loudly."""
    for table in ("cases", "analyses", "summaries", "reports", "storyboards"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in SCHEMA_SQL


# ----------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------
def test_save_case_returns_int_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    assert isinstance(case_id, int) and case_id > 0


def test_list_cases_returns_empty_initially(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    assert list_cases(db) == []


def test_list_cases_orders_by_created_at_desc(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    c1 = save_case(db, "a.jpg", "image")
    time.sleep(0.02)
    c2 = save_case(db, "b.jpg", "image")
    time.sleep(0.02)
    c3 = save_case(db, "c.jpg", "image")
    rows = list_cases(db)
    ids = [r["case_id"] for r in rows]
    assert ids == [c3, c2, c1]


def test_latest_case_for_source_returns_most_recent(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    save_case(db, "same.jpg", "image")
    time.sleep(0.02)
    new_id = save_case(db, "same.jpg", "video")
    assert latest_case_for_source(db, "same.jpg") == new_id


def test_latest_case_for_source_returns_none_when_no_match(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    assert latest_case_for_source(db, "nope.jpg") is None


def test_case_exists(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    assert case_exists(db, case_id) is True
    assert case_exists(db, 9999) is False


# ----------------------------------------------------------------------
# analyses
# ----------------------------------------------------------------------
def test_save_and_load_analysis_roundtrip(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    analysis = _make_analysis()
    aid = save_analysis(db, case_id, analysis)
    payload = load_analysis(db, aid)
    assert payload is not None
    # As_dict JSON roundtrips every field. Spot-check a few.
    assert payload["source_name"] == "scene.jpg"
    assert payload["severity_score"] == 55
    assert payload["has_threat"] is True
    assert payload["counts_by_label"] == {"person": 2, "knife": 1}


def test_save_analysis_links_to_case(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    aid = save_analysis(db, case_id, _make_analysis())
    with get_connection(db) as conn:
        row = conn.execute(
            "SELECT case_id FROM analyses WHERE analysis_id = ?", (aid,),
        ).fetchone()
    assert row["case_id"] == case_id


def test_save_analysis_persists_denormalized_columns(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    aid = save_analysis(db, case_id, _make_analysis())
    with get_connection(db) as conn:
        row = conn.execute(
            "SELECT severity_score, severity_level, suggested_category, "
            "has_threat FROM analyses WHERE analysis_id = ?", (aid,),
        ).fetchone()
    assert row["severity_score"] == 55
    assert row["severity_level"] == "high"
    assert row["suggested_category"] == "assault"
    assert row["has_threat"] == 1


def test_load_analysis_returns_none_for_missing_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    assert load_analysis(db, 9999) is None


def test_list_analyses_for_case_orders_desc(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    save_analysis(db, case_id, _make_analysis())
    time.sleep(0.02)
    save_analysis(db, case_id, _make_analysis())
    rows = list_analyses_for_case(db, case_id)
    assert len(rows) == 2
    # The newer row comes first
    assert rows[0]["created_at"] >= rows[1]["created_at"]


def test_cascade_delete_removes_children(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    aid = save_analysis(db, case_id, _make_analysis())
    # Delete the parent case directly via SQL
    with get_connection(db) as conn:
        conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
    # The analysis row should be gone (ON DELETE CASCADE)
    assert load_analysis(db, aid) is None


# ----------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------
def test_save_and_load_summary_roundtrip(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    sid = save_summary(db, case_id, _make_summary())
    payload = load_summary(db, sid)
    assert payload is not None
    assert payload["model_name"] == "google/flan-t5-base"
    assert payload["used_ai"] is True


def test_save_summary_persists_denormalized(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    sid = save_summary(db, case_id, _make_summary())
    with get_connection(db) as conn:
        row = conn.execute(
            "SELECT model_name, used_ai FROM summaries WHERE summary_id = ?", (sid,),
        ).fetchone()
    assert row["model_name"] == "google/flan-t5-base"
    assert row["used_ai"] == 1


def test_list_summaries_for_case(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    save_summary(db, case_id, _make_summary())
    rows = list_summaries_for_case(db, case_id)
    assert len(rows) == 1
    assert rows[0]["model_name"] == "google/flan-t5-base"


def test_load_summary_returns_none_for_missing_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    assert load_summary(db, 9999) is None


# ----------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------
def test_save_and_load_report_roundtrip(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    rid = save_report(db, case_id, _make_report())
    payload = load_report(db, rid)
    assert payload is not None
    assert payload["report_id"] == "RPT-20260101-120000"
    assert payload["app_version"] == "0.9.0"


def test_save_report_includes_report_uid(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    rid = save_report(db, case_id, _make_report())
    with get_connection(db) as conn:
        row = conn.execute(
            "SELECT report_uid, app_version FROM reports WHERE report_id = ?",
            (rid,),
        ).fetchone()
    assert row["report_uid"] == "RPT-20260101-120000"
    assert row["app_version"] == "0.9.0"


def test_list_reports_for_case(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    save_report(db, case_id, _make_report())
    rows = list_reports_for_case(db, case_id)
    assert len(rows) == 1
    assert rows[0]["report_uid"] == "RPT-20260101-120000"


# ----------------------------------------------------------------------
# storyboards
# ----------------------------------------------------------------------
def test_save_and_load_storyboard_roundtrip(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    sid = save_storyboard(db, case_id, _make_storyboard())
    payload = load_storyboard(db, sid)
    assert payload is not None
    assert payload["source_name"] == "scene.jpg"
    assert payload["scenes"] == []


def test_save_storyboard_includes_scene_count(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    sb = _make_storyboard()
    sid = save_storyboard(db, case_id, sb)
    with get_connection(db) as conn:
        row = conn.execute(
            "SELECT scene_count, total_duration_sec FROM storyboards "
            "WHERE storyboard_id = ?", (sid,),
        ).fetchone()
    assert row["scene_count"] == 0
    assert row["total_duration_sec"] == 0.0


def test_list_storyboards_for_case(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    save_storyboard(db, case_id, _make_storyboard())
    rows = list_storyboards_for_case(db, case_id)
    assert len(rows) == 1
    assert rows[0]["scene_count"] == 0


# ----------------------------------------------------------------------
# list_cases + cross-table integration
# ----------------------------------------------------------------------
def test_list_cases_includes_latest_severity_from_joined_analysis(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    save_analysis(db, case_id, _make_analysis())
    rows = list_cases(db)
    assert len(rows) == 1
    assert rows[0]["latest_severity_level"] == "high"
    assert rows[0]["latest_severity_score"] == 55


def test_full_pipeline_save_then_list_all(tmp_path: Path) -> None:
    """End-to-end: save case + 4 artifacts, then list everything."""
    db = _make_db(tmp_path)
    case_id = save_case(db, "x.jpg", "image")
    save_analysis(db,   case_id, _make_analysis())
    save_summary(db,    case_id, _make_summary())
    save_report(db,     case_id, _make_report())
    save_storyboard(db, case_id, _make_storyboard())

    assert list_cases(db)[0]["case_id"] == case_id
    assert list_analyses_for_case(db, case_id)
    assert list_summaries_for_case(db, case_id)
    assert list_reports_for_case(db, case_id)
    assert list_storyboards_for_case(db, case_id)


# ----------------------------------------------------------------------
# Phase 17 regressions — list_cases joined with latest analysis
# ----------------------------------------------------------------------


def test_list_cases_picks_latest_analysis_per_case(tmp_path: Path) -> None:
    """When a case has multiple analyses, list_cases returns the latest."""
    db = _make_db(tmp_path)
    case_id = save_case(db, "multi.jpg", "image")
    # First analysis: moderate.
    a1 = EvidenceAnalysis(
        source_name="multi.jpg", source_type="image",
        counts_by_label={"person": 1}, total_objects=1,
        unique_labels=["person"], average_confidence=0.6,
        person_count=1, verified_weapon_count=0,
        candidate_weapon_count=0, weapon_count=0,
        vehicle_count=0, bag_count=0,
        severity_score=40, severity_level="moderate",
        suggested_category="general", has_threat=False,
        key_observations=("nothing notable",),
        frame_count=1,
    )
    save_analysis(db, case_id, a1)
    time.sleep(0.01)  # ensure distinct created_at
    # Second analysis: critical — must be picked by the window function.
    a2 = EvidenceAnalysis(
        source_name="multi.jpg", source_type="image",
        counts_by_label={"gun": 1}, total_objects=1,
        unique_labels=["gun"], average_confidence=0.9,
        person_count=0, verified_weapon_count=1,
        candidate_weapon_count=0, weapon_count=1,
        vehicle_count=0, bag_count=0,
        severity_score=80, severity_level="critical",
        suggested_category="assault", has_threat=True,
        key_observations=("weapon detected",),
        frame_count=1,
    )
    save_analysis(db, case_id, a2)
    rows = list_cases(db)
    assert len(rows) == 1
    assert rows[0]["latest_severity_level"] == "critical"
    assert rows[0]["latest_severity_score"] == 80


def test_list_cases_returns_none_for_cases_without_analyses(tmp_path: Path) -> None:
    """A case with no analyses should still appear, with NULL columns."""
    db = _make_db(tmp_path)
    save_case(db, "bare.jpg", "image")
    rows = list_cases(db)
    assert len(rows) == 1
    assert rows[0]["latest_severity_level"] is None
    assert rows[0]["latest_severity_score"] is None


# ----------------------------------------------------------------------
# Phase 19 — get_connection rollback + _row_to_dict
# ----------------------------------------------------------------------


def test_get_connection_rolls_back_on_error(tmp_path: Path) -> None:
    """If a query raises inside the `with` block, rollback runs before re-raise."""
    from database.db import get_connection

    db = _make_db(tmp_path)
    with pytest.raises(RuntimeError, match="forced"):
        with get_connection(db) as conn:
            # Force a failure inside the context manager.
            conn.execute("CREATE TABLE _x (id INT)")
            conn.execute("INSERT INTO _x VALUES (1)")
            raise RuntimeError("forced")
    # The uncommitted insert must have been rolled back.
    with get_connection(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_x'"
        ).fetchall()
        # Table exists (it was created before the failure) but the row does not.
        assert len(rows) == 1
        count = conn.execute("SELECT COUNT(*) AS c FROM _x").fetchone()
        assert count["c"] == 0


def test_row_to_dict_handles_none_and_row() -> None:
    """_row_to_dict returns None for None input, dict for sqlite3.Row."""
    from database.db import _row_to_dict
    import sqlite3

    # None in → None out.
    assert _row_to_dict(None) is None

    # sqlite3.Row in → dict out.
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT 1 AS a, 'x' AS b").fetchone()
    out = _row_to_dict(row)
    assert out == {"a": 1, "b": "x"}
    conn.close()
