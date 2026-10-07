"""Tests for the regeneration marking (outdated) logic.

Scenario:
1. Case created.
2. Analysis run -> Summary generated (is_outdated=0).
3. Another analysis run -> Summary should be is_outdated=1.
4. Human review added -> Summary should be is_outdated=1.
5. New summary generated -> is_outdated=0.
"""

from __future__ import annotations

import datetime
import pytest
from pathlib import Path

from database.db import init_db
from database.repository import (
    load_report,
    load_storyboard,
    load_summary,
    save_analysis,
    save_case,
    save_human_review,
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

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test_outdated.db"
    init_db(path)
    return path

@pytest.fixture
def sample_analysis() -> EvidenceAnalysis:
    return EvidenceAnalysis(
        source_name="test.mp4",
        source_type="video",
        counts_by_label={"person": 1},
        total_objects=1,
        unique_labels=["person"],
        average_confidence=0.9,
        person_count=1,
        verified_weapon_count=0,
        candidate_weapon_count=0,
        weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=10,
        severity_level="low",
        suggested_category="suspicious_activity",
        key_observations=["One person detected"],
        has_threat=False,
        frame_count=1,
    )

@pytest.fixture
def sample_summary() -> InvestigationSummary:
    return InvestigationSummary(
        source_name="test.mp4",
        source_type="video",
        template_text="Template summary",
        ai_text="AI summary",
        used_ai=True,
        model_name="qwen3:14b",
        generation_time_sec=1.0,
        evidence_snapshot={},
        timestamp=datetime.datetime.now(),
    )

def test_outdated_marking_flow(db_path, sample_analysis, sample_summary):
    # 1. Setup case and first analysis
    cid = save_case(db_path, "test.mp4", "video")
    aid1 = save_analysis(db_path, cid, sample_analysis)

    # 2. Generate summary (starts as not outdated)
    sid = save_summary(db_path, cid, sample_summary)
    s_data = load_summary(db_path, sid)
    assert s_data["is_outdated"] is False

    # 3. Run another analysis -> mark outdated
    save_analysis(db_path, cid, sample_analysis)
    s_data = load_summary(db_path, sid)
    assert s_data["is_outdated"] is True

    # 4. Save new summary -> reset outdated
    sid2 = save_summary(db_path, cid, sample_summary)
    s_data2 = load_summary(db_path, sid2)
    assert s_data2["is_outdated"] is False

    # 5. Add human review -> mark outdated
    save_human_review(
        db_path, cid, "person@0.9:frame_1", "person", 0.9, "CONFIRM"
    )
    s_data2 = load_summary(db_path, sid2)
    assert s_data2["is_outdated"] is True

def test_outdated_marking_across_artifacts(db_path, sample_analysis, sample_summary):
    cid = save_case(db_path, "test.mp4", "video")
    save_analysis(db_path, cid, sample_analysis)

    # Create summary, report, and storyboard
    sid = save_summary(db_path, cid, sample_summary)
    rid = save_report(db_path, cid, type("RD", (), {
        "source_name": "test.mp4",
        "source_type": "video",
        "report_id": "R1",
        "generated_at": None,
        "model_name": "m",
        "summary_text": "s",
        "severity_score": 10,
        "severity_level": "low",
        "suggested_category": "cat",
        "has_threat": False,
        "counts_by_label": {},
        "total_objects": 1,
        "person_count": 1,
        "weapon_count": 0,
        "vehicle_count": 0,
        "bag_count": 0,
        "key_observations": [],
        "average_confidence": 0.9,
        "remarks": "r",
        "app_version": "1.0",
        "as_dict": lambda self: {}
    })())
    # Storyboard is complex, just a mock
    sb = type("SB", (), {
        "source_name": "test.mp4",
        "source_type": "video",
        "scenes": [],
        "timestamp": None,
        "scene_count": 0,
        "total_duration_sec": 0.0,
        "as_dict": lambda self: {}
    })()
    stid = save_storyboard(db_path, cid, sb)

    # All should be NOT outdated initially
    assert load_summary(db_path, sid)["is_outdated"] is False
    assert load_report(db_path, rid)["is_outdated"] is False
    assert load_storyboard(db_path, stid)["is_outdated"] is False

    # Trigger outdated
    save_analysis(db_path, cid, sample_analysis)

    # All should be outdated now
    assert load_summary(db_path, sid)["is_outdated"] is True
    assert load_report(db_path, rid)["is_outdated"] is True
    assert load_storyboard(db_path, stid)["is_outdated"] is True
