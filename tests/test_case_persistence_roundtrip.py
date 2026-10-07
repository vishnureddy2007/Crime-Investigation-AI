import pytest
from pathlib import Path
from database.db import init_db
from database.repository import (
    save_case, save_analysis, save_summary, save_report, save_storyboard,
    load_analysis, load_summary, load_report, load_storyboard
)
from models.schemas import (
    EvidenceAnalysis, InvestigationSummary, ReportData, Storyboard, StoryboardScene
)
from datetime import datetime

def test_case_persistence_roundtrip(tmp_path: Path):
    db_path = tmp_path / "roundtrip.db"
    init_db(db_path)

    # 1. Create Case
    cid = save_case(db_path, "evidence.mp4", "video")
    assert cid > 0

    # 2. Save Analysis
    analysis = EvidenceAnalysis(
        source_name="evidence.mp4",
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
        suggested_category="suspicious",
        key_observations=["Person seen"],
        has_threat=False,
        frame_count=1,
    )
    aid = save_analysis(db_path, cid, analysis)
    
    # 3. Save Summary
    summary = InvestigationSummary(
        source_name="evidence.mp4",
        source_type="video",
        template_text="Template",
        ai_text="AI Text",
        used_ai=True,
        model_name="qwen3",
        generation_time_sec=1.0,
        evidence_snapshot={},
    )
    sid = save_summary(db_path, cid, summary)

    # 4. Save Report
    report = ReportData(
        title="Title",
        report_id="R1",
        generated_at=datetime.now(),
        source_name="evidence.mp4",
        source_type="video",
        model_name="m",
        summary_text="s",
        severity_score=10,
        severity_level="low",
        suggested_category="cat",
        has_threat=False,
        counts_by_label={},
        total_objects=1,
        person_count=1,
        weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        key_observations=[],
        average_confidence=0.9,
        remarks="r",
        app_version="1.0",
    )
    rid = save_report(db_path, cid, report)

    # 5. Save Storyboard
    sb = Storyboard(
        source_name="evidence.mp4",
        source_type="video",
        scenes=[StoryboardScene(0, "S1", "C1", None, 1.0, False)],
    )
    stid = save_storyboard(db_path, cid, sb)

    # --- Round-trip Validation ---
    loaded_analysis = load_analysis(db_path, aid)
    assert loaded_analysis["source_name"] == "evidence.mp4"
    assert loaded_analysis["severity_score"] == 10

    loaded_summary = load_summary(db_path, sid)
    assert loaded_summary["ai_text"] == "AI Text"
    assert loaded_summary["is_outdated"] is False

    loaded_report = load_report(db_path, rid)
    assert loaded_report["report_id"] == "R1"
    assert loaded_report["is_outdated"] is False

    loaded_sb = load_storyboard(db_path, stid)
    assert loaded_sb["source_name"] == "evidence.mp4"
    assert loaded_sb["is_outdated"] is False

