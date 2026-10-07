"""The 'Gold Master' test. Exercises the entire system pipeline
from Case Creation to 3D Reconstruction.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from datetime import datetime
from database.db import init_db
from database.repository import (
    save_case, save_analysis, save_summary, save_report, save_storyboard,
    load_analysis, load_summary, load_report, load_storyboard,
    get_human_review_stats, save_human_review
)
from models.schemas import (
    EvidenceAnalysis, InvestigationSummary, ReportData, Storyboard, StoryboardScene
)
from models.report_generator import build_report_data
from models.scene_planner import plan_scenes, plan_scenes_from_narrative
from models.summary_generator import build_template_summary

def test_gold_master_workflow(tmp_path: Path):
    """
    The 'Gold Master' test. Exercises the entire system pipeline
    from Case Creation to 3D Reconstruction.
    """
    db_path = tmp_path / "gold_master.db"
    init_db(db_path)

    # 1. START CASE
    source_name = "gold_evidence.mp4"
    cid = save_case(db_path, source_name, "video")
    assert cid > 0

    # 2. ANALYZE EVIDENCE
    analysis = EvidenceAnalysis(
        source_name=source_name,
        source_type="video",
        counts_by_label={"person": 2, "weapon": 1},
        total_objects=3,
        unique_labels=["person", "weapon"],
        average_confidence=0.88,
        person_count=2,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        weapon_count=1,
        vehicle_count=0,
        bag_count=0,
        severity_score=80,
        severity_level="high",
        frame_count=10,
        suggested_category="robbery",
        key_observations=["Suspect holding firearm", "Accomplice present"],
        has_threat=True,
    )
    aid = save_analysis(db_path, cid, analysis)
    assert aid > 0

    # 3. HUMAN-IN-THE-LOOP VERIFICATION
    # Review 1: Confirm weapon
    save_human_review(db_path, cid, "weapon@0.9:f1", "weapon", 0.9, "CONFIRM")
    # Review 2: Reject accomplice as "not a person"
    save_human_review(db_path, cid, "person@0.7:f2", "person", 0.7, "REJECT")

    # Truth updated: 1 person, 1 weapon
    updated_analysis = EvidenceAnalysis(
        **{**analysis.__dict__, "person_count": 1, "total_objects": 2}
    )

    # 4. AI SUMMARY GENERATION
    summary_text = build_template_summary(updated_analysis)
    summary = InvestigationSummary(
        source_name=source_name,
        source_type="video",
        template_text=summary_text,
        ai_text=summary_text,
        used_ai=False,
        model_name="template",
        generation_time_sec=0.1,
        evidence_snapshot=updated_analysis.__dict__,
    )
    sid = save_summary(db_path, cid, summary)
    assert sid > 0

    # 5. PROFESSIONAL REPORT GENERATION
    h_stats = get_human_review_stats(db_path, cid)
    report_data = build_report_data(
        analysis=updated_analysis,
        summary=summary,
        human_stats=h_stats
    )
    rid = save_report(db_path, cid, report_data)

    # Verify report data contains human stats
    loaded_report = load_report(db_path, rid)
    assert loaded_report["human_confirmed_count"] == 1
    assert loaded_report["human_rejected_count"] == 1
    assert loaded_report["person_count"] == 1

    # 6. 3D RECONSTRUCTION PLANNING
    # Test AI-driven planning fallback (template)
    storyboard = plan_scenes_from_narrative(summary, updated_analysis)
    stid = save_storyboard(db_path, cid, storyboard)

    loaded_sb = load_storyboard(db_path, stid)
    assert loaded_sb["source_name"] == source_name
    assert len(storyboard.scenes) >= 4

    # 7. STATE MANAGEMENT (OUTDATED)
    # Save new analysis -> markers should trigger
    save_analysis(db_path, cid, updated_analysis)
    assert load_summary(db_path, sid)["is_outdated"] is True
    assert load_report(db_path, rid)["is_outdated"] is True
    assert load_storyboard(db_path, stid)["is_outdated"] is True
