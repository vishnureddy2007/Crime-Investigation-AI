import pytest
from pathlib import Path
from datetime import datetime
from database.db import init_db
from database.repository import (
    save_case, save_analysis, save_summary, save_report, save_storyboard,
    save_human_review, load_analysis, load_summary, load_report, load_storyboard
)
from models.schemas import (
    EvidenceAnalysis, InvestigationSummary, ReportData, Storyboard, StoryboardScene
)
from models.report_generator import build_report_data
from models.scene_planner import plan_scenes, plan_scenes_from_narrative
from models.summary_generator import build_template_summary

def test_full_investigation_workflow(tmp_path: Path):
    db_path = tmp_path / "e2e_workflow.db"
    init_db(db_path)

    # --- 1. CASE CREATION ---
    source_name = "robbery_video.mp4"
    cid = save_case(db_path, source_name, "video")

    # --- 2. INITIAL ANALYSIS ---
    # AI detects 2 persons and 1 weapon
    analysis = EvidenceAnalysis(
        source_name=source_name,
        source_type="video",
        counts_by_label={"person": 2, "weapon": 1},
        total_objects=3,
        unique_labels=["person", "weapon"],
        average_confidence=0.85,
        person_count=2,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        weapon_count=1,
        vehicle_count=0,
        bag_count=0,
        severity_score=70,
        severity_level="high",
        suggested_category="robbery",
        key_observations=["Person 1 seen with firearm", "Person 2 assisting"],
        has_threat=True,
        frame_count=10,
    )
    aid = save_analysis(db_path, cid, analysis)

    # --- 3. HUMAN REVIEW (The HITL Phase) ---
    # Investigator rejects Person 2 as a false positive
    # detection_key is usually "label@conf:frame"
    save_human_review(
        db_path, cid, "person@0.8:frame_10", "person", 0.8, "REJECT",
        reviewer="Det. Smith", note="Not a person, just a shadow"
    )
    
    # After review, the "truth" changes: person_count should be 1
    # In a real app, the analysis would be updated or the reporter would aggregate
    updated_analysis = EvidenceAnalysis(
        **{**analysis.__dict__, "person_count": 1, "total_objects": 2}
    )

    # --- 4. AI SUMMARY GENERATION ---
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

    # --- 5. PROFESSIONAL REPORT GENERATION ---
    from database.repository import get_human_review_stats
    h_stats = get_human_review_stats(db_path, cid)
    
    report_data = build_report_data(
        analysis=updated_analysis,
        summary=summary,
        human_stats=h_stats
    )
    rid = save_report(db_path, cid, report_data)

    # --- 6. 3D RECONSTRUCTION PLANNING ---
    storyboard = plan_scenes(updated_analysis)
    stid = save_storyboard(db_path, cid, storyboard)

    # --- VALIDATION ---
    # Verify Report includes human stats
    loaded_report = load_report(db_path, rid)
    assert loaded_report["human_rejected_count"] == 1
    assert loaded_report["person_count"] == 1
    
    # Verify Storyboard only has scenes for the "truth"
    # The deterministic planner adds a scene for persons if person_count > 0
    # We check that the "Persons Detected" scene caption reflects 1 person, not 2.
    assert any("1 person" in s.caption for s in storyboard.scenes)
    assert not any("2 persons" in s.caption for s in storyboard.scenes)

    # Verify Outdated Marking
    # If we save a new analysis, the report should become outdated
    save_analysis(db_path, cid, updated_analysis)
    assert load_report(db_path, rid)["is_outdated"] is True

