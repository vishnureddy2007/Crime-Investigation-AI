"""
End-to-End Pipeline Verification for Crime Investigation AI.

This script simulates the entire investigation workflow from case creation
to 3D video generation, verifying data contracts and the outdated-state
management system.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

from config import DATABASE_PATH
from database.db import init_db
from database.repository import (
    save_case, save_analysis, save_summary, save_report,
    save_storyboard, load_summary, load_report, load_storyboard,
    save_human_review, mark_case_outdated
)
from models.schemas import (
    EvidenceAnalysis, InvestigationSummary, ReportData,
    Storyboard, StoryboardScene, Detection, BoundingBox
)
from models.summary_generator import SummaryGenerator
from services.crime_situation import CrimeSituationAnalyzer
from services.scene_planner_service import ScenePlanner

def test_full_forensic_pipeline(tmp_path):
    # 1. Setup temporary DB
    db_path = tmp_path / "test_e2e.db"
    init_db(db_path)

    # 2. Create Case
    case_id = save_case(db_path, "CrimeScene_001.jpg", "image")

    # 3. Simulate Evidence Analysis
    # Case: 1 person, 1 verified weapon, 1 candidate weapon, 1 bag
    analysis = EvidenceAnalysis(
        source_name="CrimeScene_001.jpg",
        source_type="image",
        counts_by_label={"person": 1, "weapon": 1, "candidate_weapon": 1, "bag": 1},
        total_objects=4,
        unique_labels=["person", "weapon", "candidate_weapon", "bag"],
        average_confidence=0.75,
        person_count=1,
        verified_weapon_count=1,
        candidate_weapon_count=1,
        weapon_count=1,
        vehicle_count=0,
        bag_count=1,
        severity_score=80,
        severity_level="critical",
        suggested_category="robbery",
        key_observations=["Verified handgun detected", "Possible weapon candidate found"],
        has_threat=True,
        frame_count=1,
        timestamp=datetime.now()
    )
    save_analysis(db_path, case_id, analysis)

    # 4. Generate Professional Narrative Summary
    gen = SummaryGenerator() # Will use template as Ollama might not be in CI
    detailed_summary = gen.generate(analysis)
    
    # Convert DetailedNarrativeSummary to InvestigationSummary for database storage
    summary = InvestigationSummary(
        source_name=analysis.source_name,
        source_type=analysis.source_type,
        template_text=detailed_summary.investigation_summary,
        ai_text=detailed_summary.investigation_summary,
        used_ai=False,
        model_name="template",
        generation_time_sec=0.1,
        evidence_snapshot=analysis.as_dict(),
    )
    save_summary(db_path, case_id, summary)

    # 5. Perform Situation Analysis
    sit_analyzer = CrimeSituationAnalyzer()
    situation = sit_analyzer.analyze(analysis, summary_text=summary.primary_text)
    # (Situation analysis isn't saved in a dedicated table in the current schema,
    # usually held in session state or part of the report).

    # 6. Generate Final Report
    report = ReportData(
        title="Case 001 Final Report",
        report_id="REP-001",
        generated_at=datetime.now(),
        source_name="CrimeScene_001.jpg",
        source_type="image",
        model_name="yolov8n",
        summary_text=summary.primary_text,
        severity_score=80,
        severity_level="critical",
        suggested_category="robbery",
        has_threat=True,
        counts_by_label=analysis.counts_by_label,
        total_objects=4,
        person_count=1,
        weapon_count=1,
        vehicle_count=0,
        bag_count=1,
        key_observations=analysis.key_observations,
        average_confidence=0.75,
        remarks="Case closed.",
        app_version="1.0.0"
    )
    save_report(db_path, case_id, report)

    # 7. Generate 3D Scene and Storyboard
    planner = ScenePlanner()
    # Use a mock for situation as it's generated on the fly
    scene_narrative = planner.plan_structured_scene(summary, situation, analysis)
    storyboard = planner.generate_storyboard(scene_narrative, analysis)
    save_storyboard(db_path, case_id, storyboard)

    # --- VERIFICATION 1: Initial State ---
    # All results should be current (is_outdated = 0)
    sum_data = load_summary(db_path, 1) # first summary
    assert sum_data["is_outdated"] is False
    rep_data = load_report(db_path, 1)
    assert rep_data["is_outdated"] is False
    sb_data = load_storyboard(db_path, 1)
    assert sb_data["is_outdated"] is False

    # 8. Trigger Human Review (Evidence Modification)
    # Reject the candidate weapon
    save_human_review(
        db_path=db_path,
        case_id=case_id,
        detection_key="candidate_weapon@0.30",
        label="candidate_weapon",
        confidence=0.30,
        decision="REJECT",
        original_status="candidate"
    )

    # --- VERIFICATION 2: Outdated State Trigger ---
    # Everything should now be outdated
    sum_data_post = load_summary(db_path, 1)
    assert sum_data_post["is_outdated"] is True, "Summary should be outdated after review"
    rep_data_post = load_report(db_path, 1)
    assert rep_data_post["is_outdated"] is True, "Report should be outdated after review"
    sb_data_post = load_storyboard(db_path, 1)
    assert sb_data_post["is_outdated"] is True, "Storyboard should be outdated after review"

    print("\\n✅ End-to-End Pipeline Verification Passed!")
