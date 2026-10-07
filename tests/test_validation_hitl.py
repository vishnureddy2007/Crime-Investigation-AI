import pytest
from pathlib import Path
from database.db import init_db
from database.repository import (
    save_case, save_analysis, save_summary,
    load_summary, save_human_review
)
from models.schemas import EvidenceAnalysis, InvestigationSummary

def test_hitl_state_flow(tmp_path: Path):
    db_path = tmp_path / "hitl_val.db"
    init_db(db_path)

    # 1. Initial Setup
    cid = save_case(db_path, "hitl_test.mp4", "video")
    analysis = EvidenceAnalysis(
        source_name="hitl_test.mp4", source_type="video",
        counts_by_label={"weapon": 1}, total_objects=1, unique_labels=["weapon"],
        average_confidence=0.8, person_count=0, verified_weapon_count=1,
        candidate_weapon_count=0, weapon_count=1, vehicle_count=0, bag_count=0,
        severity_score=70, severity_level="high", suggested_category="robbery",
        key_observations=["Weapon detected"], has_threat=True, frame_count=1
    )
    save_analysis(db_path, cid, analysis)

    summary = InvestigationSummary(
        source_name="hitl_test.mp4", source_type="video",
        template_text="Baseline", ai_text="Baseline",
        used_ai=False, model_name="template", generation_time_sec=0.1,
        evidence_snapshot=analysis.as_dict()
    )
    sid = save_summary(db_path, cid, summary)

    # Verify: Summary is current
    assert load_summary(db_path, sid)["is_outdated"] is False

    # 2. Human Review Trigger
    save_human_review(db_path, cid, "weapon@0.8:f1", "weapon", 0.8, "REJECT")

    # Verify: Summary is now outdated
    assert load_summary(db_path, sid)["is_outdated"] is True

    # 3. Regeneration
    save_summary(db_path, cid, summary) # New summary record

    # The latest summary for this case should be current
    # Note: save_summary creates a NEW record, it doesn't update.
    # We need to check the most recent summary.
    from database.repository import list_summaries_for_case
    summaries = list_summaries_for_case(db_path, cid)
    latest_sid = summaries[0]["summary_id"]
    assert load_summary(db_path, latest_sid)["is_outdated"] is False

if __name__ == "__main__":
    # Simple manual run if not using pytest
    import sys
    try:
        # Using a fixed path for manual run to avoid tmp_path issues
        p = Path("test_hitl_manual.db")
        init_db(p)
        # (duplicate logic here or just use pytest)
        print("Manual test started...")
    except Exception as e:
        print(f"Error: {e}")
