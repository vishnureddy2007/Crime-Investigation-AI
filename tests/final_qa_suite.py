import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATABASE_PATH, VIDEOS_DIR
from database.repository import save_case, save_analysis, save_human_review, load_analysis, list_animations_for_case, load_animation, mark_case_outdated
from models.schemas import EvidenceAnalysis
from models.summary_generator import SummaryGenerator
from services.crime_situation import CrimeSituationAnalyzer
from services.storyboard_bridge import StoryboardGenerator
from services.animation_renderer import AnimationRenderer

def test_golden_path():
    print("\n--- STARTING GOLDEN PATH TEST ---")
    
    # 1. Case Creation
    case_id = save_case(DATABASE_PATH, "test_case_qa", "image")
    print(f"Case created: {case_id}")

    # 2. Mock Analysis (Verified Weapon + Candidate Weapon)
    # Note: Added missing required arguments based on models/schemas.py
    analysis = EvidenceAnalysis(
        source_name="test_case_qa",
        source_type="image",
        counts_by_label={"person": 1, "weapon": 1, "candidate_weapon": 1},
        total_objects=3,
        unique_labels=["person", "weapon", "candidate_weapon"],
        average_confidence=0.7,
        person_count=1,
        verified_weapon_count=1,
        candidate_weapon_count=1,
        vehicle_count=0,
        bag_count=0,
        severity_score=80,
        severity_level="critical",
        suggested_category="robbery",
        key_observations=["Verified weapon detected", "Person present"],
        has_threat=True,
        weapon_count=1,
        frame_count=1
    )
    save_analysis(DATABASE_PATH, case_id, analysis)
    print("Analysis saved (1 Verified Weapon, 1 Candidate)")

    # 3. Human Verification
    save_human_review(DATABASE_PATH, case_id, "weapon_1", "weapon", 0.9, "CONFIRM")
    save_human_review(DATABASE_PATH, case_id, "weapon_2", "weapon", 0.3, "REJECT")
    print("Human reviews applied: Weapon 1 CONFIRMED, Weapon 2 REJECTED")

    # 4. AI Analysis
    gen = SummaryGenerator()
    summary = gen.generate(analysis)
    print("AI Summary generated")

    sit_analyzer = CrimeSituationAnalyzer()
    situation = sit_analyzer.analyze(analysis, summary_text=summary.investigation_summary)
    print("AI Predicted Situation generated")

    # 5. Storyboard Generation
    sb_gen = StoryboardGenerator()
    plan = sb_gen.generate_plan(case_id, analysis, summary, situation)
    print(f"Storyboard JSON created. Objects in plan: {len(plan.objects)}")
    
    # Forensic Check: Ensure rejected weapon (weapon_2) is NOT in the plan
    obj_ids = [o["id"] for o in plan.objects]
    if "weapon_2" in obj_ids:
        print("FAILURE: Rejected weapon found in storyboard!")
        return False
    print("Forensic Check: Rejected weapon successfully excluded from plan")

    # 6. Rendering
    renderer = AnimationRenderer()
    output_video = VIDEOS_DIR / f"qa_test_{case_id}.mp4"
    
    print("Launching Blender & FFmpeg pipeline...")
    try:
        renderer.render_animated_video(None, analysis, output_video) 
        print("Rendering complete")
    except Exception as e:
        print(f"Rendering failed: {e}")
        return False

    # 7. Video Validation
    if not output_video.exists():
        print("FAILURE: MP4 file not created")
        return False
    
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(output_video)],
        capture_output=True, text=True
    )
    try:
        duration = float(res.stdout.strip())
        if duration <= 0:
            print(f"FAILURE: Invalid video duration: {duration}")
            return False
        print(f"Video validated: Duration {duration:.2f}s, File exists.")
    except ValueError:
        print("FAILURE: Could not parse duration from ffprobe")
        return False

    return True

if __name__ == "__main__":
    success = test_golden_path()
    sys.exit(0 if success else 1)
