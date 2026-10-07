import sys
from pathlib import Path
import dataclasses

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.schemas import EvidenceAnalysis
from services.animation_renderer import AnimationRenderer
from datetime import datetime

def test_pipeline():
    print("Starting End-to-End Render Pipeline Test...")
    
    analysis = EvidenceAnalysis(
        source_name="Test_Case_001",
        source_type="image",
        counts_by_label={"person": 2, "weapon": 1},
        total_objects=3,
        unique_labels=["person", "weapon"],
        average_confidence=0.85,
        person_count=2,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=7,
        severity_level="High",
        suggested_category="Assault",
        key_observations=["Two persons present", "One verified weapon"],
        has_threat=True,
        weapon_count=1,
        frame_count=1,
        timestamp=datetime.now()
    )

    renderer = AnimationRenderer()
    output_video = Path("Crime-Investigation-AI/tests/animation_test_data/test_output.mp4")
    
    try:
        print("Executing render_animated_video...")
        result_path = renderer.render_animated_video(None, analysis, output_video)
        
        if result_path.exists():
            print(f"SUCCESS: Video rendered to {result_path}")
            import os
            print(f"File size: {os.path.getsize(result_path)} bytes")
        else:
            print("FAILED: Output file not found.")
            sys.exit(1)
            
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()
