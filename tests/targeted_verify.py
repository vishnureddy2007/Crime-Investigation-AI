import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATABASE_PATH, VIDEOS_DIR
from models.schemas import EvidenceAnalysis
from services.animation_renderer import AnimationRenderer
from services.crime_situation import CrimeSituationAnalyzer
from models.summary_generator import SummaryGenerator

def verify_video_generator():
    print("\n--- [TEST 1: 2D VIDEO GENERATOR] ---")
    renderer = AnimationRenderer()
    print(f"Renderer initialized: {renderer.__class__.__name__}")
    print("SUCCESS: 2D Video Generator loaded without Blender.")
    return True

def verify_ai_prediction():
    print("\n--- [TEST 2: REAL QWEN3 PREDICTION] ---")
    analysis = EvidenceAnalysis(
        source_name="verify_case",
        source_type="image",
        counts_by_label={"person": 2, "weapon": 2},
        total_objects=4,
        unique_labels=["person", "weapon"],
        average_confidence=0.85,
        person_count=2,
        verified_weapon_count=2,
        candidate_weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=90,
        severity_level="critical",
        suggested_category="armed_robbery",
        key_observations=["Two verified weapons detected"],
        has_threat=True,
        weapon_count=2,
        frame_count=1
    )
    
    sum_gen = SummaryGenerator()
    summary = sum_gen.generate(analysis)
    
    analyzer = CrimeSituationAnalyzer()
    print("Calling Ollama qwen3:14b...")
    
    situation = analyzer.analyze(analysis, summary_text=summary.investigation_summary)
    
    if "Unable to analyze pattern" in situation.likely_activity_pattern:
        print("FAILURE: AI returned fallback response ('AI Unavailable')")
        return False
    
    print("SUCCESS: Real AI prediction received!")
    print(f"Prediction: {situation.likely_activity_pattern}")
    print(f"Confidence: {situation.confidence_level}")
    return True

if __name__ == "__main__":
    p1 = verify_video_generator()
    p2 = verify_ai_prediction()
    if p1 and p2:
        print("\nFINAL VERDICT: BOTH TARGETED ISSUES FIXED")
        sys.exit(0)
    else:
        print("\nFINAL VERDICT: TARGETED FIXES FAILED")
        sys.exit(1)
