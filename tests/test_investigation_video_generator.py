"""
Tests for InvestigationVideoGenerator pipeline without Blender.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from PIL import Image

from database.db import init_db
from database.repository import (
    get_case_evidence_version,
    get_latest_investigation_video,
    mark_case_outdated,
    save_analysis,
    save_case,
    save_summary,
)
from models.schemas import BoundingBox, Detection, EvidenceAnalysis, InvestigationSummary
from services.investigation_video_generator import InvestigationVideoGenerator


@pytest.fixture
def setup_test_db(tmp_path: Path):
    db_file = tmp_path / "test_video.db"
    init_db(db_file)

    case_id = save_case(db_file, "crime_scene_01.jpg", "IMAGE")

    # Create dummy image
    img_path = tmp_path / "crime_scene_01.jpg"
    img = Image.new("RGB", (640, 480), color=(100, 50, 50))
    img.save(img_path)

    det1 = Detection(class_name="person", label="person", confidence=0.92, bbox=BoundingBox(50, 50, 200, 300))
    det2 = Detection(class_name="handgun", label="handgun", confidence=0.88, bbox=BoundingBox(180, 120, 240, 180), weapon_status="verified")

    analysis = EvidenceAnalysis(
        source_name="crime_scene_01.jpg",
        source_type="IMAGE",
        counts_by_label={"person": 1, "handgun": 1},
        total_objects=2,
        unique_labels=["person", "handgun"],
        average_confidence=0.90,
        person_count=1,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=85,
        severity_level="HIGH",
        suggested_category="ARMED_INCIDENT",
        key_observations=["Detected person and verified weapon."],
        has_threat=True,
        weapon_count=1,
        frame_count=1,
    )
    save_analysis(db_file, case_id, analysis)

    from models.summary_generator import build_template_summary
    summary = build_template_summary(analysis)
    save_summary(db_file, case_id, summary)

    return db_file, case_id, img_path


def test_video_generation_without_blender(setup_test_db, tmp_path: Path):
    """Test generating a 2D explanation video without Blender."""
    db_file, case_id, img_path = setup_test_db

    generator = InvestigationVideoGenerator(db_path=db_file, fps=15, resolution=(640, 360))

    video_path, duration = generator.generate_video(
        case_id=case_id,
        output_dir=tmp_path / "videos",
        evidence_images=[img_path],
    )

    # Verification
    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert duration > 0.0
    assert video_path.suffix == ".mp4"

    # Database Check
    v_rec = get_latest_investigation_video(db_file, case_id)
    assert v_rec is not None
    assert v_rec["status"] == "READY"
    assert v_rec["evidence_version"] == get_case_evidence_version(db_file, case_id)


def test_video_caching_and_outdated_marking(setup_test_db, tmp_path: Path):
    """Test video caching and status change to OUTDATED on evidence update."""
    db_file, case_id, img_path = setup_test_db
    generator = InvestigationVideoGenerator(db_path=db_file, fps=15, resolution=(640, 360))

    # Generate initial video
    v1_path, _ = generator.generate_video(case_id=case_id, output_dir=tmp_path / "videos")

    # Second call should return cached video instantly
    v2_path, _ = generator.generate_video(case_id=case_id, output_dir=tmp_path / "videos", force=False)
    assert v1_path == v2_path

    # Update evidence/mark case outdated
    mark_case_outdated(db_file, case_id)

    v_rec = get_latest_investigation_video(db_file, case_id)
    assert v_rec["status"] == "OUTDATED"
