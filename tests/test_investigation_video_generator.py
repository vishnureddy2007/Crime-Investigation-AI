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


def test_video_generation_multimedia_and_modes(tmp_path: Path):
    """Acceptance test: 3 images + 1 video, missing media, corrupt media handling, and custom modes."""
    import cv2
    import numpy as np
    from database.repository import save_uploaded_evidence_to_disk, save_case_evidence_item

    db_file = tmp_path / "multimedia_test.db"
    init_db(db_file)
    case_id = save_case(db_file, "multimedia_case", "MIXED")

    # 1. Create 3 valid image files
    img_paths = []
    for i in range(3):
        p = tmp_path / f"test_img_{i}.jpg"
        img = Image.new("RGB", (800, 600), color=(30 * i, 100, 150))
        img.save(p)
        img_paths.append(p)
        save_case_evidence_item(
            db_file,
            case_id=case_id,
            evidence_id=f"EVD-00{i+1}",
            filename=p.name,
            file_type="image",
            file_path=str(p),
            payload={"detections": [{"label": "person", "class_name": "person", "confidence": 0.88, "bbox": [50, 50, 200, 300]}]}
        )

    # 2. Create 1 valid video file (2 seconds at 10 fps = 20 frames)
    vid_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 480))
    for f in range(20):
        frame = np.full((480, 640, 3), (f * 10, 120, 200), dtype=np.uint8)
        out.write(frame)
    out.release()

    save_case_evidence_item(
        db_file,
        case_id=case_id,
        evidence_id="EVD-004",
        filename="test_video.mp4",
        file_type="video",
        file_path=str(vid_path),
        duration=2.0,
        fps=10.0,
        frame_count=20,
        payload={"detections": [{"label": "handgun", "class_name": "handgun", "confidence": 0.95, "weapon_status": "verified", "bbox": [100, 100, 250, 250]}]}
    )

    # 3. Add 1 missing media item
    save_case_evidence_item(
        db_file,
        case_id=case_id,
        evidence_id="EVD-005",
        filename="nonexistent.jpg",
        file_type="image",
        file_path=str(tmp_path / "nonexistent.jpg"),
        payload={"detections": []}
    )

    # 4. Add 1 corrupt image file
    corrupt_path = tmp_path / "corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_AN_IMAGE_FILE_DATA")

    save_case_evidence_item(
        db_file,
        case_id=case_id,
        evidence_id="EVD-006",
        filename="corrupt.jpg",
        file_type="image",
        file_path=str(corrupt_path),
        payload={"detections": []}
    )

    # Run InvestigationVideoGenerator
    generator = InvestigationVideoGenerator(db_path=db_file, fps=10, resolution=(640, 360))

    # Test evidence collection assembly
    collection = generator.get_case_evidence_collection(case_id)
    assert len(collection["items"]) == 6  # 3 images + 1 video + 1 missing + 1 corrupt

    video_output, duration = generator.generate_video(
        case_id=case_id,
        output_dir=tmp_path / "out_videos",
        detail_level="Detailed",
        video_evidence_mode="Representative Frames",
        force=True
    )

    assert video_output.exists()
    assert video_output.stat().st_size > 0
    assert duration > 5.0

