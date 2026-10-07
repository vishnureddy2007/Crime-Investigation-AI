"""
Audit tests for the forensic report generators.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from pathlib import Path
from models.schemas import ReportData, IntelligenceReportData, EvidenceAnalysis
from models.report_generator import (
    PDFReportGenerator,
    DOCXReportGenerator,
    IntelligenceReportGenerator,
    build_report_data,
)

def test_pdf_generator_produces_valid_bytes() -> None:
    """Verify that PDFReportGenerator returns non-empty bytes and has PDF header."""
    data = ReportData(
        title="Test Report",
        report_id="RPT-123",
        generated_at=datetime.now(),
        source_name="test.jpg",
        source_type="image",
        model_name="yolo",
        summary_text="A test summary.",
        severity_score=50,
        severity_level="high",
        suggested_category="generic",
        has_threat=False,
        counts_by_label={"person": 1},
        total_objects=1,
        person_count=1,
        weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        key_observations=["Obs 1"],
        average_confidence=0.8,
        remarks="Test remarks",
        app_version="1.0",
        human_total_detections=0,
        human_confirmed_count=0,
        human_rejected_count=0,
        human_pending_count=0,
    )
    gen = PDFReportGenerator()
    pdf_bytes = gen.generate(data)
    assert len(pdf_bytes) > 0
    # PDF files start with %PDF
    assert pdf_bytes.startswith(b"%PDF")

def test_docx_generator_produces_valid_bytes() -> None:
    """Verify that DOCXReportGenerator returns non-empty bytes."""
    # Reuse data from previous test
    from tests.test_report_audit import _get_mock_report_data
    data = _get_mock_report_data()
    gen = DOCXReportGenerator()
    docx_bytes = gen.generate(data)
    assert len(docx_bytes) > 0
    # DOCX files are ZIP archives, starting with PK
    assert docx_bytes.startswith(b"PK")

def test_intelligence_report_generator_produces_valid_pdf() -> None:
    """Verify that IntelligenceReportGenerator returns a valid PDF."""
    data = IntelligenceReportData(
        report_id="INTEL-123",
        generated_at=datetime.now(),
        global_distribution={"person": 10},
        crime_series={0: [1, 2]},
        case_details={1: "Case 1", 2: "Case 2"},
        app_version="1.0",
    )
    gen = IntelligenceReportGenerator()
    pdf_bytes = gen.generate(data)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")

def test_build_report_data_correctly_maps_fields() -> None:
    """Verify that build_report_data correctly transforms Analysis -> ReportData."""
    analysis = EvidenceAnalysis(
        source_name="scene.jpg", source_type="image",
        counts_by_label={"knife": 1}, total_objects=1, unique_labels=["knife"],
        average_confidence=0.8, person_count=0, verified_weapon_count=1,
        candidate_weapon_count=0, weapon_count=1, vehicle_count=0, bag_count=0,
        severity_score=80, severity_level="critical", suggested_category="robbery",
        key_observations=["Weapon found"], has_threat=True, frame_count=1
    )

    # Test without summary (should use template)
    report_data = build_report_data(analysis)

    assert report_data.source_name == "scene.jpg"
    assert report_data.severity_score == 80
    assert report_data.severity_level == "critical"
    assert report_data.has_threat is True
    assert report_data.counts_by_label == {"knife": 1}
    assert "robbery" in report_data.summary_text.lower() # template usually includes category

def _get_mock_report_data() -> ReportData:
    return ReportData(
        title="Test Report",
        report_id="RPT-123",
        generated_at=datetime.now(),
        source_name="test.jpg",
        source_type="image",
        model_name="yolo",
        summary_text="A test summary.",
        severity_score=50,
        severity_level="high",
        suggested_category="generic",
        has_threat=False,
        counts_by_label={"person": 1},
        total_objects=1,
        person_count=1,
        weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        key_observations=["Obs 1"],
        average_confidence=0.8,
        remarks="Test remarks",
        app_version="1.0",
        human_total_detections=0,
        human_confirmed_count=0,
        human_rejected_count=0,
        human_pending_count=0,
    )
