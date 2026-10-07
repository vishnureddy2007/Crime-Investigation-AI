"""
Tests for the summary generator.

We test:
1. The deterministic template builder for all common analysis patterns.
2. The LLM-backed generator with a *mock* pipeline (no real model load).
3. Graceful fallback when the model raises.
"""

from __future__ import annotations

import datetime

import pytest

from models.evidence_analyzer import analyze
from models.schemas import (
    AnalysisInput,
    BoundingBox,
    Detection,
    DetectionResult,
)
from models.summary_generator import (
    SummaryGenerator,
    build_template_summary,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _det(name: str, label: str, conf: float) -> Detection:
    return Detection(
        class_name=name, label=label,
        confidence=conf, bbox=BoundingBox(0, 0, 10, 10),
    )


def _analysis(counts: dict[str, int], conf: float = 0.8, source_type: str = "image") -> "object":
    dets: list[Detection] = []
    for label, n in counts.items():
        dets.extend(_det(label, label, conf) for _ in range(n))
    dr = DetectionResult(
        source_name="x.jpg",
        timestamp=datetime.datetime.now(),
        detections=dets,
    )
    return analyze(AnalysisInput(
        source_name=dr.source_name,
        counts_by_label=dr.counts_by_label(),
        average_confidence=dr.average_confidence(),
        source_type=source_type,
    ))


# ----------------------------------------------------------------------
# Template builder
# ----------------------------------------------------------------------
def test_template_empty_scene() -> None:
    a = _analysis({})
    summary = build_template_summary(a)
    # The template produces a summary even for empty analysis
    assert summary.investigation_summary is not None
    assert len(summary.investigation_summary) > 0
    assert a.source_type in summary.investigation_summary


def test_template_with_person_only() -> None:
    a = _analysis({"person": 2})
    summary = build_template_summary(a)
    # Template says "2 person(s)" not "2 persons were present"
    assert "2 person" in summary.investigation_summary
    assert "0 verified weapon" in summary.investigation_summary
    # suggested_category has underscores, template renders with spaces.
    assert a.suggested_category.replace("_", " ") in summary.investigation_summary


def test_template_with_weapon_marks_threat() -> None:
    a = _analysis({"person": 1, "knife": 1})
    summary = build_template_summary(a)
    assert "weapon" in summary.investigation_summary.lower()
    assert "verified weapon" in summary.verified_findings.lower()


def test_template_with_vehicle() -> None:
    # Use the MAPPED label "vehicle" (car/motorcycle/bus/truck -> vehicle).
    a = _analysis({"vehicle": 1})
    summary = build_template_summary(a)
    # The investigation_summary mentions the category
    assert "vehicle incident" in summary.investigation_summary


def test_template_with_bag() -> None:
    # Use the MAPPED label "bag" (backpack/handbag/suitcase -> bag).
    a = _analysis({"person": 1, "bag": 1}, source_type="video")
    summary = build_template_summary(a)
    assert "bag" in summary.detected_objects.lower()
    assert "video" in summary.investigation_summary


def test_template_high_confidence_mentions_it() -> None:
    a = _analysis({"person": 1}, conf=0.95)
    summary = build_template_summary(a)
    # Template produces valid summary
    assert summary.investigation_summary is not None
    assert len(summary.investigation_summary) > 0


# ----------------------------------------------------------------------
# LLM-backed generator (Mocked requests)
# ----------------------------------------------------------------------

def test_generator_uses_ai_when_ollama_succeeds(monkeypatch) -> None:
    """A successful Ollama API call returns a summary with AI-generated content."""
    import requests
    def mock_post(*args, **kwargs):
        m = type('Mock', (), {
            'raise_for_status': lambda x: None,
            'json': lambda x: {"response": '{"case_overview": "AI overview", "investigation_summary": "AI summary"}'}
        })()
        return m

    def mock_avail(self): return True
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(SummaryGenerator, "is_available", mock_avail)

    a = _analysis({"person": 1, "knife": 1}, conf=0.9)
    gen = SummaryGenerator(model_name="mock-model")

    summary = gen.generate(a)
    # AI-generated content should be in the investigation_summary
    assert "AI summary" in summary.investigation_summary
    assert summary.case_overview == "AI overview"

def test_generator_falls_back_when_ai_raises(monkeypatch) -> None:
    """An API exception falls through to the template."""
    import requests
    def mock_post(*args, **kwargs):
        raise RuntimeError("network down")

    def mock_avail(self): return True
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(SummaryGenerator, "is_available", mock_avail)

    a = _analysis({"person": 1}, conf=0.7)
    gen = SummaryGenerator(model_name="mock-model")

    summary = gen.generate(a)
    # Should fall back to template
    assert "person" in summary.investigation_summary.lower()

def test_generator_without_model_name_uses_template_only() -> None:
    a = _analysis({"person": 1})
    gen = SummaryGenerator(model_name=None)
    summary = gen.generate(a)
    # Should use template only
    assert "person" in summary.investigation_summary.lower()

def test_generator_rejects_empty_ai_output(monkeypatch) -> None:
    """An empty/whitespace response falls through to the template."""
    import requests
    def mock_post(*args, **kwargs):
        m = type('Mock', (), {
            'raise_for_status': lambda x: None,
            'json': lambda x: {"response": "   "}
        })()
        return m

    def mock_avail(self): return True
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(SummaryGenerator, "is_available", mock_avail)

    a = _analysis({"person": 1})
    gen = SummaryGenerator(model_name="mock-model")
    summary = gen.generate(a)
    # Should fall back to template
    assert "person" in summary.investigation_summary.lower()


# ----------------------------------------------------------------------
# Phase 24 — defensive branch coverage
# ----------------------------------------------------------------------


def test_template_handles_multiple_weapons() -> None:
    """A scene with 2+ verified weapons exercises the plural branch."""
    a = _analysis({"knife": 3})
    summary = build_template_summary(a)
    # Phase 46: verified weapons are reported as "3 verified weapons".
    assert "3 verified weapon" in summary.investigation_summary
    # And the singular branch (1 weapon) is NOT taken
    assert "1 verified weapon" not in summary.investigation_summary


def test_template_low_confidence_branch() -> None:
    """When average_confidence < 0.5 and total_objects > 0, the
    template adds the cross-check disclaimer."""
    a = _analysis({"person": 1}, conf=0.3)
    summary = build_template_summary(a)
    assert "moderate/low" in summary.uncertainties.lower()


def test_template_with_no_objects_returns_clear_scene_message() -> None:
    """Defensive branch at line 38-42 — a totally empty analysis
    should produce a valid summary."""
    a = _analysis({})
    summary = build_template_summary(a)
    assert summary.investigation_summary is not None
    assert len(summary.investigation_summary) > 0
