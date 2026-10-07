"""Robustness tests for edge cases and service failures."""

from __future__ import annotations

import pytest
from unittest.mock import patch
from pathlib import Path

import requests
from database.db import init_db
from database.repository import save_case
from models.schemas import EvidenceAnalysis
from models.summary_generator import SummaryGenerator
from services.chat_assistant import ChatAssistant

def test_empty_detections_robustness(tmp_path: Path):
    """Verify that cases with 0 detections don't crash the summary or chat."""
    db_path = tmp_path / "robustness.db"
    init_db(db_path)
    cid = save_case(db_path, "empty.jpg", "image")

    empty_analysis = EvidenceAnalysis(
        source_name="empty.jpg",
        source_type="image",
        counts_by_label={},
        total_objects=0,
        unique_labels=[],
        average_confidence=0.0,
        person_count=0,
        verified_weapon_count=0,
        candidate_weapon_count=0,
        weapon_count=0,
        vehicle_count=0,
        bag_count=0,
        severity_score=0,
        severity_level="low",
        suggested_category="none",
        key_observations=[],
        has_threat=False,
        frame_count=1,
    )

    # Summary should not crash
    gen = SummaryGenerator()
    summary = gen.generate(empty_analysis)
    assert summary.investigation_summary is not None
    assert len(summary.investigation_summary) > 0

    # Chat should not crash - ask a specific question to trigger fallback logic
    assistant = ChatAssistant(analysis=empty_analysis)
    reply = assistant.reply("Were any weapons detected?")
    assert reply is not None
    assert reply.answer is not None

def test_ollama_offline_graceful_fallback():
    """Verify that when Ollama is simulated as offline, the app uses fallbacks.
    This test mocks is_available to return False to force template mode."""
    analysis = EvidenceAnalysis(
        source_name="test.jpg", source_type="image", counts_by_label={"person": 1},
        total_objects=1, unique_labels=["person"], average_confidence=0.9,
        person_count=1, verified_weapon_count=0, candidate_weapon_count=0,
        weapon_count=0, vehicle_count=0, bag_count=0, severity_score=10, severity_level="low",
        suggested_category="suspicious", key_observations=[], has_threat=False, frame_count=1
    )

    # Summary Generator fallback - use model_name=None to force template mode
    gen = SummaryGenerator(model_name=None)
    summary = gen.generate(analysis)
    # The fallback template is used
    assert "person" in summary.investigation_summary.lower()

    # Chat Assistant fallback - mock is_available on the instance
    assistant = ChatAssistant(analysis=analysis, model_name=None)
    with patch.object(assistant, 'is_available', return_value=False):
        reply = assistant.reply("How many people are here?")
        # When AI is unavailable, it falls back gracefully
        assert reply is not None
        assert reply.source in ("empty", "template")
        assert "fallback" in reply.answer.lower() or "unavailable" in reply.answer.lower() or "person" in reply.answer.lower()
