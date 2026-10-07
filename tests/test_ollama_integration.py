"""End-to-end integration tests for Local AI (Ollama).

This suite verifies that both the SummaryGenerator and ChatAssistant
correctly interact with the Ollama API, handle payloads, and fall back
gracefully when the service is unavailable.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from models.schemas import EvidenceAnalysis
from models.summary_generator import SummaryGenerator
from services.chat_assistant import ChatAssistant

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_analysis() -> EvidenceAnalysis:
    return EvidenceAnalysis(
        source_name="evidence_001.mp4",
        source_type="video",
        counts_by_label={"person": 2, "weapon": 1, "vehicle": 1, "bag": 1},
        total_objects=5,
        unique_labels=["person", "weapon", "vehicle", "bag"],
        average_confidence=0.82,
        person_count=2,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        weapon_count=1,
        vehicle_count=1,
        bag_count=1,
        severity_score=65,
        severity_level="high",
        suggested_category="robbery",
        key_observations=["Suspect 1 seen with firearm", "Vehicle idling outside"],
        has_threat=True,
        frame_count=10,
    )

# ----------------------------------------------------------------------
# Summary Integration Tests
# ----------------------------------------------------------------------
def test_summary_ollama_integration(sample_analysis):
    """Verify SummaryGenerator -> Ollama API flow."""
    gen = SummaryGenerator()
    ai_text = "Two persons and one verified weapon were detected. A vehicle was present."

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": json.dumps({
            "investigation_summary": ai_text
        })}

        summary = gen.generate(sample_analysis)

        # The AI-generated content should appear in the investigation_summary
        assert ai_text in summary.investigation_summary

        # Verify the request sent to Ollama
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["model"] == OLLAMA_MODEL
        assert "Persons(2)" in payload["prompt"]

def test_summary_ollama_fallback(sample_analysis):
    """Verify SummaryGenerator falls back on API failure."""
    gen = SummaryGenerator()
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 500 # Service down
        summary = gen.generate(sample_analysis)
        # Should fall back to template - summary still valid
        assert summary.investigation_summary is not None
        assert len(summary.investigation_summary) > 0

# ----------------------------------------------------------------------
# Chat Integration Tests
# ----------------------------------------------------------------------
def test_chat_ollama_integration(sample_analysis):
    """Verify ChatAssistant -> Ollama API flow."""
    assistant = ChatAssistant(analysis=sample_analysis)
    ai_reply = "Yes, one verified weapon was detected."

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": ai_reply}

        reply = assistant.reply("Were any weapons detected?")

        assert reply.used_ai is True
        assert reply.answer == ai_reply
        assert reply.model_name == OLLAMA_MODEL

        # Verify prompt contains facts
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert "Verified weapons: 1" in payload["prompt"]

def test_chat_ollama_fallback(sample_analysis):
    """Verify ChatAssistant falls back on API failure."""
    assistant = ChatAssistant(analysis=sample_analysis)
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 404 # Service down
        reply = assistant.reply("Were any weapons detected?")
        assert reply.used_ai is False
        assert reply.source == "empty"  # AI unavailable returns empty source
