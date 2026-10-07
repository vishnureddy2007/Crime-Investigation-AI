"""Tests for Ollama-backed summary generation.

Covers:
1. `is_available` correctly reports service status.
2. `generate` sends the correct payload to the Ollama API.
3. `generate` correctly parses a successful AI response.
4. `generate` falls back to the template on API errors (404, 500, timeout).
5. `generate` falls back to the template on empty AI responses.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from models.schemas import EvidenceAnalysis
from models.summary_generator import SummaryGenerator

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

@pytest.fixture
def generator() -> SummaryGenerator:
    return SummaryGenerator()

# ----------------------------------------------------------------------
# Availability tests
# ----------------------------------------------------------------------
def test_is_available_success(generator):
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        assert generator.is_available() is True

def test_is_available_fail(generator):
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 500
        assert generator.is_available() is False

def test_is_available_exception(generator):
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.RequestException("Connection refused")
        assert generator.is_available() is False

# ----------------------------------------------------------------------
# Generation tests
# ----------------------------------------------------------------------
def test_generate_ai_success(generator, sample_analysis):
    """Verify successful AI generation flow."""
    ai_response = {
        "investigation_summary": "Two persons and one verified weapon were detected. A vehicle was present."
    }

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # 1. Mock availability check
        mock_get.return_value.status_code = 200

        # 2. Mock generation response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": json.dumps(ai_response)}

        summary = generator.generate(sample_analysis)

        # The AI-generated content should appear in the investigation_summary
        assert ai_response["investigation_summary"] in summary.investigation_summary

        # Verify payload
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["model"] == OLLAMA_MODEL
        assert "Persons(2)" in payload["prompt"]
        assert "1 verified weapon" in payload["prompt"]

def test_generate_ai_empty_response(generator, sample_analysis):
    """Verify fallback when AI returns empty string."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": ""}

        summary = generator.generate(sample_analysis)

        # Should fall back to template - summary still valid
        assert summary.investigation_summary is not None
        assert len(summary.investigation_summary) > 0

def test_generate_api_error(generator, sample_analysis):
    """Verify fallback on HTTP error."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200

        # Simulate 500 Internal Server Error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        summary = generator.generate(sample_analysis)

        # Should fall back to template - summary still valid
        assert summary.investigation_summary is not None
        assert len(summary.investigation_summary) > 0

def test_generate_timeout(generator, sample_analysis):
    """Verify fallback on request timeout."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_post.side_effect = requests.Timeout("Request timed out")

        summary = generator.generate(sample_analysis)

        # Should fall back to template - summary still valid
        assert summary.investigation_summary is not None
        assert len(summary.investigation_summary) > 0

def test_generate_service_unavailable(generator, sample_analysis):
    """Verify behavior when Ollama is not running."""
    with patch("requests.get") as mock_get:
        # is_available returns False
        mock_get.return_value.status_code = 404

        summary = generator.generate(sample_analysis)

        # Should fall back to template - summary still valid
        assert summary.investigation_summary is not None
        assert len(summary.investigation_summary) > 0
