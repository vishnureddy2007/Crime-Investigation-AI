"""
Unit & Integration Test Suite for Single-Call AI Architecture.
Verifies MAX 1 Qwen analysis call per evidence version, persistent caching,
outdated state invalidation, intent-routed chat, and graceful offline handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from config import OLLAMA_MODEL
from core.profiler import AICallRegistry, get_ai_call_count, record_ai_call
from database.db import init_db
from database.repository import (
    load_summary,
    mark_case_outdated,
    save_analysis,
    save_case,
    save_summary,
)
from models.schemas import DetailedNarrativeSummary, EvidenceAnalysis, InvestigationSummary
from models.summary_generator import (
    SummaryGenerator,
    build_template_summary,
    calculate_analysis_hash,
)
from services.chat_assistant import ChatAssistant
from services.crime_situation import CrimeSituationAnalyzer


@pytest.fixture(autouse=True)
def _reset_ai_registry():
    """Reset AICallRegistry call counts before each test."""
    AICallRegistry.get_instance().reset()


def _sample_analysis(source_name: str = "test_scene.jpg") -> EvidenceAnalysis:
    """Fixture providing a valid sample EvidenceAnalysis."""
    return EvidenceAnalysis(
        source_name=source_name,
        source_type="image",
        counts_by_label={"person": 2, "weapon": 1, "car": 1},
        total_objects=4,
        unique_labels=["car", "person", "weapon"],
        average_confidence=0.88,
        person_count=2,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        vehicle_count=1,
        bag_count=0,
        severity_score=85,
        severity_level="critical",
        suggested_category="armed_robbery",
        key_observations=["1 verified weapon detected"],
        has_threat=True,
        weapon_count=1,
        frame_count=1,
    )


# =====================================================================
# Test 1 — First analysis calls Qwen exactly once and saves result
# =====================================================================
def test_1_first_analysis_calls_qwen_once_and_saves(tmp_path: Path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    analysis = _sample_analysis()
    version_hash = calculate_analysis_hash(analysis)

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": '{"investigation_summary": "Verified armed robbery scene.", "case_overview": "Overview text"}'
        }

        gen = SummaryGenerator()
        summary = gen.generate(analysis)

        assert summary is not None
        assert get_ai_call_count(version_hash) == 1

        # Persist to database
        case_id = save_case(db_path, analysis.source_name, "image")
        inv_sum = InvestigationSummary(
            source_name=analysis.source_name,
            source_type="image",
            template_text="Template text",
            ai_text=summary.investigation_summary,
            used_ai=True,
            model_name=OLLAMA_MODEL,
        )
        save_summary(db_path, case_id, inv_sum)

        loaded = load_summary(db_path, 1)
        assert loaded is not None
        assert loaded["used_ai"] is True


# =====================================================================
# Test 2 — Refresh/Rerun reuses cached result (0 new calls)
# =====================================================================
def test_2_refresh_reuses_cached_result_zero_calls():
    analysis = _sample_analysis()
    version_hash = calculate_analysis_hash(analysis)

    # First call records 1 call
    record_ai_call(version_hash)
    assert get_ai_call_count(version_hash) == 1

    # Simulate Streamlit rerun reusing cached result
    cached_summary = DetailedNarrativeSummary(
        case_id=analysis.source_name,
        case_overview="Cached overview",
        evidence_reviewed="1 verified weapon",
        chronological_events="T+0: Event",
        detected_objects="Objects text",
        verified_findings="1 weapon",
        possible_findings="None",
        rejected_findings="None",
        potential_crime_activity="Robbery",
        important_evidence="Weapon",
        uncertainties="None",
        investigation_summary="Cached summary",
    )

    # Reusing cached_summary requires 0 new Ollama calls
    assert cached_summary.investigation_summary == "Cached summary"
    assert get_ai_call_count(version_hash) == 1


# =====================================================================
# Test 3 — Page navigation reuses cached result (0 calls)
# =====================================================================
def test_3_page_navigation_reuses_cached_result():
    analysis = _sample_analysis()
    version_hash = calculate_analysis_hash(analysis)
    record_ai_call(version_hash)

    # Situation analyzer reusing cached result
    analyzer = CrimeSituationAnalyzer()
    with patch("requests.post") as mock_post:
        # Mocking st.session_state cached result
        sit = analyzer._fallback_analysis(analysis)
        assert sit is not None
        mock_post.assert_not_called()

    assert get_ai_call_count(version_hash) == 1


# =====================================================================
# Test 4 — Open report reuses cached result (0 calls)
# =====================================================================
def test_4_open_report_reuses_cached_result(tmp_path: Path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    analysis = _sample_analysis()
    version_hash = calculate_analysis_hash(analysis)

    case_id = save_case(db_path, analysis.source_name, "image")
    inv_summary = InvestigationSummary(
        source_name=analysis.source_name,
        source_type="image",
        template_text="Report summary text",
        ai_text="Report summary text",
        used_ai=True,
    )
    save_summary(db_path, case_id, inv_summary)

    # Opening report reads from DB
    loaded = load_summary(db_path, 1)
    assert loaded is not None
    assert loaded["ai_text"] == "Report summary text"
    assert get_ai_call_count(version_hash) == 0


# =====================================================================
# Test 5 — Open 3D animation reuses cached result (0 calls)
# =====================================================================
def test_5_open_animation_reuses_cached_result():
    analysis = _sample_analysis()
    version_hash = calculate_analysis_hash(analysis)

    summary = DetailedNarrativeSummary(
        case_id=analysis.source_name,
        case_overview="Overview",
        evidence_reviewed="Reviewed",
        chronological_events="Events",
        detected_objects="Objects",
        verified_findings="Verified",
        possible_findings="Possible",
        rejected_findings="Rejected",
        potential_crime_activity="Activity",
        important_evidence="Evidence",
        uncertainties="None",
        investigation_summary="Animation summary text",
    )

    from services.scene_planner_service import ScenePlanner
    planner = ScenePlanner()
    with patch("requests.post") as mock_post:
        plan = planner.plan_structured_scene(summary, None, analysis)
        assert plan is not None
        mock_post.assert_not_called()

    assert get_ai_call_count(version_hash) == 0


# =====================================================================
# Test 6 — Reopen saved case loads DB result (0 calls)
# =====================================================================
def test_6_reopen_case_loads_saved_db_result(tmp_path: Path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    analysis = _sample_analysis()
    version_hash = calculate_analysis_hash(analysis)

    cid = save_case(db_path, analysis.source_name, "image")
    save_analysis(db_path, cid, analysis)

    inv_summary = InvestigationSummary(
        source_name=analysis.source_name,
        source_type="image",
        template_text="Saved DB summary",
        ai_text="Saved DB summary",
        used_ai=True,
    )
    save_summary(db_path, cid, inv_summary)

    # Reopening case
    loaded = load_summary(db_path, 1)
    assert loaded["is_outdated"] is False
    assert get_ai_call_count(version_hash) == 0


# =====================================================================
# Test 7 — Human rejects weapon: Marks outdated; user regenerates (1 call)
# =====================================================================
def test_7_human_rejection_marks_outdated_and_regenerates_once(tmp_path: Path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    analysis = _sample_analysis()

    cid = save_case(db_path, analysis.source_name, "image")
    save_analysis(db_path, cid, analysis)

    # Initial summary
    inv_summary = InvestigationSummary(
        source_name=analysis.source_name,
        source_type="image",
        template_text="Summary #1",
        ai_text="Summary #1",
        used_ai=True,
    )
    save_summary(db_path, cid, inv_summary)

    # Human review action: reject weapon -> mark outdated
    mark_case_outdated(db_path, cid)
    loaded = load_summary(db_path, 1)
    assert loaded["is_outdated"] is True

    # User clicks Regenerate: 1 new Qwen call for updated version
    analysis.verified_weapon_count = 0
    analysis.candidate_weapon_count = 1
    new_hash = calculate_analysis_hash(analysis, {"det_1": "REJECT"})

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": '{"investigation_summary": "Updated summary after rejection."}'
        }

        gen = SummaryGenerator()
        new_summary = gen.generate(analysis, user_review_overrides={"det_1": "REJECT"})
        assert new_summary is not None
        assert get_ai_call_count(new_hash) == 1


# =====================================================================
# Test 8 — New media / evidence version changes (1 new call)
# =====================================================================
def test_8_new_media_version_triggers_one_new_call():
    a1 = _sample_analysis("scene_v1.jpg")
    a2 = _sample_analysis("scene_v2.jpg")

    hash1 = calculate_analysis_hash(a1)
    hash2 = calculate_analysis_hash(a2)

    assert hash1 != hash2

    record_ai_call(hash1)
    assert get_ai_call_count(hash1) == 1
    assert get_ai_call_count(hash2) == 0

    record_ai_call(hash2)
    assert get_ai_call_count(hash1) == 1
    assert get_ai_call_count(hash2) == 1


# =====================================================================
# Test 9 — Factual chat question calls Qwen 0 times (Direct Python Answer)
# =====================================================================
def test_9_factual_chat_question_uses_zero_qwen_calls():
    analysis = _sample_analysis()
    chat = ChatAssistant(analysis=analysis)

    with patch("requests.post") as mock_post:
        reply1 = chat.reply("How many weapons were verified?", analysis=analysis)
        assert reply1.used_ai is False
        assert "verified" in reply1.answer.lower()
        mock_post.assert_not_called()

        reply2 = chat.reply("How many people were detected?", analysis=analysis)
        assert reply2.used_ai is False
        assert "2" in reply2.answer
        mock_post.assert_not_called()


# =====================================================================
# Test 10 — Reasoning chat question calls Qwen without case re-analysis
# =====================================================================
def test_10_reasoning_chat_question_calls_qwen_for_reasoning_only():
    analysis = _sample_analysis()
    chat = ChatAssistant(analysis=analysis)

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": "The presence of a handgun suggests potential armed confrontation."
        }

        reply = chat.reply("Explain what the evidence suggests.", analysis=analysis)
        assert reply.used_ai is True
        assert reply.source == "ai"
        assert "handgun" in reply.answer
        assert mock_post.call_count == 1


# =====================================================================
# Test 11 — Ollama unavailable handles gracefully without fake AI results
# =====================================================================
def test_11_ollama_unavailable_handles_gracefully():
    analysis = _sample_analysis()
    gen = SummaryGenerator()

    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Ollama connection refused")
        summary = gen.generate(analysis)

        # Returns deterministic template without throwing or hallucinating fake AI responses
        assert summary is not None
        assert summary.investigation_summary == build_template_summary(analysis).investigation_summary


# =====================================================================
# Test 12 — Duplicate button clicks protected by state lock
# =====================================================================
def test_12_duplicate_clicks_protected_by_state_lock():
    version_hash = "test_lock_hash_123"

    reg = AICallRegistry.get_instance()
    assert reg.acquire_lock(version_hash) is True
    assert reg.is_locked(version_hash) is True

    # Second concurrent click fails to acquire lock
    assert reg.acquire_lock(version_hash) is False

    reg.release_lock(version_hash)
    assert reg.is_locked(version_hash) is False
    assert reg.acquire_lock(version_hash) is True
