"""Unit tests for `services/chat_assistant.py`."""
from __future__ import annotations

from datetime import datetime

from models.schemas import EvidenceAnalysis
from services.chat_assistant import ChatAssistant, ChatReply, _fallback_reply


def _analysis(**kw) -> EvidenceAnalysis:
    base = dict(
        source_name="clip.png",
        source_type="image",
        counts_by_label={"person": 2, "knife": 1},
        total_objects=3,
        unique_labels=["person", "knife"],
        average_confidence=0.8,
        person_count=2,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        weapon_count=1,
        vehicle_count=0,
        bag_count=0,
        severity_score=70,
        severity_level="high",
        suggested_category="assault",
        key_observations=[],
        has_threat=True,
        frame_count=1,
        timestamp=datetime.now(),
    )
    base.update(kw)
    # Keep verified_weapon_count in sync with weapon_count when the test
    # sets one but not the other (most tests still set weapon_count).
    if "verified_weapon_count" not in kw and "weapon_count" in kw:
        base["verified_weapon_count"] = base["weapon_count"]
    if "weapon_count" not in kw and "verified_weapon_count" in kw:
        base["weapon_count"] = base["verified_weapon_count"]
    return EvidenceAnalysis(**base)


class TestChatReply:
    def test_as_dict_roundtrip(self) -> None:
        r = ChatReply(answer="hi", source="template", used_ai=False, model_name="x")
        d = r.as_dict()
        assert d["answer"] == "hi"
        assert d["source"] == "template"
        assert d["used_ai"] is False
        assert d["model_name"] == "x"
        assert "timestamp" in d


class TestFallbackReply:
    def test_no_analysis_returns_empty(self) -> None:
        r = _fallback_reply("anything", None)
        assert r.source == "empty"
        assert r.used_ai is False

    def test_weapon_question(self) -> None:
        r = _fallback_reply("Were any weapons detected?", _analysis())
        assert r.source == "template"
        assert "weapon" in r.answer.lower()

    def test_no_weapon_question(self) -> None:
        r = _fallback_reply(
            "were any weapons detected?",
            _analysis(weapon_count=0, verified_weapon_count=0,
                      candidate_weapon_count=0),
        )
        assert "no weapons" in r.answer.lower()

    def test_person_count(self) -> None:
        r = _fallback_reply("how many people?", _analysis(person_count=5))
        assert "5" in r.answer

    def test_severity_question(self) -> None:
        r = _fallback_reply("what is the severity?", _analysis())
        assert "high" in r.answer.lower()

    def test_generic_question_summarises(self) -> None:
        r = _fallback_reply("tell me everything", _analysis())
        assert r.source == "template"
        assert "persons" in r.answer.lower() or "Severity" in r.answer


class TestChatAssistant:
    def test_empty_question_returns_please_type(self) -> None:
        ca = ChatAssistant(model_name=None)
        r = ca.reply("   ", analysis=_analysis())
        assert "please type" in r.answer.lower()

    def test_no_analysis_uses_fallback(self) -> None:
        ca = ChatAssistant(model_name=None)
        r = ca.reply("anything", analysis=None)
        assert r.source == "empty"

    def test_unavailable_model_uses_fallback(self, monkeypatch) -> None:
        # Mock is_available to return False
        def mock_avail(self): return False
        monkeypatch.setattr(ChatAssistant, "is_available", mock_avail)

        ca = ChatAssistant(model_name="fake-model")
        r = ca.reply("Were weapons detected?", analysis=_analysis())
        # When AI is unavailable, it returns empty source with a fallback message
        assert r.source == "empty"
        assert r.used_ai is False
        assert "unavailable" in r.answer.lower() or "fallback" in r.answer.lower()

    def test_is_available_mocked(self, monkeypatch) -> None:
        import requests
        def mock_get(*args, **kwargs):
            m = type('Mock', (), {'status_code': 200})()
            return m
        monkeypatch.setattr(requests, "get", mock_get)
        assert ChatAssistant(model_name="fake-model").is_available() is True

    def test_prompt_contains_facts(self) -> None:
        a = _analysis()
        prompt = ChatAssistant._build_prompt("any?", a)
        assert "Persons: 2" in prompt
        assert "Verified weapons: 1" in prompt
        assert "Candidate weapons" in prompt
        assert "Severity" in prompt

# ----------------------------------------------------------------------
# Phase 18 — LLM success path + edge cases (Mocked requests)
# ----------------------------------------------------------------------

def test_reply_uses_ai_when_ollama_succeeds(monkeypatch) -> None:
    """A successful Ollama API call returns a ChatReply with source=ai."""
    import requests
    def mock_post(*args, **kwargs):
        m = type('Mock', (), {
            'raise_for_status': lambda x: None,
            'json': lambda x: {"response": "The scene shows 1 weapon."}
        })()
        return m

    def mock_avail(self): return True
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(ChatAssistant, "is_available", mock_avail)

    ca = ChatAssistant(model_name="fake-model")
    r = ca.reply("what weapons?", analysis=_analysis())
    assert r.source == "ai"
    assert r.used_ai is True
    assert r.model_name == "fake-model"
    assert r.answer == "The scene shows 1 weapon."

def test_reply_handles_ai_returning_empty_text(monkeypatch) -> None:
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
    monkeypatch.setattr(ChatAssistant, "is_available", mock_avail)

    ca = ChatAssistant(model_name="fake-model")
    r = ca.reply("Were weapons detected?", analysis=_analysis())
    assert r.source == "template"
    assert r.used_ai is False

def test_reply_handles_ai_exception(monkeypatch) -> None:
    """An API exception falls through to the template silently."""
    import requests
    def mock_post(*args, **kwargs):
        raise RuntimeError("network down")

    def mock_avail(self): return True
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(ChatAssistant, "is_available", mock_avail)

    ca = ChatAssistant(model_name="fake-model")
    r = ca.reply("Were weapons detected?", analysis=_analysis())
    assert r.source == "template"
    assert r.used_ai is False

def test_reply_handles_ollama_returning_no_response_key(monkeypatch) -> None:
    """An unexpected JSON shape (no `response` key) falls through."""
    import requests
    def mock_post(*args, **kwargs):
        m = type('Mock', (), {
            'raise_for_status': lambda x: None,
            'json': lambda x: {"unrelated": "noise"}
        })()
        return m

    def mock_avail(self): return True
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(ChatAssistant, "is_available", mock_avail)

    ca = ChatAssistant(model_name="fake-model")
    r = ca.reply("Were weapons detected?", analysis=_analysis())
    assert r.source == "template"
