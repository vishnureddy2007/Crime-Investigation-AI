"""Lightweight chat assistant for investigators.

Uses a local LLM (Qwen3 14B via Ollama) to answer questions about
evidence analysis. If Ollama is unavailable, it falls back to
deterministic template replies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests
from core.logging import get_logger
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SUMMARY_MAX_NEW_TOKENS,
)
from models.schemas import EvidenceAnalysis

# ----------------------------------------------------------------------
# Data Classes
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ChatReply:
    """One assistant turn. `source` is "ai" | "template" | "empty"."""

    answer: str
    source: str
    used_ai: bool
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer":     self.answer,
            "source":     self.source,
            "used_ai":    self.used_ai,
            "model_name": self.model_name,
            "timestamp":  self.timestamp.isoformat(timespec="seconds"),
        }

# ----------------------------------------------------------------------
# Deterministic Fallbacks
# ----------------------------------------------------------------------
def _facts_block(analysis: EvidenceAnalysis) -> str:
    """Compact fact block used in prompts and fallback replies."""
    return (
        f"Persons: {analysis.person_count}\n"
        f"Verified weapons: {analysis.verified_weapon_count}\n"
        f"Candidate weapons (requires manual verification): "
        f"{analysis.candidate_weapon_count}\n"
        f"Vehicles: {analysis.vehicle_count}\n"
        f"Bags: {analysis.bag_count}\n"
        f"Severity: {analysis.severity_level} ({analysis.severity_score}/100)\n"
        f"Suggested category: {analysis.suggested_category}\n"
        f"Has threat (verified weapons only): {analysis.has_threat}\n"
        f"Source type: {analysis.source_type}"
    )


def _fallback_reply(question: str, analysis: EvidenceAnalysis | None) -> ChatReply:
    """Deterministic answer when no analysis is in scope or AI is off."""
    q = question.strip().lower()
    if analysis is None:
        return ChatReply(
            answer=(
                "I don't have any evidence analysis yet. Please run "
                "**Image Detection** or **Video Processing** first, then "
                "ask your question here."
            ),
            source="empty",
            used_ai=False,
            model_name="template-only",
        )

    if any(kw in q for kw in ("weapon", "knife", "threat", "dangerous")):
        verified = analysis.verified_weapon_count
        candidate = analysis.candidate_weapon_count
        if verified == 0 and candidate == 0:
            text = (
                f"No weapons were detected in {analysis.source_name}. The "
                "scene appears to be non-threatening on this axis."
            )
        elif verified > 0 and candidate == 0:
            text = (
                f"Yes — **{verified}** verified weapon(s) were detected in "
                f"{analysis.source_name}. The scene is flagged as having a "
                "potential threat. Review the storyboard and report for details."
            )
        elif verified == 0 and candidate > 0:
            text = (
                f"No verified weapons in {analysis.source_name}, but "
                f"**{candidate}** possible weapon candidate(s) were flagged "
                "(low confidence). These **require manual verification** "
                "and do NOT trigger the threat flag on their own."
            )
        else:
            text = (
                f"**{verified}** verified weapon(s) and **{candidate}** "
                f"possible candidate(s) were detected in {analysis.source_name}. "
                "Only the verified weapons drive the threat flag; the "
                "candidates must be reviewed by an investigator."
            )
        return ChatReply(answer=text, source="template", used_ai=False, model_name="template-only")

    if "person" in q or "people" in q or "how many" in q:
        return ChatReply(
            answer=(
                f"Approximately **{analysis.person_count}** person(s) were "
                f"detected in {analysis.source_name}."
            ),
            source="template",
            used_ai=False,
            model_name="template-only",
        )

    if "severity" in q or "score" in q or "risk" in q:
        return ChatReply(
            answer=(
                f"The current severity is **{analysis.severity_level.upper()}** "
                f"(score {analysis.severity_score}/100). Suggested category: "
                f"**{analysis.suggested_category.replace('_', ' ')}**."
            ),
            source="template",
            used_ai=False,
            model_name="template-only",
        )

    return ChatReply(
        answer=(
            "Here is the evidence I can summarise:\n\n"
            f"- Source: `{analysis.source_name}` ({analysis.source_type})\n"
            f"- Severity: **{analysis.severity_level.upper()}** "
            f"({analysis.severity_score}/100)\n"
            f"- Persons: {analysis.person_count}, "
            f"Verified weapons: {analysis.verified_weapon_count}, "
            f"Candidate weapons: {analysis.candidate_weapon_count}, "
            f"Vehicles: {analysis.vehicle_count}, "
            f"Bags: {analysis.bag_count}\n"
            f"- Suggested category: **{analysis.suggested_category.replace('_', ' ')}**\n\n"
            "Ask a more specific question (e.g. 'were weapons detected?' or "
            "'what is the severity?') for a focused answer."
        ),
        source="template",
        used_ai=False,
        model_name="template-only",
    )


# ----------------------------------------------------------------------
# Local AI-backed Assistant (Ollama)
# ----------------------------------------------------------------------
class ChatAssistant:
    """Stateful chat helper. Uses Ollama API for Qwen3 14B."""

    def __init__(
        self,
        analysis: EvidenceAnalysis | None = None,
        model_name: str | None = OLLAMA_MODEL,
        max_new_tokens: int = SUMMARY_MAX_NEW_TOKENS,
    ) -> None:
        self.analysis = analysis
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def is_available(self) -> bool:
        """Check if the local AI service is reachable."""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            return response.status_code == 200
        except (requests.RequestException, Exception):
            return False

    def reply(
        self,
        question: str,
        analysis: EvidenceAnalysis | None = None,
        human_decisions: dict[str, str] | None = None,
    ) -> ChatReply:
        """Return one reply to `question`, given the optional analysis."""
        import time
        from core.profiler import profile_stage
        t0 = time.perf_counter()

        active_analysis = analysis or self.analysis

        question = (question or "").strip()
        if not question:
            return ChatReply(
                answer="Please type a question regarding the active case evidence.",
                source="empty",
                used_ai=False,
                model_name="template-only",
            )

        if active_analysis is None:
            return _fallback_reply(question, active_analysis)

        # Check if AI is available
        if not self.is_available():
            profile_stage("chat_assistant_reply", time.perf_counter() - t0, "Ollama Offline - Fallback Used")
            return ChatReply(
                answer=(
                    "The AI Assistant is currently unavailable. "
                    "Please ensure Ollama is running locally and the "
                    f"model `{self.model_name}` is installed. "
                    "Falling back to rule-based answers."
                ),
                source="empty",
                used_ai=False,
                model_name="template-only",
            )

        prompt = self._build_prompt(question, active_analysis, human_decisions)
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 250,
                    "temperature": 0.2,
                }
            }
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=25.0,
            )
            response.raise_for_status()
            result = response.json()

            raw = result.get("response", "").strip()
            if raw:
                profile_stage("chat_assistant_reply", time.perf_counter() - t0, f"Qwen3 14B Success ({len(raw)} chars)")
                return ChatReply(
                    answer=raw,
                    source="ai",
                    used_ai=True,
                    model_name=self.model_name or "unknown",
                )
        except Exception as exc:
            get_logger(__name__).warning("Ollama chat API request failed: %s", exc)
            profile_stage("chat_assistant_reply", time.perf_counter() - t0, f"Fallback Used ({exc})")

        return _fallback_reply(question, active_analysis)

    @staticmethod
    def _build_prompt(question: str, analysis: EvidenceAnalysis, human_decisions: dict[str, str] | None = None) -> str:
        decisions_summary = "None recorded"
        if human_decisions:
            decisions_summary = ", ".join(f"{k}: {v}" for k, v in human_decisions.items())

        return (
            "You are an AI Forensic Assistant helping a crime investigator. "
            "Answer the question in ONE concise paragraph using ONLY the verified facts below.\n"
            "GROUNDING RULES:\n"
            "1. Strictly distinguish VERIFIED EVIDENCE from UNKNOWN or REJECTED details.\n"
            "2. Do NOT invent people, weapons, timestamps, or locations.\n"
            "3. If the requested information is not in the facts below, reply exactly: "
            "'This information is not present in the verified case evidence.'\n\n"
            f"VERIFIED CASE FACTS:\n"
            f"Case / Media Source: {analysis.source_name}\n"
            f"Persons: {analysis.person_count}\n"
            f"Verified weapons: {analysis.verified_weapon_count}\n"
            f"Candidate weapons (requires manual verification): {analysis.candidate_weapon_count}\n"
            f"Human Verification Decisions: {decisions_summary}\n"
            f"Severity Score: {analysis.severity_level.upper()} ({analysis.severity_score}/100)\n"
            f"Severity: {analysis.severity_level.upper()}\n"
            f"Suggested Category: {analysis.suggested_category.replace('_', ' ')}\n"
            f"Threat Present: {analysis.has_threat}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )
