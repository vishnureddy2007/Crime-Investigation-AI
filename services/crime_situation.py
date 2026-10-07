"""
AI-powered Crime Situation Analysis.

This service analyzes verified evidence, detections, and the timeline to
predict the likely activity pattern. It uses cautious, forensic terminology
to avoid false claims of guilt or certainty.
"""

from __future__ import annotations

import requests
import json
from datetime import datetime
from typing import Any

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SUMMARY_MAX_NEW_TOKENS,
)
from models.schemas import EvidenceAnalysis, CrimeSituationAnalysis

class CrimeSituationAnalyzer:
    """
    Analyzes crime scenes using a local LLM (Qwen3 14B).
    """

    def __init__(
        self,
        model_name: str | None = OLLAMA_MODEL,
        max_new_tokens: int = SUMMARY_MAX_NEW_TOKENS,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def is_available(self) -> bool:
        """Check if the local AI service is reachable."""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            return response.status_code == 200
        except (requests.RequestException, Exception):
            return False

    def analyze(self, analysis: EvidenceAnalysis, summary_text: str | None = None) -> CrimeSituationAnalysis:
        """
        Perform a situation analysis on the provided evidence. Reuses cached analysis if present.
        """
        import time
        from core.profiler import profile_stage
        t0 = time.perf_counter()

        try:
            import streamlit as st
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                existing = st.session_state.get("last_situation_analysis")
                if existing is not None and hasattr(existing, "likely_activity_pattern"):
                    profile_stage("qwen3_situation_analysis", time.perf_counter() - t0, "Reused Single-Pass Cached Result")
                    return existing
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

        if not self.is_available():
            profile_stage("qwen3_situation_analysis", time.perf_counter() - t0, "Ollama Offline - Fallback Used")
            return self._fallback_analysis(analysis)

        try:
            prompt = self._build_prompt(analysis, summary_text)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": "1h",
                "options": {
                    "num_predict": 300,
                    "temperature": 0.2,
                }
            }
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            result = response.json()

            raw_json = result.get("response", "").strip()
            if not raw_json:
                raise ValueError("AI returned an empty response body")

            if raw_json.startswith("```"):
                import re
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_json, re.DOTALL)
                if match:
                    raw_json = match.group(1)

            data = json.loads(raw_json)
            profile_stage("qwen3_situation_analysis", time.perf_counter() - t0, f"Qwen3 14B Success ({len(raw_json)} chars)")

            return CrimeSituationAnalysis(
                likely_activity_pattern=data.get("likely_activity_pattern", "Unknown"),
                possible_sequence_of_events=data.get("possible_sequence_of_events", "Unknown"),
                potential_next_activity=data.get("potential_next_activity", "Unknown"),
                suspicious_behavior_indicators=data.get("suspicious_behavior_indicators", []),
                risk_indicators=data.get("risk_indicators", []),
                supporting_evidence=data.get("supporting_evidence", []),
                confidence_level=data.get("confidence_level", "Low"),
                uncertainties=data.get("uncertainties", "No data"),
                alternative_explanations=data.get("alternative_explanations", "None"),
            )
        except Exception as e:
            from core.logging import get_logger
            get_logger(__name__).warning("AI Analysis Error: %s", e)
            profile_stage("qwen3_situation_analysis", time.perf_counter() - t0, f"Fallback Used ({e})")
            return self._fallback_analysis(analysis)

    def _fallback_analysis(self, analysis: EvidenceAnalysis) -> CrimeSituationAnalysis:
        """Deterministic fallback when AI is unavailable."""
        return CrimeSituationAnalysis(
            likely_activity_pattern="AI Analysis Unavailable",
            possible_sequence_of_events="Sequential evidence review required.",
            potential_next_activity="Human verification of detected objects.",
            suspicious_behavior_indicators=["Deterministic analysis based on detected objects"],
            risk_indicators=[f"Severity Level: {analysis.severity_level.upper()} ({analysis.severity_score}/100)"],
            supporting_evidence=[f"Verified weapons: {analysis.weapon_count}", f"Persons detected: {analysis.person_count}"],
            confidence_level="Deterministic Fallback",
            uncertainties="AI model unreachable or timing out during cold load.",
            alternative_explanations="Manual review of raw footage.",
        )

    def _build_prompt(self, analysis: EvidenceAnalysis, summary_text: str | None) -> str:
        """
        Build a strict prompt for the situation analysis.
        """
        facts = (
            f"Source: {analysis.source_name} ({analysis.source_type})\n"
            f"Verified Weapons: {analysis.weapon_count}\n"
            f"Candidate Weapons: {analysis.candidate_weapon_count}\n"
            f"Persons Detected: {analysis.person_count}\n"
            f"Vehicles: {analysis.vehicle_count}, Bags: {analysis.bag_count}\n"
            f"Severity: {analysis.severity_level} ({analysis.severity_score}/100)\n"
            f"Suggested Category: {analysis.suggested_category}"
        )

        prompt = (
            "You are a senior crime scene analyst. Analyze the provided evidence "
            "and predict the likely crime situation. \n\n"
            "CRITICAL RULES:\n"
            "1. Use cautious terminology (e.g., 'Potential situation', 'Evidence-supported possibility').\n"
            "2. NEVER state guilt or certainty (e.g., do NOT say 'the suspect definitely...').\n"
            "3. Use ONLY the provided facts. Do NOT invent people, weapons, or motives.\n"
            "4. Distinguish between VERIFIED evidence and POSSIBLE findings.\n\n"
            f"FACTS:\n{facts}\n\n"
        )

        if summary_text:
            prompt += f"\nNARRATIVE SUMMARY:\n{summary_text}\n\n"

        prompt += (
            "Return the result as a JSON object with these keys: "
            "likely_activity_pattern, possible_sequence_of_events, potential_next_activity, "
            "suspicious_behavior_indicators (list), risk_indicators (list), "
            "supporting_evidence (list), confidence_level (Low/Medium/High), "
            "uncertainties, alternative_explanations."
        )
        return prompt
