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

            seq = data.get("possible_sequence_of_events", [])
            seq_str = "\n".join(seq) if isinstance(seq, list) else str(seq)

            return CrimeSituationAnalysis(
                likely_activity_pattern=data.get("likely_activity_pattern") or data.get("predicted_summary", "Situation observed"),
                possible_sequence_of_events=seq_str or "Evidence sequence recorded.",
                potential_next_activity=data.get("potential_next_activity", "Further evidence review"),
                suspicious_behavior_indicators=data.get("suspicious_behavior_indicators") or data.get("suspicious_indicators", []),
                risk_indicators=data.get("risk_indicators", []),
                supporting_evidence=data.get("supporting_evidence") or data.get("important_evidence", []),
                confidence_level=data.get("confidence_level", "High"),
                uncertainties=data.get("uncertainties", "None reported"),
                alternative_explanations=data.get("alternative_explanations", "Standard scenario"),
                source="qwen3_14b",
            )
        except Exception as e:
            from core.logging import get_logger
            get_logger(__name__).warning("AI Analysis Error: %s", e)
            profile_stage("qwen3_situation_analysis", time.perf_counter() - t0, f"Fallback Used ({e})")
            return self._fallback_analysis(analysis)

    def _fallback_analysis(self, analysis: EvidenceAnalysis) -> CrimeSituationAnalysis:
        """Deterministic, evidence-grounded local prediction when Qwen is unavailable."""
        from services.prediction_engine import generate_local_prediction
        pred = generate_local_prediction(analysis)
        seq_str = "\n".join(pred.possible_sequence_of_events) if isinstance(pred.possible_sequence_of_events, list) else str(pred.possible_sequence_of_events)
        return CrimeSituationAnalysis(
            likely_activity_pattern=pred.likely_activity_pattern or pred.predicted_summary,
            possible_sequence_of_events=seq_str,
            potential_next_activity=pred.potential_next_activity,
            suspicious_behavior_indicators=pred.suspicious_indicators,
            risk_indicators=pred.risk_indicators,
            supporting_evidence=pred.important_evidence or [f"Verified weapons: {analysis.weapon_count}", f"Persons detected: {analysis.person_count}"],
            confidence_level=pred.confidence_level,
            uncertainties="; ".join(pred.uncertainties) if isinstance(pred.uncertainties, list) else str(pred.uncertainties),
            alternative_explanations="Review of visual evidence by lead investigator.",
            source="local_evidence_engine",
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
