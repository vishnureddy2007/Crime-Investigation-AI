"""
Investigation summary generator.

This module generates a professional, structured forensic summary.
It uses a two-layer approach:
1. `build_template_summary()`: Deterministic, rule-based construction of all 11 points.
2. `SummaryGenerator.generate()`: Uses Qwen3 14B via Ollama to rewrite the
   template into a natural, professional investigative report.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import requests
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SUMMARY_MAX_NEW_TOKENS,
    SUMMARY_GENERATION_TIMEOUT_SEC,
)
from models.schemas import EvidenceAnalysis, DetailedNarrativeSummary

# ----------------------------------------------------------------------
# Deterministic template (Forensic Baseline)
# ----------------------------------------------------------------------
def build_template_summary(analysis: EvidenceAnalysis) -> DetailedNarrativeSummary:
    """
    Build a deterministic, structured forensic summary from the analysis.
    This ensures that even if AI fails, we have a factual, audit-ready baseline.
    """
    if analysis is None:
        # Return empty summary for None analysis
        return DetailedNarrativeSummary(
            case_id="unknown",
            case_overview="No case data provided.",
            evidence_reviewed="None",
            chronological_events="None",
            detected_objects="None",
            verified_findings="None",
            possible_findings="None",
            rejected_findings="None",
            potential_crime_activity="None",
            important_evidence="None",
            uncertainties="None",
            investigation_summary="No evidence available for summary."
        )

    # 1. Case Overview
    overview = (
        f"Analysis of {analysis.source_type} evidence source '{analysis.source_name}'. "
        f"The scene is classified as {analysis.severity_level.upper()} severity, "
        f"suggesting a {analysis.suggested_category.replace('_', ' ')} incident."
    )

    # 2. Evidence Reviewed
    reviewed = (
        f"Processed {analysis.source_type} source '{analysis.source_name}' "
        f"containing {analysis.frame_count} frame(s) of visual data."
    )

    # 3. Chronological Events (Deterministic sequence)
    events = [
        f"T+0: Evidence '{analysis.source_name}' acquired.",
        f"T+1: Automated detection run completed with {analysis.total_objects} objects."
    ]
    if analysis.weapon_count > 0:
        events.append(f"T+2: {analysis.weapon_count} weapon(s) identified.")
    if analysis.vehicle_count > 0:
        events.append(f"T+3: {analysis.vehicle_count} vehicle(s) identified.")
    if analysis.bag_count > 0:
        events.append(f"T+4: {analysis.bag_count} bag(s) identified.")
    events.append(f"T+5: Forensic analysis finalized.")
    chronological = " | ".join(events)

    # 4. Detected Objects
    objects = (
        f"Total objects detected: {analysis.total_objects}. "
        f"Breakdown: Persons({analysis.person_count}), "
        f"Verified Weapons({analysis.verified_weapon_count}), "
        f"Vehicles({analysis.vehicle_count}), Bags({analysis.bag_count})."
    )

    # 5. Verified Findings
    verified = []
    if analysis.verified_weapon_count > 0:
        verified.append(f"{analysis.verified_weapon_count} VERIFIED weapon(s)")
    if analysis.person_count > 0:
        verified.append(f"{analysis.person_count} person(s)")
    if not verified:
        verified_text = "No verified weapons detected."
    else:
        verified_text = "Verified: " + ", ".join(verified)

    # 6. Possible Findings
    possible = []
    if analysis.candidate_weapon_count > 0:
        possible.append(f"{analysis.candidate_weapon_count} unverified weapon candidate(s) (excluded from verified count)")
    if analysis.average_confidence < 0.6:
        possible.append("Low average detection confidence")
    if not possible:
        possible_text = "No ambiguous findings flagged."
    else:
        possible_text = "Possible: " + ", ".join(possible)

    # 7. Rejected Findings
    # (In a real system, we'd check human_review_stats. For now, we use a placeholder
    # or check if candidates were converted to non-weapons).
    rejected_text = "No findings explicitly rejected by human review in this analysis."

    # 8. Potential Crime Activity
    activity = (
        f"Based on the presence of {analysis.weapon_count} verified weapons "
        f"and {analysis.person_count} persons, the evidence is consistent with "
        f"{analysis.suggested_category.replace('_', ' ')}."
    )

    # 9. Important Evidence
    important = []
    if analysis.has_threat:
        important.append("Verified weapon detection (High Priority)")
    if analysis.severity_score > 70:
        important.append("High severity score")
    if not important:
        important_text = "No critical evidence flagged."
    else:
        important_text = "Key Evidence: " + ", ".join(important)

    # 10. Uncertainties
    uncertainties = []
    if analysis.candidate_weapon_count > 0:
        uncertainties.append(f"{analysis.candidate_weapon_count} weapon candidates require verification.")
    if analysis.average_confidence < 0.7:
        uncertainties.append("General detection confidence is moderate/low.")
    if not uncertainties:
        uncertainties_text = "No significant uncertainties identified."
    else:
        uncertainties_text = "Uncertainties: " + ", ".join(uncertainties)

    # 11. Investigation Summary
    summary_text = (
        f"The {analysis.source_type} evidence from {analysis.source_name} suggests a "
        f"{analysis.severity_level} severity incident. {analysis.person_count} person(s) "
        f"and {analysis.weapon_count} verified weapon(s) were detected. "
        f"The findings are consistent with {analysis.suggested_category.replace('_', ' ')}."
    )

    return DetailedNarrativeSummary(
        case_id=analysis.source_name,
        case_overview=overview,
        evidence_reviewed=reviewed,
        chronological_events=chronological,
        detected_objects=objects,
        verified_findings=verified_text,
        possible_findings=possible_text,
        rejected_findings=rejected_text,
        potential_crime_activity=activity,
        important_evidence=important_text,
        uncertainties=uncertainties_text,
        investigation_summary=summary_text
    )


# ----------------------------------------------------------------------
# Local AI-backed generator (Ollama)
# ----------------------------------------------------------------------
def calculate_analysis_hash(analysis: EvidenceAnalysis, human_decisions: dict[str, str] | None = None) -> str:
    """Compute a unique hash based on evidence data and human verification decisions."""
    import hashlib
    import json
    dec_str = json.dumps(sorted((human_decisions or {}).items()))
    data_str = (
        f"{analysis.source_name}_{analysis.person_count}_{analysis.verified_weapon_count}_"
        f"{analysis.candidate_weapon_count}_{analysis.severity_score}_"
        f"{analysis.suggested_category}_{dec_str}"
    )
    return hashlib.sha256(data_str.encode()).hexdigest()


class SummaryGenerator:
    """
    Generates a comprehensive forensic summary using a local LLM via Ollama.
    """

    def __init__(
        self,
        model_name: str | None = OLLAMA_MODEL,
        max_new_tokens: int = 350,
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

    def generate(
        self,
        analysis: EvidenceAnalysis,
        human_decisions: dict[str, str] | None = None,
        user_review_overrides: dict[str, str] | None = None,
    ) -> DetailedNarrativeSummary:
        """
        Generate a DetailedNarrativeSummary.
        Tries to use AI to rewrite the template into a professional report.
        Falls back to the template if AI fails.
        """
        from core.profiler import profile_stage
        t0 = time.perf_counter()
        template = build_template_summary(analysis)

        decisions = human_decisions or user_review_overrides

        if not self.is_available():
            profile_stage("qwen3_summary_generation", time.perf_counter() - t0, "Ollama Offline - Fallback Used")
            return template

        try:
            from core.profiler import record_ai_call
            version_hash = calculate_analysis_hash(analysis, decisions)
            record_ai_call(version_hash)

            prompt = self._build_prompt(template, analysis)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": "1h",
                "options": {
                    "num_predict": 450,
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

            import json
            import re
            raw_json = (result.get("response") or "").strip()

            if not raw_json:
                raise ValueError("Empty response from Ollama")

            # Strip markdown json blocks if present
            if raw_json.startswith("```"):
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_json, re.DOTALL)
                if match:
                    raw_json = match.group(1)

            data = json.loads(raw_json)

            profile_stage("qwen3_summary_generation", time.perf_counter() - t0, f"Qwen3 14B Success ({len(raw_json)} chars)")

            # Extract optional single-pass situation analysis if generated
            sit_data = data.get("situation_analysis")
            if sit_data and isinstance(sit_data, dict):
                try:
                    from models.schemas import CrimeSituationAnalysis
                    sit_obj = CrimeSituationAnalysis(
                        likely_activity_pattern=sit_data.get("likely_activity_pattern", "Likely suspicious movement."),
                        possible_sequence_of_events=sit_data.get("possible_sequence_of_events", "Sequence under investigation."),
                        potential_next_activity=sit_data.get("potential_next_activity", "Verification required."),
                        suspicious_behavior_indicators=sit_data.get("suspicious_behavior_indicators", [f"{analysis.weapon_count} weapon(s) detected"]),
                        risk_indicators=sit_data.get("risk_indicators", [f"Severity level {analysis.severity_level}"]),
                        supporting_evidence=sit_data.get("supporting_evidence", [f"Source: {analysis.source_name}"]),
                        confidence_level=sit_data.get("confidence_level", "Medium"),
                        uncertainties=sit_data.get("uncertainties", "None reported"),
                        alternative_explanations=sit_data.get("alternative_explanations", "Standard workflow"),
                    )
                    import streamlit as st
                    from streamlit.runtime.scriptrunner import get_script_run_ctx
                    if get_script_run_ctx() is not None:
                        st.session_state["last_situation_analysis"] = sit_obj
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass

            return DetailedNarrativeSummary(
                case_id=template.case_id,
                case_overview=data.get("case_overview", template.case_overview),
                evidence_reviewed=data.get("evidence_reviewed", template.evidence_reviewed),
                chronological_events=data.get("chronological_events", template.chronological_events),
                detected_objects=data.get("detected_objects", template.detected_objects),
                verified_findings=data.get("verified_findings", template.verified_findings),
                possible_findings=data.get("possible_findings", template.possible_findings),
                rejected_findings=data.get("rejected_findings", template.rejected_findings),
                potential_crime_activity=data.get("potential_crime_activity", template.potential_crime_activity),
                important_evidence=data.get("important_evidence", template.important_evidence),
                uncertainties=data.get("uncertainties", template.uncertainties),
                investigation_summary=data.get("investigation_summary", template.investigation_summary),
            )
        except Exception as exc:
            profile_stage("qwen3_summary_generation", time.perf_counter() - t0, f"Fallback Used ({exc})")
            return template

    def _build_prompt(self, template: DetailedNarrativeSummary, analysis: EvidenceAnalysis) -> str:
        """
        Build a concise prompt to rewrite the template into a professional report and situation prediction.
        """
        return (
            "You are a senior forensic investigator. Rewrite the following factual "
            "investigation points into a professional, formal narrative summary and situation prediction. "
            "Keep the tone clinical and objective. Do NOT invent any new facts, "
            "people, timestamps, or locations. Use ONLY the provided information.\n\n"
            f"FACTS:\n"
            f"- Case Overview: {template.case_overview}\n"
            f"- Evidence: {template.evidence_reviewed}\n"
            f"- Timeline: {template.chronological_events}\n"
            f"- Objects: {template.detected_objects}\n"
            f"- Verified: {template.verified_findings}\n"
            f"- Possible: {template.possible_findings}\n"
            f"- Summary: {template.investigation_summary}\n\n"
            "Return the result as a JSON object with keys: "
            "case_overview, evidence_reviewed, chronological_events, detected_objects, "
            "verified_findings, possible_findings, rejected_findings, potential_crime_activity, "
            "important_evidence, uncertainties, investigation_summary, and situation_analysis "
            "(containing likely_activity_pattern, possible_sequence_of_events, potential_next_activity, "
            "suspicious_behavior_indicators, risk_indicators, supporting_evidence, confidence_level, uncertainties)."
        )
