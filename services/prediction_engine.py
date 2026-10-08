"""
Local Evidence Prediction Engine.

Deterministic, evidence-grounded rule engine that generates structured predicted
situation summaries, sequences of events, risk indicators, and animation plans
directly from verified evidence without requiring external LLM / Qwen models.

Design Rules:
1. Pure Python heuristic engine — 0 dependencies on external network or Ollama.
2. Cautious, forensic tone ("Evidence suggests...", "The observed pattern is consistent with...").
3. NEVER claims guilt, certainty, or unobserved facts (identities, motives, unobserved items).
4. NEVER returns generic placeholders ("AI Analysis Unavailable", "Human verification of detected objects").
5. Produces the exact schema compatible with UnifiedAIAnalysis.
"""

from __future__ import annotations

from typing import Any
from models.schemas import EvidenceAnalysis, UnifiedAIAnalysis


def generate_local_prediction(
    analysis: EvidenceAnalysis | None,
    case_context: dict[str, Any] | None = None,
    case_id: str = "case_001",
    evidence_version: int = 1,
) -> UnifiedAIAnalysis:
    """
    Generate an evidence-grounded predicted situation summary and analysis.
    """
    if analysis is None:
        return UnifiedAIAnalysis(
            source="local_evidence_engine",
            case_id=case_id,
            evidence_version=evidence_version,
            case_summary="No evidence media provided for analysis.",
            investigation_narrative="Insufficient evidence source file loaded.",
            situation_analysis="No visual evidence has been submitted.",
            likely_activity_pattern="Insufficient verified evidence to determine a reliable activity pattern.",
            predicted_summary="Insufficient evidence available for situation summary.",
            possible_sequence_of_events=["Evidence media upload required."],
            potential_next_activity="Upload evidence media for automated verification.",
            suspicious_indicators=[],
            risk_indicators=["No evidence analyzed"],
            risk_level="LOW",
            confidence_level="Low",
            important_evidence=[],
            uncertainties=["No visual media loaded in the investigation workspace."],
            limitations=["Analysis requires valid image or video evidence."],
            animation_scene_description="Empty scene. Awaiting evidence upload.",
            animation_events=[],
        )

    v_weapons = getattr(analysis, "verified_weapon_count", getattr(analysis, "weapon_count", 0))
    persons = getattr(analysis, "person_count", 0)
    vehicles = getattr(analysis, "vehicle_count", 0)
    bags = getattr(analysis, "bag_count", 0)
    severity = getattr(analysis, "severity_score", 0)
    severity_lvl = getattr(analysis, "severity_level", "low").upper()
    category = getattr(analysis, "suggested_category", "unknown").replace("_", " ").title()
    source_type = getattr(analysis, "source_type", "image")
    source_name = getattr(analysis, "source_name", "evidence")
    avg_conf = getattr(analysis, "average_confidence", 0.0)

    # --- Confidence Level Heuristic ---
    if v_weapons > 0 and persons > 0 and avg_conf >= 0.70:
        conf_level = "High"
    elif (v_weapons > 0 or persons >= 2) and avg_conf >= 0.50:
        conf_level = "Medium"
    else:
        conf_level = "Low"

    # --- Risk Level ---
    if severity >= 75 or v_weapons >= 2:
        risk_lvl = "CRITICAL"
    elif severity >= 50 or v_weapons == 1:
        risk_lvl = "HIGH"
    elif severity >= 25 or persons >= 2 or bags >= 1:
        risk_lvl = "MEDIUM"
    else:
        risk_lvl = "LOW"

    # --- Likely Activity Pattern ---
    if v_weapons > 0 and persons >= 2:
        activity_pattern = f"Potential armed interaction involving multiple individuals ({category})."
    elif v_weapons > 0 and persons == 1:
        activity_pattern = f"Potential presence or handling of a weapon by an individual ({category})."
    elif v_weapons > 0 and persons == 0:
        activity_pattern = f"Presence of verified weapon in scene ({category})."
    elif persons >= 2 and bags >= 1:
        activity_pattern = f"Multi-person interaction with potential property transfer or theft indicator ({category})."
    elif persons >= 2:
        activity_pattern = f"Interaction involving multiple persons at the scene ({category})."
    elif vehicles >= 1 and persons >= 1:
        activity_pattern = f"Vehicle activity with person presence in scene ({category})."
    elif bags >= 1:
        activity_pattern = f"Object or property presence detected at scene ({category})."
    else:
        activity_pattern = f"General scene observation ({category})."

    # --- Predicted Situation Summary ---
    summary_parts = []
    summary_parts.append(
        f"The verified evidence from '{source_name}' indicates a {risk_lvl.lower()}-risk "
        f"situation ({severity}/100 severity) categorized under {category}."
    )

    if v_weapons > 0 and persons > 0:
        summary_parts.append(
            f"Automated verification confirmed {v_weapons} weapon(s) in proximity to {persons} person(s). "
            f"The combination of verified weapon presence, person density, and critical severity points to "
            f"a possible armed confrontation or high-risk encounter."
        )
    elif v_weapons > 0:
        summary_parts.append(
            f"Automated verification confirmed {v_weapons} weapon(s) at the scene. "
            f"Weapon presence elevates the potential threat assessment."
        )
    elif persons > 0:
        summary_parts.append(
            f"The visual evidence records {persons} person(s) at the scene. "
            f"No verified weapons were identified in the primary scan."
        )

    if vehicles > 0 or bags > 0:
        details = []
        if vehicles: details.append(f"{vehicles} vehicle(s)")
        if bags: details.append(f"{bags} bag(s)")
        summary_parts.append(f"Additional scene context includes {', '.join(details)}.")

    predicted_summary = " ".join(summary_parts)

    # --- Possible Sequence of Events ---
    sequence: list[str] = []
    step = 1

    if source_type == "video":
        sequence.append(f"{step}. Video evidence sample acquired and processed across keyframes.")
        step += 1
    else:
        sequence.append(f"{step}. Image evidence source '{source_name}' ingested into analysis pipeline.")
        step += 1

    if persons > 0:
        sequence.append(f"{step}. {persons} person(s) entered/located within the recorded scene.")
        step += 1

    if v_weapons > 0:
        sequence.append(f"{step}. Automated weapon verification identified {v_weapons} weapon(s) at the scene.")
        step += 1

    if bags > 0 or vehicles > 0:
        items = []
        if bags: items.append("bag(s)")
        if vehicles: items.append("vehicle(s)")
        sequence.append(f"{step}. Secondary objects ({', '.join(items)}) recorded in scene spatial bounds.")
        step += 1

    if v_weapons > 0 and persons >= 2:
        sequence.append(f"{step}. The evidence pattern is consistent with an escalating armed interaction.")
    elif v_weapons > 0:
        sequence.append(f"{step}. The presence of a verified weapon establishes a potential threat condition.")
    else:
        sequence.append(f"{step}. Routine or suspicious activity observed without confirmed weapon escalation.")

    # --- Potential Next Activity ---
    if v_weapons > 0 and persons >= 2:
        potential_next = (
            "Further interaction, movement, or resolution between the observed persons may occur. "
            "Forensic review of secondary angles or timestamps is recommended."
        )
    elif v_weapons > 0:
        potential_next = (
            "Movement or handling of the verified weapon within the location may occur. "
            "Sequential frame tracking is suggested."
        )
    elif persons >= 1:
        potential_next = (
            "Continued movement or interaction within the scene space may take place based on observed positioning."
        )
    else:
        potential_next = (
            "Static scene observation. Additional media ingestion recommended to establish temporal progression."
        )

    # --- Suspicious & Risk Indicators ---
    suspicious: list[str] = []
    risk_ind: list[str] = []

    if v_weapons > 0:
        suspicious.append(f"{v_weapons} verified weapon(s) in scene")
        risk_ind.append(f"Armed threat condition ({v_weapons} verified weapon(s))")
    if persons >= 2:
        suspicious.append(f"Multiple persons present simultaneously ({persons})")
        risk_ind.append(f"Multi-person encounter density ({persons} persons)")
    if severity >= 50:
        risk_ind.append(f"Elevated scene severity score ({severity}/100 - {severity_lvl})")
    if bags > 0:
        suspicious.append(f"Bag/container present ({bags} - possible theft or concealment indicator)")
    if vehicles > 0:
        suspicious.append(f"Vehicle proximity ({vehicles} vehicle(s))")

    if not suspicious:
        suspicious.append("No immediate suspicious behavioral indicators flagged in primary scan")
    if not risk_ind:
        risk_ind.append(f"Baseline risk level ({severity_lvl})")

    # --- Important Evidence & Uncertainties ---
    important_ev: list[str] = []
    if v_weapons > 0:
        important_ev.append(f"Confirmed weapon detection ({v_weapons} item(s)) - High Priority")
    if persons > 0:
        important_ev.append(f"Person localization ({persons} individual(s))")
    if severity >= 70:
        important_ev.append("Critical severity assessment trigger")

    uncertainties: list[str] = [
        "Visual evidence does not establish individual intent, identities, or verbal communication.",
        "Out-of-frame activity prior to or following the recorded sample cannot be determined from single media source.",
    ]

    limitations: list[str] = [
        "Analysis based exclusively on visual feature extraction and automated object detection.",
        "Audio and biometric verification not included in visual dataset.",
    ]

    # --- Animation Scene Description & Events ---
    anim_desc = f"Reconstructed 3D crime scene layout: {persons} person(s), {v_weapons} weapon(s), {vehicles} vehicle(s), {bags} bag(s)."
    anim_events: list[dict[str, Any]] = []

    t_sec = 0.0
    for i in range(persons):
        anim_events.append({
            "timestamp": t_sec,
            "actor_id": f"person_{i+1}",
            "action": "APPEAR",
            "label": f"Person #{i+1}",
            "status": "VERIFIED",
            "description": f"Person #{i+1} detected in scene bounds"
        })
        t_sec += 1.0

    for i in range(v_weapons):
        anim_events.append({
            "timestamp": t_sec,
            "actor_id": f"weapon_{i+1}",
            "action": "HIGHLIGHT",
            "label": f"Weapon #{i+1}",
            "status": "VERIFIED",
            "description": f"Verified weapon #{i+1} flagged"
        })
        t_sec += 1.0

    narrative = f"Investigative report for '{source_name}': {predicted_summary}"
    sit_analysis = f"Likely Activity: {activity_pattern}\nRisk: {risk_lvl} ({severity}/100)\nNext Activity: {potential_next}"

    return UnifiedAIAnalysis(
        source="local_evidence_engine",
        case_id=case_id,
        evidence_version=evidence_version,
        case_summary=predicted_summary,
        investigation_narrative=narrative,
        situation_analysis=sit_analysis,
        likely_activity_pattern=activity_pattern,
        predicted_summary=predicted_summary,
        possible_sequence_of_events=sequence,
        potential_next_activity=potential_next,
        suspicious_indicators=suspicious,
        risk_indicators=risk_ind,
        risk_level=risk_lvl,
        confidence_level=conf_level,
        important_evidence=important_ev,
        uncertainties=uncertainties,
        limitations=limitations,
        animation_scene_description=anim_desc,
        animation_events=anim_events,
    )
