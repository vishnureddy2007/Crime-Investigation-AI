"""
AI Insights Hub — Crime Investigation AI.

Consolidates AI Summary (FLAN-T5 / Deterministic narrative), Chat Assistant (Interactive Q&A),
Crime Category Risk Prediction, and Temporal Crime Timeline into a single unified AI Hub.
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime

from config import REPORTS_DIR, DATABASE_PATH, SEVERITY_LEVEL_COLORS
from core.icons import ACTION_DOWNLOAD_DOCX, ACTION_DOWNLOAD_PDF, ICON_REPORT
from database.repository import get_human_review_stats
from models.report_generator import (
    DOCXReportGenerator,
    PDFReportGenerator,
    build_report_data,
)
from models.scene_planner import plan_scenes_from_narrative
from models.reconstruction import (
    render_storyboard_images,
    synthesize_reconstruction_video,
)
from models.schemas import EvidenceAnalysis, CrimeSituationAnalysis
from models.summary_generator import SummaryGenerator, build_template_summary
from pages._layout import render_page_header
from pages._state import resolve_active_analysis, resolve_batch_or_video, SessionKeys
from services.chat_assistant import ChatAssistant
from services.crime_prediction import predict_risk
from services.crime_situation import CrimeSituationAnalyzer
from services.animation_renderer import AnimationRenderer
from services.crime_timeline import build_timeline
from services.ai_health import get_ai_health


def _get_severity_badge(level: str | None) -> str:
    """Return a color-coded HTML badge for the severity level."""
    if not level:
        return '<span style="color: grey;">N/A</span>'
    lvl = level.lower()
    color = SEVERITY_LEVEL_COLORS.get(lvl, "#cbd5e1")
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 700;">{lvl.upper()}</span>'


def _get_base_image():
    """Pick the best available annotated image from session state."""
    # Simplified version of reconstruction.py logic
    last_video = st.session_state.get("last_video")
    if last_video is not None and hasattr(last_video, "keyframes") and last_video.keyframes:
        for kf in last_video.keyframes:
            if kf.detection and kf.detection.annotated_image is not None:
                return kf.detection.annotated_image

    last_detection = st.session_state.get("last_detection")
    if last_detection is not None and hasattr(last_detection, "annotated_image") and last_detection.annotated_image is not None:
        return last_detection.annotated_image

    return None


def _render_summary_tab(analysis: EvidenceAnalysis | None) -> None:
    """Tab 1: Unified Investigation Intelligence Report."""
    st.markdown("### 📑 Investigation Intelligence Report")
    st.caption("Combines formal forensic narratives with predictive situational analysis for a comprehensive case overview.")

    if analysis is None:
        st.warning("No active evidence analysis available. Process evidence in the Investigation workspace first.")
        return

    human_decisions = st.session_state.get("human_decisions", {})
    from models.summary_generator import calculate_analysis_hash
    current_hash = calculate_analysis_hash(analysis, human_decisions)

    # Header with severity badge
    st.markdown(f"**Current Analysis:** `{analysis.source_name}` { _get_severity_badge(analysis.severity_level) }", unsafe_allow_html=True)

    stored_hash = st.session_state.get("ai_analysis_hash")
    is_outdated = (stored_hash is not None and stored_hash != current_hash)

    if is_outdated:
        st.warning("⚠️ AI analysis is OUTDATED. Verified evidence or human review decisions changed since this report was generated. Please regenerate.")

    btn_label = "Regenerate AI Intelligence Report" if is_outdated else "Generate Full Intelligence Report"
    has_existing = st.session_state.get(SessionKeys.LAST_SUMMARY) is not None

    if not has_existing or is_outdated:
        if st.button(btn_label, type="primary", use_container_width=True):
            with st.spinner("🧠 Qwen3 14B (Local AI) is synthesizing narrative and predicting situational patterns..."):
                gen = SummaryGenerator()
                summary_obj = gen.generate(analysis)
                st.session_state[SessionKeys.LAST_SUMMARY] = summary_obj

                sit_analyzer = CrimeSituationAnalyzer()
                situation = sit_analyzer.analyze(analysis, summary_text=summary_obj.investigation_summary)
                st.session_state["last_situation_analysis"] = situation
                st.session_state["ai_analysis_hash"] = current_hash
                st.rerun()

    summary = st.session_state.get(SessionKeys.LAST_SUMMARY)
    situation = st.session_state.get("last_situation_analysis")

    if summary:
        if not is_outdated:
            st.success("✓ Intelligence Report Ready (Cached for Active Evidence State)")

        with st.container(border=True):
            st.markdown("#### 🖋️ Professional Investigation Narrative")

            points = [
                ("Case Overview", summary.case_overview),
                ("Evidence Reviewed", summary.evidence_reviewed),
                ("Chronological Events", summary.chronological_events),
                ("Detected Objects", summary.detected_objects),
                ("Verified Findings", summary.verified_findings),
                ("Possible Findings", summary.possible_findings),
                ("Rejected Findings", summary.rejected_findings),
                ("Potential Crime Activity", summary.potential_crime_activity),
                ("Important Evidence", summary.important_evidence),
                ("Uncertainties", summary.uncertainties),
                ("Final Investigation Summary", summary.investigation_summary),
            ]

            for label, text in points:
                st.markdown(f"**{label}:** {text}")

    if situation:
        with st.container(border=True):
            st.markdown("#### 🔮 Predictive Situation Analysis")
            st.markdown(f"**Likely Activity Pattern:** {situation.likely_activity_pattern}")
            st.markdown(f"**Possible Sequence of Events:** {situation.possible_sequence_of_events}")
            st.markdown(f"**Potential Next Activity:** {situation.potential_next_activity}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Suspicious Indicators:**")
                for ind in situation.suspicious_behavior_indicators:
                    st.markdown(f"- {ind}")
            with c2:
                st.markdown("**Risk Indicators:**")
                for risk in situation.risk_indicators:
                    st.markdown(f"- {risk}")

            st.markdown("---")
            st.markdown(f"**Supporting Evidence:** {', '.join(situation.supporting_evidence)}")
            st.markdown(f"**Confidence Level:** `{situation.confidence_level}`")
            st.markdown(f"**Uncertainties:** {situation.uncertainties}")

    if summary:
        st.markdown("---")
        if st.button("🚀 Generate 3D Reconstruction from Intelligence", type="primary"):
            with st.spinner("AI is planning the scene and synthesizing the video..."):
                try:
                    from services.scene_planner_service import ScenePlanner
                    from services.animation_renderer import AnimationRenderer
                    from config import VIDEOS_DIR

                    analysis = resolve_active_analysis()
                    if analysis is None:
                        st.error("No active analysis found.")
                        st.stop()

                    situation_obj = st.session_state.get("last_situation_analysis")
                    if situation_obj is None:
                        with st.spinner("Analyzing crime situation for scene planning..."):
                            sit_analyzer = CrimeSituationAnalyzer()
                            situation_obj = sit_analyzer.analyze(analysis, summary_text=summary.investigation_summary)
                            st.session_state["last_situation_analysis"] = situation_obj

                    planner = ScenePlanner()
                    scene_narrative = planner.plan_structured_scene(summary, situation_obj, analysis)

                    renderer = AnimationRenderer()
                    video_path = renderer.render_animated_video(
                        scene_narrative=scene_narrative,
                        analysis=analysis,
                        output_path=VIDEOS_DIR / f"reconstruction_3d_{analysis.source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    )

                    st.session_state["last_reconstruction_video_path"] = str(video_path)
                    st.session_state["reconstruction_auto_trigger"] = True

                    from database.repository import save_animation
                    from config import DATABASE_PATH

                    case_id = st.session_state.get("last_case_id")
                    if case_id:
                        payload = {
                            "summary_id": getattr(summary, "summary_id", "unknown"),
                            "situation_analysis": situation_obj.as_dict() if hasattr(situation_obj, "as_dict") else str(situation_obj),
                            "scene_narrative": scene_narrative.as_dict() if hasattr(scene_narrative, "as_dict") else str(scene_narrative),
                            "model": "Qwen3-14B",
                            "renderer": "Blender-FFmpeg-Pipeline-v1"
                        }
                        save_animation(DATABASE_PATH, case_id, str(video_path), payload)

                    st.success("Professional 3D Reconstruction generated successfully!")
                    st.session_state["nav_selection"] = "Reconstruction"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Reconstruction failed: {exc}")


def _render_chat_tab(analysis: EvidenceAnalysis | None) -> None:
    """Tab 2: Interactive Investigation Chat Assistant."""
    st.markdown("### Investigation Chat Assistant")
    st.caption("Ask questions grounded strictly in active case evidence and human verification records.")

    if analysis is None:
        st.warning("No active evidence available. Please process evidence in the Investigation workspace first.")
        return

    human_decisions = st.session_state.get("human_decisions", {})

    st.markdown("**Suggested Questions:**")
    q_cols = st.columns(4)
    preset_q = None
    with q_cols[0]:
        if st.button("What weapons were verified?", key="btn_q1", use_container_width=True):
            preset_q = "What weapons were verified?"
    with q_cols[1]:
        if st.button("Summarize the evidence", key="btn_q2", use_container_width=True):
            preset_q = "Summarize the evidence"
    with q_cols[2]:
        if st.button("What is the risk severity?", key="btn_q3", use_container_width=True):
            preset_q = "What is the risk severity?"
    with q_cols[3]:
        if st.button("Why was a detection rejected?", key="btn_q4", use_container_width=True):
            preset_q = "Why was a detection rejected?"

    if SessionKeys.CHAT_HISTORY not in st.session_state:
        st.session_state[SessionKeys.CHAT_HISTORY] = []

    history = st.session_state[SessionKeys.CHAT_HISTORY]

    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask a question about the active evidence...") or preset_q
    if user_query:
        history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        assistant = ChatAssistant(analysis=analysis)
        reply_obj = assistant.reply(user_query, human_decisions=human_decisions)
        response_text = reply_obj.answer
        history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.write(response_text)


def _render_prediction_tab(analysis: EvidenceAnalysis | None) -> None:
    """Tab 3: Category Risk Prediction."""
    st.markdown("### Category Risk & Prediction Matrix")
    st.caption("Predict risk levels across standardized crime categories based on evidence metrics.")

    if analysis is None:
        st.warning("No active evidence available. Process evidence in the Investigation workspace first.")
        return

    assessment = predict_risk(analysis)

    st.markdown("#### Predicted Category Risk Distribution")
    for category, score in assessment.scores.items():
        cat_name = category.replace("_", " ").title()
        percentage = score * 100

        with st.container(border=True):
            cols = st.columns([2, 1, 3])
            cols[0].markdown(f"**{cat_name}**")
            cols[1].markdown(f"**{percentage:.1f}%**")
            cols[2].progress(min(1.0, max(0.0, score)))


def _render_situation_tab(analysis: EvidenceAnalysis | None) -> None:
    """Tab 4: Crime Situation Analysis."""
    st.markdown("### AI Crime Situation Analysis")
    st.caption("Predictive insights into the likely sequence of events and activity patterns.")

    if analysis is None:
        st.warning("No active evidence available. Process evidence in the Investigation workspace first.")
        return

    summary = st.session_state.get(SessionKeys.LAST_SUMMARY)
    summary_text = getattr(summary, "investigation_summary", None) if summary else None

    if st.button("Analyze Crime Situation", type="primary"):
        with st.spinner("AI is analyzing evidence and predicting activity patterns..."):
            analyzer = CrimeSituationAnalyzer()
            situation = analyzer.analyze(analysis, summary_text=summary_text)
            st.session_state["last_situation_analysis"] = situation

    situation = st.session_state.get("last_situation_analysis")
    if situation:
        with st.container(border=True):
            st.markdown("#### 🔮 Predictive Situation Report")
            st.markdown(f"**Likely Activity Pattern:** {situation.likely_activity_pattern}")
            st.markdown(f"**Possible Sequence of Events:** {situation.possible_sequence_of_events}")
            st.markdown(f"**Potential Next Activity:** {situation.potential_next_activity}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Suspicious Indicators:**")
                for ind in situation.suspicious_behavior_indicators:
                    st.markdown(f"- {ind}")
            with c2:
                st.markdown("**Risk Indicators:**")
                for risk in situation.risk_indicators:
                    st.markdown(f"- {risk}")

            st.markdown("---")
            st.markdown(f"**Supporting Evidence:** {', '.join(situation.supporting_evidence)}")
            st.markdown(f"**Confidence Level:** `{situation.confidence_level}`")
            st.markdown(f"**Uncertainties:** {situation.uncertainties}")
            st.markdown(f"**Alternative Explanations:** {situation.alternative_explanations}")


def _render_animation_tab(analysis: EvidenceAnalysis | None) -> None:
    """Tab 5: AI Animated Crime Scene Visualization."""
    st.markdown("### 🎬 AI Animated Crime Scene Visualization")
    st.caption("Generates a data-driven animation of the incident based on verified evidence and AI analysis.")

    if analysis is None:
        st.warning("No active evidence available. Process evidence in the Investigation workspace first.")
        return

    case_id = st.session_state.get("last_case_id")
    if case_id is None:
        st.error("No active Case ID found. Please ensure a case is properly loaded.")
        return

    summary = st.session_state.get(SessionKeys.LAST_SUMMARY)
    situation = st.session_state.get("last_situation_analysis")

    if not summary:
        st.error("❌ A **Narrative Summary** is required before generating an animation. Please generate one in the 'AI Summary' tab.")
        return
    if not situation:
        st.error("❌ A **Crime Situation Analysis** is required. Please generate one in the 'Situation Analysis' tab.")
        return

    st.info("✅ All prerequisites met: Narrative Summary, Situation Analysis, and Verified Evidence are available.")

    from database.repository import load_animation, list_animations_for_case, save_animation
    from config import DATABASE_PATH

    animations = list_animations_for_case(DATABASE_PATH, case_id)
    last_anim = None
    if animations:
        last_anim_id = animations[0]["animation_id"]
        last_anim = load_animation(DATABASE_PATH, last_anim_id)

    if last_anim:
        if last_anim["is_outdated"]:
            st.warning("⚠️ This animation is based on an earlier investigation state. Evidence or human reviews have changed since it was generated.")

        st.markdown("#### 📺 Existing Animation")
        st.video(last_anim["video_path"])
        st.caption(f"Generated on: {last_anim.get('timestamp', 'Unknown')}")

    if st.button("Generate Animated Crime Scene", type="primary", use_container_width=True):
        with st.spinner("Rendering animated forensic visualization..."):
            try:
                from services.scene_planner_service import ScenePlanner
                from config import VIDEOS_DIR

                planner = ScenePlanner()
                scene_narrative = planner.plan_structured_scene(summary, situation, analysis)

                renderer = AnimationRenderer()
                video_path = renderer.render_animated_video(
                    scene_narrative=scene_narrative,
                    analysis=analysis,
                    output_path=VIDEOS_DIR / f"animation_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                )

                payload = {
                    "summary_id": getattr(summary, "summary_id", "unknown"),
                    "situation_analysis": situation.as_dict() if hasattr(situation, "as_dict") else str(situation),
                    "scene_narrative": scene_narrative.as_dict(),
                    "model": "Qwen3-14B",
                    "renderer": "ForensicAnimationRenderer_v1"
                }
                save_animation(DATABASE_PATH, case_id, str(video_path), payload)

                st.session_state["last_animation_video_path"] = str(video_path)
                st.success("Animation generated and saved to case successfully!")
                st.rerun()
            except Exception as exc:
                st.error(f"Animation generation failed: {exc}")


def _render_timeline_tab(analysis: EvidenceAnalysis | None) -> None:
    """Tab 4: Temporal Crime Timeline."""
    st.markdown("### Crime Event Timeline")
    st.caption("Ordered temporal breakdown of investigative events derived from active evidence.")

    if analysis is None:
        st.info("No active evidence available to build timeline. Process evidence in the Investigation workspace first.")
        return

    events = build_timeline(analysis)
    st.markdown(f"**Total Timeline Events:** {events.count}")

    for ev in events.events:
        sev_tag = ev.severity.upper()
        with st.container(border=True):
            st.markdown(f"**[{ev.timestamp.strftime('%H:%M:%S')}] {ev.label}** (Status: `{sev_tag}`)")
            st.write(ev.detail)


def render() -> None:
    """Render the AI Insights page."""
    render_page_header(
        "🤖 AI Insights & Assistant",
        subtitle="AI Insights Hub — AI Narrative Summaries, Interactive Q&A Assistant, Risk Prediction, and Temporal Timeline.",
    )

    with st.expander("🛠️ AI System Health & Developer Diagnostics", expanded=False):
        health = get_ai_health()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ollama", health["connection"])
        col2.metric("Model", health["model_available"])
        col3.metric("Generation", health["generation"])
        col4.metric("Parsing", health["parsing"])

        st.markdown("---")
        st.markdown("**Pipeline Performance Profile (Live Measurements):**")
        from core.profiler import get_performance_report
        st.code(get_performance_report(), language="text")

    analysis = resolve_active_analysis()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📑 Investigation Intelligence",
        "💬 Chat Assistant",
        "🔮 Category Risk Prediction",
        "📉 Situation Analysis",
        "🎬 Animation",
        "⏱️ Event Timeline",
    ])

    with tab1:
        _render_summary_tab(analysis)

    with tab2:
        _render_chat_tab(analysis)

    with tab3:
        _render_prediction_tab(analysis)

    with tab4:
        _render_situation_tab(analysis)

    with tab5:
        _render_animation_tab(analysis)

    with tab6:
        _render_timeline_tab(analysis)
