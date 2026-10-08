"""
Investigation Video page.

Generates and previews the 2D forensic investigation explanation MP4 video.
Displays video status (NOT GENERATED, READY, OUTDATED, FAILED), metadata,
video player, download options, and regeneration triggers.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from config import DATABASE_PATH, VIDEOS_DIR
from core.icons import ICON_RECONSTRUCTION
from database.repository import (
    delete_investigation_video,
    get_case_evidence_version,
    get_latest_analysis,
    get_latest_investigation_video,
    list_cases,
)
from models.schemas import EvidenceAnalysis
from pages._layout import empty_state, friendly_error, render_page_header
from pages._state import resolve_active_analysis, resolve_batch_or_detection_image
from services.investigation_video_generator import InvestigationVideoGenerator


def render() -> None:
    render_page_header(
        title="🎥 Investigation Video",
        subtitle="Generate, preview, and export 2D forensic investigation explanation videos.",
    )

    cases = list_cases(DATABASE_PATH)
    if not cases:
        empty_state(
            "No active cases found.",
            "Process evidence in the Investigation workspace first to generate a case and investigation video.",
            action_label="Open Investigation Workspace",
            action_target="Investigation",
        )
        return

    # Select Case
    case_options = {f"Case #{c['case_id']:03d} - {c['source_name']}": c["case_id"] for c in cases}
    selected_label = st.selectbox("Select Case for Video Generation", list(case_options.keys()))
    case_id = case_options[selected_label]

    # Get analysis & video record
    db_analysis = get_latest_analysis(DATABASE_PATH, case_id)
    video_rec = get_latest_investigation_video(DATABASE_PATH, case_id)
    evidence_version = get_case_evidence_version(DATABASE_PATH, case_id)

    # Determine status
    if video_rec is None:
        status = "NOT GENERATED"
    elif video_rec.get("status") == "OUTDATED" or video_rec.get("evidence_version", 1) < evidence_version:
        status = "OUTDATED"
    else:
        status = video_rec.get("status", "READY")

    # Status Banner Card
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Case ID", f"#{case_id:03d}")
    with col2:
        if status == "READY":
            st.success("🟢 STATUS: READY")
        elif status == "OUTDATED":
            st.warning("⚠️ STATUS: OUTDATED")
        else:
            st.info("⚪ STATUS: NOT GENERATED")
    with col3:
        dur = float(video_rec.get("duration", 0.0)) if video_rec else 0.0
        st.metric("Video Duration", f"{dur:.1f}s" if dur > 0 else "N/A")
    with col4:
        st.metric("Evidence Version", f"v{evidence_version}")

    if status == "OUTDATED":
        st.warning("⚠️ Evidence or human reviews have updated since this video was generated. Please regenerate to reflect the latest evidence.")

    # Load Evidence Collection Preview
    generator = InvestigationVideoGenerator(db_path=DATABASE_PATH)
    collection = generator.get_case_evidence_collection(case_id)

    st.markdown("---")
    st.subheader("Case Evidence Overview")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.metric("Evidence Items", len(collection["inventory"]))
    with mc2:
        st.metric("Images", len(collection["images"]))
    with mc3:
        st.metric("Videos", len(collection["videos"]))
    with mc4:
        st.metric("Verified Weapons", collection["verified_weapon_count"])
    with mc5:
        st.metric("Persons", collection["person_count"])

    # Video Generator Controls
    st.markdown("---")
    st.subheader("Video Generator Settings")
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        detail_level = st.selectbox(
            "Video Detail Level",
            ["Detailed", "Standard", "Brief"],
            index=0,
            help="Detailed: Includes all evidence, overlays, timeline, AI assessment & summaries.",
        )
    with c_opt2:
        video_mode = st.selectbox(
            "Video Evidence Mode",
            ["Representative Frames", "Short Evidence Clips", "Full Video"],
            index=0,
            help="Representative Frames samples keyframes across uploaded videos; Short Evidence Clips embeds clips around detection timestamps.",
        )

    # Evidence Selection Checkboxes
    selected_ev_ids = None
    if collection["inventory"]:
        with st.expander("🔍 Filter Evidence Items to Include (Optional)", expanded=False):
            all_ev = st.checkbox("☑ Include All Evidence Files", value=True)
            if not all_ev:
                selected_ev_ids = []
                for item in collection["inventory"]:
                    chk = st.checkbox(f"Include {item['evidence_id']} — {item['filename']} ({item['file_type'].upper()})", value=True)
                    if chk:
                        selected_ev_ids.append(item["evidence_id"])

    st.markdown("---")

    # Action Buttons
    b_col1, b_col2, b_col3 = st.columns([2, 2, 2])

    with b_col1:
        btn_label = "Regenerate Video" if status == "READY" else "Generate Investigation Video"
        if st.button(btn_label, type="primary", use_container_width=True):
            with st.spinner("Generating evidence-grounded 2D forensic investigation explanation video..."):
                try:
                    video_path, total_duration = generator.generate_video(
                        case_id=case_id,
                        force=True,
                        selected_evidence_ids=selected_ev_ids,
                        detail_level=detail_level,
                        video_evidence_mode=video_mode,
                    )
                    st.session_state["last_investigation_video_path"] = str(video_path)
                    st.success(f"Generated investigation video ({total_duration:.1f}s).")
                    st.rerun()
                except Exception as exc:
                    friendly_error(exc, fallback_title="Video generation failed.")

    with b_col2:
        if video_rec and video_rec.get("video_path") and Path(video_rec["video_path"]).exists():
            v_path = Path(video_rec["video_path"])
            video_bytes = v_path.read_bytes()
            st.download_button(
                label="📥 Download Video (MP4)",
                data=video_bytes,
                file_name=v_path.name,
                mime="video/mp4",
                use_container_width=True,
            )

    with b_col3:
        if video_rec:
            if st.button("🗑️ Delete Video", type="secondary", use_container_width=True):
                delete_investigation_video(DATABASE_PATH, video_rec["id"])
                if video_rec.get("video_path") and Path(video_rec["video_path"]).exists():
                    try:
                        Path(video_rec["video_path"]).unlink()
                    except Exception:
                        pass
                st.success("Video deleted.")
                st.rerun()

    # Video Preview Player
    st.subheader("Video Preview")
    current_video_path = video_rec.get("video_path") if video_rec else st.session_state.get("last_investigation_video_path")

    if current_video_path and Path(current_video_path).exists():
        video_bytes = Path(current_video_path).read_bytes()
        st.video(video_bytes)
        st.caption(f"File Path: `{current_video_path}`")
    else:
        st.info("No video available to preview. Click 'Generate Investigation Video' above.")

    # Information Card
    with st.expander("ℹ️ About Investigation Video Generation"):
        st.markdown(
            """
            - **Python-native Lightweight Engine**: Uses OpenCV and Pillow (no external 3D software required).
            - **Evidence-Grounded**: Integrates original uploaded evidence photos, YOLO object detections, weapon verification, and Qwen3/local AI narrative summaries.
            - **Visual Effects**: Applies smooth Ken Burns zoom/pan effects, dark forensic overlay cards, clean typography, and timeline visualization.
            - **Offline Capable**: Runs completely locally without cloud API dependencies.
            """
        )
