"""
Primary Investigation Workspace — Crime Investigation AI.

Consolidates evidence upload (images and CCTV videos), multi-source object and weapon
detection (YOLOv8), keyframe extraction, object breakdown, and structured evidence analysis.
"""

from __future__ import annotations

import streamlit as st

from config import (
    MAX_VIDEO_SIZE_MB,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
)
from core.icons import (
    ICON_INVESTIGATION,
    ICON_EVIDENCE_ANALYSIS,
    SECTION_COMBINED,
    SECTION_PER_FILE,
)
from models.batch_processor import BatchEvidenceProcessor, FileEvidence
from models.evidence_analyzer import analyze
from models.schemas import AnalysisInput, EvidenceAnalysis
from models.yolo_detector import get_multi_source_detector
from pages._layout import friendly_error, render_page_header, status_pill
from pages._state import resolve_active_analysis, SessionKeys
from utils.db_hooks import auto_save_last_analysis
from utils.image_io import is_allowed_image
from utils.video_io import is_allowed_video


def _get_processor() -> BatchEvidenceProcessor:
    """Return a batch processor wired to the multi-source detector."""
    detector = get_multi_source_detector()
    return BatchEvidenceProcessor(detector=detector)


def _render_file_evidence_card(file_evidence: FileEvidence) -> None:
    """Render detection details for a processed image or video file."""
    header = f"Evidence File: {file_evidence.filename}"
    if file_evidence.source_type == "image":
        header += " (Image)"
    elif file_evidence.source_type == "video":
        header += " (Video)"
    st.subheader(header)

    if file_evidence.error is not None:
        st.warning(
            f"Could not process `{file_evidence.filename}`\n\n"
            f"**Reason:** {file_evidence.error.get('reason', 'Processing failed')}\n\n"
            f"**Details:** {file_evidence.error.get('message', '')}"
        )
        return

    human_decisions = st.session_state.setdefault("human_decisions", {})

    if file_evidence.detection is not None:
        d = file_evidence.detection
        if d.annotated_image is not None:
            st.image(
                d.annotated_image,
                caption=f"Annotated Detection: {d.count} object(s)",
                use_container_width=True,
            )

        # Count metrics (Section 7: Separate raw, candidate, verified)
        v_count = sum(1 for det in d.detections if det.weapon_status == "verified" and human_decisions.get(f"{det.label}@{det.confidence:.2f}") != "REJECT")
        c_count = sum(1 for det in d.detections if det.weapon_status == "candidate" and human_decisions.get(f"{det.label}@{det.confidence:.2f}") != "REJECT")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Persons", file_evidence.person_count)
        c2.metric("Verified Weapons", v_count)
        c3.metric("Vehicles", file_evidence.vehicle_count)
        c4.metric("Bags", file_evidence.bag_count)

        # HITL Weapon Verification Section (Section 15)
        weapon_dets = [det for det in d.detections if det.label in {"weapon", "knife", "candidate_weapon"} or "weapon" in det.class_name.lower()]
        if weapon_dets:
            st.markdown("#### 🛡️ Human-in-the-Loop (HITL) Weapon Verification")
            st.caption("Review detected weapon candidates. Your confirmation or rejection updates final evidence, reports, and 3D scene reconstruction.")

            for idx, w_det in enumerate(weapon_dets, start=1):
                key = f"{w_det.label}@{w_det.confidence:.2f}"
                decision = human_decisions.get(key)

                with st.container(border=True):
                    w_col1, w_col2, w_col3 = st.columns([3, 2, 3])

                    with w_col1:
                        st.markdown(f"**Weapon Detection #{idx}**")
                        st.markdown(f"Type: `{w_det.class_name.title()}` | Conf: `{w_det.confidence * 100:.1f}%`")
                        st.markdown(f"Source Model: `{w_det.source}`")

                    with w_col2:
                        if decision == "CONFIRM":
                            st.markdown("🟢 **Status: VERIFIED WEAPON**")
                        elif decision == "REJECT":
                            st.markdown("🔴 **Status: REJECTED**")
                            st.caption("Excluded from final report & 3D scene")
                        elif w_det.weapon_status == "verified":
                            st.markdown("🟢 **Status: VERIFIED WEAPON**")
                        else:
                            st.markdown("🟡 **Status: PENDING VERIFICATION**")

                    with w_col3:
                        b_col1, b_col2 = st.columns(2)
                        with b_col1:
                            if st.button("CONFIRM", key=f"btn_confirm_{idx}_{key}", use_container_width=True, type="primary" if decision != "CONFIRM" else "secondary"):
                                human_decisions[key] = "CONFIRM"
                                active_analysis = resolve_active_analysis()
                                if active_analysis:
                                    from models.weapon_verifier import apply_human_review_to_analysis
                                    updated = apply_human_review_to_analysis(active_analysis, human_decisions)
                                    st.session_state[SessionKeys.LAST_ANALYSIS] = updated
                                st.rerun()

                        with b_col2:
                            if st.button("REJECT", key=f"btn_reject_{idx}_{key}", use_container_width=True, type="secondary"):
                                human_decisions[key] = "REJECT"
                                active_analysis = resolve_active_analysis()
                                if active_analysis:
                                    from models.weapon_verifier import apply_human_review_to_analysis
                                    updated = apply_human_review_to_analysis(active_analysis, human_decisions)
                                    st.session_state[SessionKeys.LAST_ANALYSIS] = updated
                                st.rerun()

    elif file_evidence.video is not None:
        v = file_evidence.video
        st.info(f"Video duration: {v.duration_sec:.1f}s | Keyframes extracted: {len(v.keyframes)}")
        if v.keyframes:
            st.markdown("**Extracted Keyframes:**")
            cols = st.columns(min(len(v.keyframes), 4))
            for idx, kf in enumerate(v.keyframes):
                col = cols[idx % len(cols)]
                with col:
                    if kf.detection and kf.detection.annotated_image:
                        st.image(
                            kf.detection.annotated_image,
                            caption=f"t={kf.timestamp_sec:.1f}s ({kf.detection.count} obj)",
                            use_container_width=True,
                        )


def _render_upload_and_detect_tab() -> None:
    """Tab 1: Evidence Upload & Detection Workspace."""
    st.markdown("### Upload & Detect Evidence")
    st.caption("Upload single or multiple crime-scene images and/or CCTV video files for automated detection.")

    col1, col2 = st.columns(2)
    with col1:
        conf = st.slider(
            "Confidence threshold",
            min_value=0.10, max_value=0.90,
            value=YOLO_CONFIDENCE_THRESHOLD, step=0.05,
            key="inv_conf_slider",
        )
    with col2:
        iou = st.slider(
            "IoU threshold",
            min_value=0.10, max_value=0.90,
            value=YOLO_IOU_THRESHOLD, step=0.05,
            key="inv_iou_slider",
        )

    images = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="inv_images_uploader",
    )

    videos = st.file_uploader(
        "Upload CCTV Videos (max 100MB)",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        accept_multiple_files=True,
        key="inv_videos_uploader",
    )

    images_list = images or []
    videos_list = videos or []

    if not images_list and not videos_list:
        st.info("Select one or more images or video files above to begin investigation.")
        return

    st.markdown(f"**Selected:** {len(images_list)} image(s), {len(videos_list)} video(s)")

    if st.button("Run AI Detection & Analysis", type="primary", use_container_width=True):
        processor = _get_processor()
        with st.spinner("Processing evidence files with YOLOv8 multi-source detector..."):
            image_inputs = [(up.name, up.getvalue()) for up in images_list]
            video_inputs = [(up.name, up.getvalue()) for up in videos_list]
            all_files = image_inputs + video_inputs

            batch_result = processor.process(
                files=all_files,
                confidence=conf,
                iou=iou,
            )

            st.session_state[SessionKeys.LAST_BATCH_RESULT] = batch_result
            if batch_result.combined_analysis:
                st.session_state[SessionKeys.LAST_ANALYSIS] = batch_result.combined_analysis
                auto_save_last_analysis(batch_result.combined_analysis)

            st.success(f"Processing complete! Processed {len(batch_result.files)} file(s).")

    # Render results if available
    batch_res = st.session_state.get(SessionKeys.LAST_BATCH_RESULT)
    if batch_res is not None and batch_res.files:
        st.markdown("---")
        st.markdown("### Detection Results")
        for f_ev in batch_res.files:
            with st.container(border=True):
                _render_file_evidence_card(f_ev)


def _render_evidence_analysis_tab() -> None:
    """Tab 2: Structured Evidence Analysis View."""
    st.markdown("### Structured Evidence Analysis")
    st.caption("Inspect crime severity scoring, threat indicators, category classification, and diagnostic observations.")

    analysis = resolve_active_analysis()
    if analysis is None:
        st.warning("No active evidence analysis available. Upload and detect evidence in the 'Upload & Detect' tab first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Severity Score", f"{analysis.severity_score}/100")
    c2.metric("Severity Level", analysis.severity_level.upper())
    c3.metric("Category", analysis.suggested_category.replace("_", " ").title())
    c4.metric("Total Objects", sum(analysis.counts_by_label.values()))

    if analysis.has_threat:
        st.error("⚠️ THREAT DETECTED: Weapon or high-threat item detected in active evidence.")
    else:
        st.success("✅ No high-threat weapon detected in active evidence.")

    st.markdown("---")
    col1, col2 = col2_cols = st.columns(2)
    with col1:
        st.markdown("#### Detected Objects Breakdown")
        # Separate verified weapons from other detections for clarity
        verified_weapons = 0
        others = {}

        for lbl, count in analysis.counts_by_label.items():
            # If the label is actually a weapon, we can highlight it
            if "weapon" in lbl.lower():
                verified_weapons += count
            else:
                others[lbl] = count

        if verified_weapons > 0:
            st.markdown(f"- 🚨 **Verified Weapons**: {verified_weapons}")

        for lbl, count in others.items():
            st.markdown(f"- **{lbl.replace('_', ' ').title()}**: {count}")

    with col2:
        st.markdown("#### Key Observations")
        for obs in analysis.key_observations:
            st.markdown(f"- {obs}")

    if getattr(analysis, "diagnostics", None):
        with st.expander("View Diagnostic Technical Logs"):
            st.json(analysis.diagnostics)


def render() -> None:
    """Render the primary Investigation workspace page."""
    render_page_header(
        ICON_INVESTIGATION,
        subtitle="Primary Investigation Workspace — Upload evidence, run YOLOv8 object detection, and generate structured evidence analysis.",
    )

    tab1, tab2 = st.tabs(["🔍 Upload & Detect", "📊 Structured Analysis"])

    with tab1:
        _render_upload_and_detect_tab()

    with tab2:
        _render_evidence_analysis_tab()
