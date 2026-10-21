"""
Evidence Management page.

Displays a gallery of uploaded crime scene evidence images, detection counts,
verified weapon counts, confidence metrics, and analysis status.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from config import DATABASE_PATH
from core.logging import get_logger
from database.repository import (
    get_latest_analysis,
    list_case_evidence_items,
    list_cases,
    mark_case_outdated,
)
from pages._layout import empty_state, render_page_header

log = get_logger(__name__)


def render() -> None:
    render_page_header(
        title="🔍 Evidence Gallery",
        subtitle="Catalog and inspect uploaded crime scene evidence, detected persons, and verified weapons.",
    )

    cases = list_cases(DATABASE_PATH)
    if not cases:
        empty_state(
            "No evidence uploaded yet.",
            "Start a new investigation to upload and analyze crime scene evidence.",
            action_label="New Investigation",
            action_target="New Investigation",
        )
        return

    # Filter by Case
    case_map = {f"Case #{c['case_id']:03d} ({c['source_name']})": c for c in cases}
    case_keys = list(case_map.keys())
    current_cid = st.session_state.get("current_case_id") or st.session_state.get("last_analysis_case_id")
    default_idx = 0
    if current_cid is not None:
        for idx, c in enumerate(cases):
            if c["case_id"] == current_cid:
                default_idx = idx
                break
    selected_label = st.selectbox("Select Case", case_keys, index=default_idx)
    c_data = case_map[selected_label]
    case_id = c_data["case_id"]

    analysis = get_latest_analysis(DATABASE_PATH, case_id)
    payload = analysis.get("payload_json") if analysis else {}
    if isinstance(payload, str):
        import json
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            log.warning("Could not parse payload for case %s: %s", case_id, exc)
            payload = {}

    st.markdown("---")

    # Metrics Summary
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Source File", c_data["source_name"])
    with m2:
        st.metric("Persons Detected", payload.get("person_count", 0))
    with m3:
        st.metric("Verified Weapons", payload.get("verified_weapon_count", 0))
    with m4:
        st.metric("Candidate Weapons", payload.get("candidate_weapon_count", 0), help="AI candidate suggestions")
    with m5:
        st.metric("Severity Score", f"{analysis.get('severity_score', 0)}/100" if analysis else "N/A")

    if payload.get("verified_weapon_count", 0) == 0:
        st.info("ℹ️ **Forensic Note:** No weapons detected in analyzed frames. This indicates non-detection under configured thresholds, not affirmative proof that an item or activity is absent.")

    # Model and Evidence Traceability Expander
    with st.expander("🔍 Model & Traceability Provenance", expanded=False):
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(f"**Case ID:** `#{case_id:03d}`")
            st.markdown(f"**Evidence Version:** `v{c_data.get('evidence_version', 1)}`")
            st.markdown(f"**Model Name:** `{payload.get('model_name', 'yolov8n')}`")
            st.markdown(f"**Model Version:** `{payload.get('model_version', 'YOLOv8-audited')}`")
        with p2:
            st.markdown(f"**Model Weights / Hash:** `{payload.get('model_weights', 'MS COCO / Audited')}`")
            st.markdown(f"**Raw Detections Count:** `{payload.get('raw_detections_count', payload.get('total_objects', 0))}`")
            st.markdown(f"**Analysis Timestamp:** `{analysis.get('created_at', 'N/A') if analysis else 'N/A'}`")

    # Case Evidence Items Gallery
    evidence_items = list_case_evidence_items(DATABASE_PATH, case_id)
    if evidence_items:
        st.subheader("Case Evidence Inventory")
        item_rows = []
        for itm in evidence_items:
            item_rows.append({
                "Evidence ID": itm.get("evidence_id", ""),
                "Filename": itm.get("filename", ""),
                "Type": itm.get("file_type", "").upper(),
                "Duration": f"{itm.get('duration', 0.0):.1f}s" if itm.get("duration") else "Image",
                "Frames": itm.get("frame_count", 1),
                "File Path": itm.get("file_path", ""),
                "Annotated": "YES" if itm.get("annotated_path") else "NO",
            })
        st.dataframe(item_rows, use_container_width=True)

        for itm in evidence_items:
            ann_p = itm.get("annotated_path")
            if ann_p and Path(ann_p).exists():
                st.image(ann_p, caption=f"{itm.get('evidence_id')} - {itm.get('filename')}", use_container_width=True)

    st.subheader("Primary Evidence Annotated Output")

    # Display annotated image if present
    annotated_path = payload.get("annotated_image_path")
    if annotated_path and Path(annotated_path).exists():
        st.image(annotated_path, caption=f"Annotated Evidence - Case #{case_id:03d}", use_container_width=True)
    elif not evidence_items:
        st.info("No annotated evidence image path stored for this case.")

    # Detections Breakdown Table
    detections = payload.get("detections", [])
    if detections:
        st.subheader("Detections & Verification Breakdown")
        table_data = []
        for idx, d in enumerate(detections, start=1):
            status = "VERIFIED WEAPON" if d.get("weapon_status") == "verified" or d.get("is_verified_weapon") else (
                "CANDIDATE WEAPON" if d.get("weapon_status") == "candidate" or d.get("label") == "candidate_weapon" else "OBJECT"
            )
            table_data.append({
                "Item #": idx,
                "Label": d.get("label", "Unknown").upper(),
                "Confidence": f"{d.get('confidence', 0.0):.1%}",
                "Status": status,
                "Bounding Box": str(d.get("box") or d.get("bbox") or "N/A"),
            })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.caption("No individual object bounding boxes recorded.")
