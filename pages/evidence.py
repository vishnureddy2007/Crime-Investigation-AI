"""
Evidence Management page.

Displays a gallery of uploaded crime scene evidence images, detection counts,
verified weapon counts, confidence metrics, and analysis status.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from config import DATABASE_PATH
from database.repository import (
    get_latest_analysis,
    list_cases,
    mark_case_outdated,
)
from pages._layout import empty_state, render_page_header


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
    selected_label = st.selectbox("Select Case", list(case_map.keys()))
    c_data = case_map[selected_label]
    case_id = c_data["case_id"]

    analysis = get_latest_analysis(DATABASE_PATH, case_id)
    payload = analysis.get("payload_json") if analysis else {}
    if isinstance(payload, str):
        import json
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}

    st.markdown("---")

    # Metrics Summary
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Source File", c_data["source_name"])
    with m2:
        st.metric("Persons Detected", payload.get("person_count", 0))
    with m3:
        st.metric("Verified Weapons", payload.get("verified_weapon_count", 0))
    with m4:
        st.metric("Severity Score", f"{analysis.get('severity_score', 0)}/100" if analysis else "N/A")

    st.subheader("Evidence Items & Annotated Output")

    # Display annotated image if present
    annotated_path = payload.get("annotated_image_path")
    if annotated_path and Path(annotated_path).exists():
        st.image(annotated_path, caption=f"Annotated Evidence - Case #{case_id:03d}", use_container_width=True)
    else:
        st.info("No annotated evidence image path stored for this case.")

    # Detections Breakdown Table
    detections = payload.get("detections", [])
    if detections:
        st.subheader("Detections & Verification Breakdown")
        table_data = []
        for idx, d in enumerate(detections, start=1):
            table_data.append({
                "Item #": idx,
                "Label": d.get("label", "Unknown").upper(),
                "Confidence": f"{d.get('confidence', 0.0):.1%}",
                "Verified Weapon": "YES" if d.get("is_verified_weapon") else "NO",
                "Bounding Box": str(d.get("box") or d.get("bbox") or "N/A"),
            })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.caption("No individual object bounding boxes recorded.")
