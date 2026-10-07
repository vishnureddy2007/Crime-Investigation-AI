"""
Case Repository & Analytics — Crime Investigation AI.

Browse, inspect, and reload previously persisted investigation runs, and inspect
aggregated database analytics.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import DATABASE_PATH, SEVERITY_LEVEL_COLORS
from core.icons import ICON_CASE_HISTORY
from database.db import init_db
from database.repository import (
    list_analyses_for_case,
    list_cases,
    list_reports_for_case,
    list_summaries_for_case,
    load_analysis,
    load_report,
    list_storyboards_for_case,
    load_storyboard,
    load_summary,
    list_animations_for_case,
    load_animation,
)
from pages._layout import empty_state, render_page_header
from services.analytics import compute_snapshot


def _get_severity_badge(level: str | None) -> str:
    """Return a color-coded HTML badge for the severity level."""
    if not level:
        return '<span style="color: grey;">N/A</span>'

    lvl = level.lower()
    color = SEVERITY_LEVEL_COLORS.get(lvl, "#cbd5e1")
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 700;">{lvl.upper()}</span>'


def _case_label(case: dict) -> str:
    """Format a case row for a selectbox."""
    return f"#{case['case_id']} — {case['source_name']}"


def _render_case_browser_tab() -> None:
    """Tab 1: Case Repository Browser."""
    st.markdown("### Saved Case Repository")
    init_db(DATABASE_PATH)
    cases = list_cases(DATABASE_PATH)

    if not cases:
        empty_state(
            "No saved cases in database.",
            "Process evidence in the Investigation workspace to persist your first investigation case.",
            action_label="Open Investigation Workspace",
            action_target="pages/investigation.py",
        )
        return

    # Filter / Search
    search_q = st.text_input("Search cases by filename or ID", key="case_search_q")
    filtered = cases
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = [
            c for c in cases
            if q in str(c["case_id"]) or q in c["source_name"].lower()
        ]

    st.markdown(f"**Showing {len(filtered)} case(s)**")

    # Professional Case Table
    if filtered:
        # We use a custom HTML table for the list to support the badges
        table_html = """
        <table style="width: 100%; border-collapse: collapse; font-family: sans-serif; margin-bottom: 1rem;">
            <thead>
                <tr style="text-align: left; border-bottom: 2px solid #e2e8f0; color: #64748b;">
                    <th style="padding: 10px;">ID</th>
                    <th style="padding: 10px;">Source File</th>
                    <th style="padding: 10px;">Severity</th>
                    <th style="padding: 10px;">Date</th>
                </tr>
            </thead>
            <tbody>
        """
        for c in filtered:
            badge = _get_severity_badge(c.get("latest_severity_level"))
            table_html += f"""
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px; font-weight: 700;">#{c['case_id']}</td>
                    <td style="padding: 10px;">{c['source_name']}</td>
                    <td style="padding: 10px;">{badge}</td>
                    <td style="padding: 10px; color: #94a3b8; font-size: 0.85rem;">{c.get('created_at', '')[:10]}</td>
                </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

    selected_case = st.selectbox(
        "Select a Case to Inspect",
        options=filtered,
        format_func=_case_label,
        key="sel_case_obj",
    )

    if selected_case is not None:
        case_id = selected_case["case_id"]

        with st.container(border=True):
            st.markdown(f"### Case #{case_id}: `{selected_case['source_name']}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Source Type", (selected_case.get("source_type") or "n/a").upper())
            c2.metric("Severity Level", (selected_case.get("latest_severity_level") or "n/a").upper())
            c3.metric("Created At", selected_case.get("created_at", "")[:19])

        # Sub-sections
        st.markdown("#### Case Artifacts")

        # --- Analyses ---
        rows = list_analyses_for_case(DATABASE_PATH, case_id)
        if rows:
            st.markdown(f"**Analyses:** {len(rows)} record(s)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            sel_aid = st.selectbox("Inspect Analysis Payload", options=[r["analysis_id"] for r in rows], key=f"aid_{case_id}")
            if sel_aid:
                payload = load_analysis(DATABASE_PATH, sel_aid)
                with st.expander("Analysis Payload JSON"):
                    st.json(payload)

        # --- Summaries (with Outdated Badge) ---
        summaries = list_summaries_for_case(DATABASE_PATH, case_id)
        if summaries:
            st.markdown(f"**Summaries:** {len(summaries)} record(s)")
            summary_rows = []
            for s in summaries:
                outdated_tag = "⚠️ OUTDATED" if s.get("is_outdated") else "✅ CURRENT"
                summary_rows.append({
                    "ID": s["summary_id"],
                    "Model": s["model_name"],
                    "Date": s["created_at"][:19],
                    "Status": outdated_tag
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # --- Reports (with Outdated Badge) ---
        reports = list_reports_for_case(DATABASE_PATH, case_id)
        if reports:
            st.markdown(f"**Reports:** {len(reports)} record(s)")
            report_rows = []
            for r in reports:
                outdated_tag = "⚠️ OUTDATED" if r.get("is_outdated") else "✅ CURRENT"
                report_rows.append({
                    "ID": r["report_id"],
                    "UID": r["report_uid"],
                    "Date": r["created_at"][:19],
                    "Status": outdated_tag
                })
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

        # --- Animations (with Outdated Badge) ---
        animations = list_animations_for_case(DATABASE_PATH, case_id)
        if animations:
            st.markdown(f"**Animations:** {len(animations)} record(s)")
            anim_rows = []
            for a in animations:
                # We need to load the full record to check is_outdated
                full_anim = load_animation(DATABASE_PATH, a["animation_id"])
                outdated_tag = "⚠️ OUTDATED" if full_anim and full_anim.get("is_outdated") else "✅ CURRENT"
                anim_rows.append({
                    "ID": a["animation_id"],
                    "Path": a["video_path"],
                    "Date": a["created_at"][:19],
                    "Status": outdated_tag
                })
            st.dataframe(pd.DataFrame(anim_rows), use_container_width=True)

            # Let the user play the latest one
            if st.button("Play Latest Animation", key=f"play_anim_{case_id}"):
                latest_anim = load_animation(DATABASE_PATH, animations[0]["animation_id"])
                if latest_anim:
                    st.video(latest_anim["video_path"])


def _render_case_analytics_tab() -> None:
    """Tab 2: Aggregated Case Analytics."""
    st.markdown("### Database Analytics & Statistics")
    st.caption("Aggregated analytics sourced directly from the SQLite database repository.")

    snapshot = compute_snapshot(DATABASE_PATH)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", snapshot.total_cases)
    c2.metric("Total Analyses", snapshot.total_analyses)
    c3.metric("Threat Cases", snapshot.threats)
    c4.metric("Avg Severity", f"{snapshot.avg_severity:.1f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Crime Category Distribution")
        if snapshot.category_counts:
            df_cat = pd.DataFrame([
                {"Category": k.replace("_", " ").title(), "Count": v}
                for k, v in snapshot.category_counts.items()
            ])
            st.bar_chart(df_cat, x="Category", y="Count")
        else:
            st.info("No category data available.")

    with col2:
        st.markdown("#### Severity Band Breakdown")
        if snapshot.severity_counts:
            df_sev = pd.DataFrame([
                {"Severity Level": k.upper(), "Count": v}
                for k, v in snapshot.severity_counts.items()
            ])
            st.bar_chart(df_sev, x="Severity Level", y="Count")
        else:
            st.info("No severity band data available.")


def render() -> None:
    """Render the Case Repository & Analytics page."""
    render_page_header(
        ICON_CASE_HISTORY,
        subtitle="Case Repository & Analytics — Inspect saved investigation cases and system-wide database analytics.",
    )

    tab1, tab2 = st.tabs(["📁 Case Repository", "📊 Database Analytics"])

    with tab1:
        _render_case_browser_tab()

    with tab2:
        _render_case_analytics_tab()