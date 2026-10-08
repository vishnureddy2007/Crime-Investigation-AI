"""
Report Generation page.

Reads the last `EvidenceAnalysis` (and optionally an
`InvestigationSummary`) from session state, builds a `ReportData`,
and offers PDF + DOCX download buttons. Also persists a copy of
each generated report under `outputs/reports/`.
"""

from __future__ import annotations

import streamlit as st

from config import REPORTS_DIR, DATABASE_PATH, SEVERITY_LEVEL_COLORS
from core.icons import ACTION_DOWNLOAD_DOCX, ACTION_DOWNLOAD_PDF, ICON_REPORT
from database.repository import get_human_review_stats
from models.report_generator import (
    DOCXReportGenerator,
    PDFReportGenerator,
    build_report_data,
)
from models.schemas import (
    EvidenceAnalysis,
    InvestigationSummary,
    ReportData,
)
from pages._layout import empty_state, friendly_error, render_page_header
from utils.db_hooks import auto_save_last_report


def _get_severity_badge(level: str | None) -> str:
    """Return a color-coded HTML badge for the severity level."""
    if not level:
        return '<span style="color: grey;">N/A</span>'
    lvl = level.lower()
    color = SEVERITY_LEVEL_COLORS.get(lvl, "#cbd5e1")
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 700;">{lvl.upper()}</span>'


def render() -> None:
    render_page_header(
        ICON_REPORT,
        subtitle=(
            "Generate a PDF or DOCX investigation report from the latest "
            "evidence analysis. For multi-file uploads, the Crime Scene "
            "Investigation page produces a combined report automatically."
        ),
    )

    analysis: EvidenceAnalysis | None = st.session_state.get("last_analysis")

    # If the latest batch result is fresher, prefer its combined analysis.
    batch = st.session_state.get("last_batch_result")
    if batch is not None and getattr(batch, "combined_analysis", None) is not None:
        analysis = batch.combined_analysis

    if analysis is None:
        empty_state(
            "No evidence analysis found.",
            "Process crime scene images or CCTV videos in the Investigation workspace to generate the report.",
            action_label="Open Investigation Workspace",
            action_target="pages/investigation.py",
        )
        return

    # Check if the currently held report is outdated in the DB
    db_id = st.session_state.get("last_report_db_id")
    if db_id is not None:
        from database.repository import load_report
        report_data = load_report(DATABASE_PATH, db_id)
        if report_data and report_data.get("is_outdated"):
            st.warning("⚠️ This report is outdated. Evidence or human reviews have changed since it was generated. Please regenerate.")

    summary: InvestigationSummary | None = st.session_state.get("last_summary")
    if summary is None:
        st.info(
            "No AI summary in session. The report will use the deterministic "
            "template summary. Run **AI Summary** to enhance the report."
        )

    st.info(f"Source: **{analysis.source_name}** ({analysis.source_type})")

    # Get human review stats for the current case
    human_stats = {}
    if analysis is not None:
        from database.repository import latest_case_for_source
        cid = latest_case_for_source(DATABASE_PATH, analysis.source_name)
        if cid is not None:
            human_stats = get_human_review_stats(DATABASE_PATH, cid)

    # Re-build ReportData only when the inputs actually change. The
    # summary_text is baked into ReportData, so comparing it (plus
    # source_name) is enough to detect a re-render with identical
    # inputs and skip the auto-save hook, which would otherwise run
    # on every page reload.
    new_summary_text = getattr(summary, "primary_text", getattr(summary, "investigation_summary", "")) if summary is not None else ""
    needs_rebuild = st.session_state.get("last_report_data") is None
    if not needs_rebuild:
        prev: ReportData = st.session_state["last_report_data"]
        if (
            prev.source_name != analysis.source_name
            or prev.summary_text != new_summary_text
            or prev.human_total_detections != human_stats.get("human_total_detections", 0)
        ):
            needs_rebuild = True

    if needs_rebuild:
        report: ReportData = build_report_data(
            analysis=analysis,
            summary=summary,
            human_stats=human_stats
        )
        st.session_state["last_report_data"] = report
        auto_save_last_report()
    else:
        report = st.session_state["last_report_data"]

    with st.expander("Report preview"):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Severity**<br>{_get_severity_badge(report.severity_level)} ({report.severity_score})", unsafe_allow_html=True)
        c2.metric("Category", report.suggested_category.replace("_", " ").title())
        c3.metric("Total Objects", report.total_objects)
        c4.metric("Threat", "YES" if report.has_threat else "NO")
        st.json({
            "report_id":    report.report_id,
            "generated_at": report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name":   report.model_name,
            "source_name":  report.source_name,
            "source_type":  report.source_type,
        })

    st.subheader("Download")
    safe_stem = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in analysis.source_name.rsplit(".", 1)[0]
    )

    pdf_error: str | None = None
    try:
        pdf_bytes = PDFReportGenerator().generate(report)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = REPORTS_DIR / f"{report.report_id}_{safe_stem}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        st.download_button(
            label=ACTION_DOWNLOAD_PDF,
            data=pdf_bytes,
            file_name=pdf_path.name,
            mime="application/pdf",
        )
    except ImportError as exc:
        pdf_error = f"PDF generation requires reportlab: {exc}"
        friendly_error(ImportError(pdf_error), fallback_title="PDF report unavailable.")
        st.code("pip install reportlab")
    except Exception as exc:
        # Broad catch - reportlab can fail mid-render on font/locale/etc.
        # issues, and we don't want the whole page to crash; the DOCX
        # download below should still work.
        pdf_error = f"PDF generation failed: {exc}"
        friendly_error(exc, fallback_title="PDF report generation failed.")

    docx_error: str | None = None
    try:
        docx_bytes = DOCXReportGenerator().generate(report)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        docx_path = REPORTS_DIR / f"{report.report_id}_{safe_stem}.docx"
        docx_path.write_bytes(docx_bytes)
        st.download_button(
            label=ACTION_DOWNLOAD_DOCX,
            data=docx_bytes,
            file_name=docx_path.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except ImportError as exc:
        docx_error = f"DOCX generation requires python-docx: {exc}"
        friendly_error(ImportError(docx_error), fallback_title="DOCX report unavailable.")
        st.code("pip install python-docx")
    except Exception as exc:
        # Same rationale as the PDF block above.
        docx_error = f"DOCX generation failed: {exc}"
        friendly_error(exc, fallback_title="DOCX report generation failed.")

    if pdf_error and docx_error:
        st.warning(
            "Both report generators are unavailable. Check the errors above "
            "and verify the optional dependencies are installed."
        )

    st.caption(f"Reports saved to: {REPORTS_DIR}")
