"""
Report generators.

Two pure generators:
- `PDFReportGenerator.generate(report_data) -> bytes`
- `DOCXReportGenerator.generate(report_data) -> bytes`

Both consume a `ReportData` dataclass and return the raw bytes of the
generated document. No I/O, no Streamlit, no LLM. Easy to test.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from io import BytesIO

from config import APP_NAME, APP_VERSION, REPORT_REMARKS, REPORT_TITLE
from models.schemas import EvidenceAnalysis, InvestigationSummary, ReportData, IntelligenceReportData


# ----------------------------------------------------------------------
# ReportData builder
# ----------------------------------------------------------------------
def build_report_data(
    analysis: EvidenceAnalysis,
    summary: InvestigationSummary | None = None,
    report_id: str | None = None,
    title: str = REPORT_TITLE,
    remarks: str = REPORT_REMARKS,
    human_stats: dict[str, int] | None = None,
) -> ReportData:
    """
    Build a `ReportData` from an `EvidenceAnalysis` and (optionally)
    an `InvestigationSummary`. If no summary is provided, the
    template text is used.

    For batch analyses (``source_type == "batch"``) the helper also
    derives a `batch_totals` payload from the analysis itself.
    """
    if summary is not None:
        summary_text = summary.primary_text
        model_name = f"yolov8n | {summary.model_name}"
    else:
        # Use a fresh template summary as a sensible default
        from models.summary_generator import build_template_summary
        template_summary = build_template_summary(analysis)
        summary_text = template_summary.investigation_summary
        model_name = "yolov8n | template-only"

    remarks = _batched_remarks(remarks, analysis)

    return ReportData(
        title=title,
        report_id=report_id or f"RPT-{uuid.uuid4().hex[:8].upper()}",
        generated_at=datetime.now(),
        source_name=analysis.source_name,
        source_type=analysis.source_type,
        model_name=model_name,
        summary_text=summary_text,
        severity_score=analysis.severity_score,
        severity_level=analysis.severity_level,
        suggested_category=analysis.suggested_category,
        has_threat=analysis.has_threat,
        counts_by_label=dict(analysis.counts_by_label),
        total_objects=analysis.total_objects,
        person_count=analysis.person_count,
        weapon_count=analysis.weapon_count,
        vehicle_count=analysis.vehicle_count,
        bag_count=analysis.bag_count,
        key_observations=list(analysis.key_observations),
        average_confidence=analysis.average_confidence,
        remarks=remarks,
        app_version=APP_VERSION,
        human_total_detections=human_stats.get("human_total_detections", 0) if human_stats else 0,
        human_confirmed_count=human_stats.get("human_confirmed_count", 0) if human_stats else 0,
        human_rejected_count=human_stats.get("human_rejected_count", 0) if human_stats else 0,
        human_pending_count=human_stats.get("human_pending_count", 0) if human_stats else 0,
    )


def _batched_remarks(default_remarks: str, analysis: EvidenceAnalysis) -> str:
    """For batch analyses, append a clear "what this report covers and
    does NOT cover" block so the document cannot be misread as
    identifying suspects or locations.
    """
    if analysis.source_type != "batch":
        return default_remarks
    extra = (
        "\n\nScope of this report: This document consolidates the "
        "detection results from the uploaded evidence file(s). It "
        "lists ONLY objects that the configured object-detection "
        "model(s) reported in the file(s). It does NOT identify any "
        "person, weapon owner, location, motive, or event. Any "
        "category label (e.g. 'assault', 'robbery') is a suggestion "
        "based on the reported detections and MUST be verified by a "
        "qualified investigator."
    )
    return default_remarks + extra



# ----------------------------------------------------------------------
# PDF generator
# ----------------------------------------------------------------------
class PDFReportGenerator:
    """
    Produces a PDF investigation report via ReportLab.

    Layout: A4 portrait, 2 cm margins, simple Platypus flow.
    """

    def __init__(self) -> None:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        self._colors = colors
        self._pagesize = A4
        self._cm = cm
        self._styles = getSampleStyleSheet()
        self._make_styles()
        self._Paragraph = Paragraph
        self._SimpleDocTemplate = SimpleDocTemplate
        self._Spacer = Spacer
        self._Table = Table
        self._TableStyle = TableStyle

    def _make_styles(self) -> None:
        from reportlab.lib.styles import ParagraphStyle
        s = self._styles
        s.add(ParagraphStyle(
            name="ReportTitle",
            parent=s["Title"],
            fontSize=18, leading=22, spaceAfter=10,
        ))
        s.add(ParagraphStyle(
            name="ReportH2",
            parent=s["Heading2"],
            fontSize=13, leading=16, spaceBefore=10, spaceAfter=6,
        ))
        s.add(ParagraphStyle(
            name="ReportBody",
            parent=s["BodyText"],
            fontSize=10.5, leading=14, spaceAfter=6,
        ))
        s.add(ParagraphStyle(
            name="ReportBullet",
            parent=s["BodyText"],
            fontSize=10.5, leading=14, leftIndent=14, bulletIndent=2,
        ))
        s.add(ParagraphStyle(
            name="ReportFooter",
            parent=s["BodyText"],
            fontSize=8, leading=10, textColor=self._colors.grey,
        ))

    def generate(self, data: ReportData) -> bytes:
        Paragraph = self._Paragraph
        SimpleDocTemplate = self._SimpleDocTemplate
        Spacer = self._Spacer
        Table = self._Table
        TableStyle = self._TableStyle
        cm = self._cm
        colors = self._colors

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=self._pagesize,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=data.title,
            author=APP_NAME,
        )

        story: list = []

        # Title block
        story.append(Paragraph(data.title, self._styles["ReportTitle"]))
        story.append(Paragraph(
            f"<b>Report ID:</b> {data.report_id} &nbsp;&nbsp; "
            f"<b>Generated:</b> {data.generated_at.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;&nbsp; "
            f"<b>App version:</b> {data.app_version}",
            self._styles["ReportBody"],
        ))
        story.append(Spacer(1, 0.3 * cm))

        # 1. Source info
        story.append(Paragraph("1. Source Information", self._styles["ReportH2"]))
        source_table = Table(
            [
                ["Source file",   data.source_name],
                ["Source type",   data.source_type],
                ["Model used",    data.model_name],
            ],
            colWidths=[4 * cm, 12 * cm],
        )
        source_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(source_table)

        # 2. Executive summary
        story.append(Paragraph("2. Executive Summary", self._styles["ReportH2"]))
        safe_summary = (
            data.summary_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        story.append(Paragraph(safe_summary, self._styles["ReportBody"]))

        # 3. Severity & category
        story.append(Paragraph("3. Severity &amp; Category", self._styles["ReportH2"]))
        sev_table = Table(
            [
                ["Severity score (0-100)", str(data.severity_score)],
                ["Severity level",         data.severity_level.upper()],
                ["Suggested category",     data.suggested_category.replace("_", " ").title()],
                ["Threat detected",        "YES" if data.has_threat else "NO"],
                ["Average confidence",     f"{data.average_confidence:.2%}"],
            ],
            colWidths=[5 * cm, 11 * cm],
        )
        sev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ]))
        story.append(sev_table)

        # 4. Evidence table
        story.append(Paragraph("4. Detected Evidence", self._styles["ReportH2"]))
        if data.counts_by_label:
            rows = [["Label", "Count"]]
            for label, count in sorted(data.counts_by_label.items()):
                rows.append([label.title(), str(count)])
            ev_table = Table(rows, colWidths=[8 * cm, 3 * cm])
            ev_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 10),
                ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("ALIGN",      (1, 0), (1, -1), "CENTER"),
            ]))
            story.append(ev_table)
        else:
            story.append(Paragraph("<i>No objects detected.</i>", self._styles["ReportBody"]))

        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            f"<b>Total objects:</b> {data.total_objects} &nbsp;&nbsp; "
            f"<b>Persons:</b> {data.person_count} &nbsp;&nbsp; "
            f"<b>Weapons:</b> {data.weapon_count} &nbsp;&nbsp; "
            f"<b>Vehicles:</b> {data.vehicle_count} &nbsp;&nbsp; "
            f"<b>Bags:</b> {data.bag_count}",
            self._styles["ReportBody"],
        ))

        # 5. Observations
        story.append(Paragraph("5. Key Observations", self._styles["ReportH2"]))
        if data.key_observations:
            for line in data.key_observations:
                story.append(Paragraph(f"&bull; {line}", self._styles["ReportBullet"]))
        else:
            story.append(Paragraph("<i>No observations.</i>", self._styles["ReportBody"]))

        # 6. Remarks
        story.append(Paragraph("6. Remarks", self._styles["ReportH2"]))
        story.append(Paragraph(data.remarks, self._styles["ReportBody"]))

        # 7. Human Verification Summary
        story.append(Paragraph("7. Human Verification Summary", self._styles["ReportH2"]))
        human_table = Table(
            [
                ["Total reviewed detections", str(data.human_total_detections)],
                ["Confirmed (Verified)",      str(data.human_confirmed_count)],
                ["Rejected (False Positives)", str(data.human_rejected_count)],
                ["Pending / Uncertain",       str(data.human_pending_count)],
            ],
            colWidths=[7 * cm, 11 * cm],
        )
        human_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ]))
        story.append(human_table)

        # Footer
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            f"Generated by {APP_NAME} v{data.app_version} &nbsp;|&nbsp; "
            f"Report ID: {data.report_id}",
            self._styles["ReportFooter"],
        ))

        doc.build(story)
        return buf.getvalue()


# ----------------------------------------------------------------------
# DOCX generator
# ----------------------------------------------------------------------
class DOCXReportGenerator:
    """
    Produces a .docx investigation report via python-docx.

    Layout: title, metadata table, sections (summary, severity,
    evidence, observations, remarks), simple footer.
    """

    def __init__(self) -> None:
        from docx import Document
        self._Document = Document

    def generate(self, data: ReportData) -> bytes:
        doc = self._Document()

        # Title
        title = doc.add_heading(data.title, level=0)
        title.alignment = 1  # centered

        # Metadata table
        meta = doc.add_table(rows=4, cols=2)
        meta.style = "Light Grid Accent 1"
        meta_rows = [
            ("Report ID",      data.report_id),
            ("Generated at",   data.generated_at.strftime("%Y-%m-%d %H:%M:%S")),
            ("App version",    data.app_version),
            ("Source file",    f"{data.source_name}  ({data.source_type})"),
        ]
        for i, (k, v) in enumerate(meta_rows):
            meta.cell(i, 0).text = k
            meta.cell(i, 1).text = v
            for run in meta.cell(i, 0).paragraphs[0].runs:
                run.bold = True

        # 1. Source
        doc.add_heading("1. Source Information", level=1)
        doc.add_paragraph(f"Model used: {data.model_name}")

        # 2. Executive summary
        doc.add_heading("2. Executive Summary", level=1)
        doc.add_paragraph(data.summary_text)

        # 3. Severity & category
        doc.add_heading("3. Severity & Category", level=1)
        sev = doc.add_table(rows=5, cols=2)
        sev.style = "Light List Accent 1"
        sev_rows = [
            ("Severity score (0-100)", str(data.severity_score)),
            ("Severity level",         data.severity_level.upper()),
            ("Suggested category",     data.suggested_category.replace("_", " ").title()),
            ("Threat detected",        "YES" if data.has_threat else "NO"),
            ("Average confidence",     f"{data.average_confidence:.2%}"),
        ]
        for i, (k, v) in enumerate(sev_rows):
            sev.cell(i, 0).text = k
            sev.cell(i, 1).text = v
            for run in sev.cell(i, 0).paragraphs[0].runs:
                run.bold = True

        # 4. Evidence
        doc.add_heading("4. Detected Evidence", level=1)
        if data.counts_by_label:
            ev = doc.add_table(rows=1 + len(data.counts_by_label), cols=2)
            ev.style = "Light Grid Accent 1"
            ev.cell(0, 0).text = "Label"
            ev.cell(0, 1).text = "Count"
            for run in (
                ev.rows[0].cells[0].paragraphs[0].runs
                + ev.rows[0].cells[1].paragraphs[0].runs
            ):
                run.bold = True
            for i, (label, count) in enumerate(sorted(data.counts_by_label.items()), start=1):
                ev.cell(i, 0).text = label.title()
                ev.cell(i, 1).text = str(count)
        else:
            doc.add_paragraph("No objects detected.")

        p = doc.add_paragraph()
        p.add_run(
            f"Total objects: {data.total_objects}    "
            f"Persons: {data.person_count}    "
            f"Weapons: {data.weapon_count}    "
            f"Vehicles: {data.vehicle_count}    "
            f"Bags: {data.bag_count}"
        ).bold = True

        # 5. Observations
        doc.add_heading("5. Key Observations", level=1)
        if data.key_observations:
            for line in data.key_observations:
                doc.add_paragraph(line, style="List Bullet")
        else:
            doc.add_paragraph("No observations.")

        # 6. Remarks
        doc.add_heading("6. Remarks", level=1)
        doc.add_paragraph(data.remarks)

        # 7. Human Verification Summary
        doc.add_heading("7. Human Verification Summary", level=1)
        human = doc.add_table(rows=4, cols=2)
        human.style = "Light List Accent 1"
        human_rows = [
            ("Total reviewed detections", str(data.human_total_detections)),
            ("Confirmed (Verified)",      str(data.human_confirmed_count)),
            ("Rejected (False Positives)", str(data.human_rejected_count)),
            ("Pending / Uncertain",       str(data.human_pending_count)),
        ]
        for i, (k, v) in enumerate(human_rows):
            human.cell(i, 0).text = k
            human.cell(i, 1).text = v
            for run in human.cell(i, 0).paragraphs[0].runs:
                run.bold = True

        # Footer
        footer = doc.sections[0].footer
        footer.paragraphs[0].text = (
            f"Generated by {APP_NAME} v{data.app_version} | Report ID: {data.report_id}"
        )

        buf = BytesIO()
        doc.save(buf)
        return buf.getvalue()

# ----------------------------------------------------------------------
# Intelligence Report Generator (Cross-Case)
# ----------------------------------------------------------------------
class IntelligenceReportGenerator:
    """
    Produces a PDF intelligence report for cross-case analysis.
    """

    def __init__(self) -> None:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        self._colors = colors
        self._pagesize = A4
        self._cm = cm
        self._styles = getSampleStyleSheet()
        self._Paragraph = Paragraph
        self._SimpleDocTemplate = SimpleDocTemplate
        self._Spacer = Spacer
        self._Table = Table
        self._TableStyle = TableStyle

    def generate(self, data: IntelligenceReportData) -> bytes:
        Paragraph = self._Paragraph
        SimpleDocTemplate = self._SimpleDocTemplate
        Spacer = self._Spacer
        Table = self._Table
        TableStyle = self._TableStyle
        cm = self._cm
        colors = self._colors

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=self._pagesize,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title="Intelligence Analysis",
            author=APP_NAME,
        )

        story: list = []

        # Title block
        story.append(Paragraph("Cross-Case Intelligence Analysis Report", self._styles["Title"]))
        story.append(Paragraph(
            f"<b>Report ID:</b> {data.report_id} &nbsp;&nbsp; "
            f"<b>Generated:</b> {data.generated_at.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;&nbsp; "
            f"<b>App version:</b> {data.app_version}",
            self._styles["BodyText"],
        ))
        story.append(Spacer(1, 0.5 * cm))

        # 1. Global Distribution
        story.append(Paragraph("1. Global Evidence Distribution", self._styles["Heading2"]))
        if data.global_distribution:
            rows = [["Label", "Total Occurrences"]]
            for label, count in sorted(data.global_distribution.items(), key=lambda x: x[1], reverse=True):
                rows.append([label.title(), str(count)])

            dist_table = Table(rows, colWidths=[8 * cm, 4 * cm])
            dist_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 10),
                ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("ALIGN",      (1, 0), (1, -1), "CENTER"),
            ]))
            story.append(dist_table)
        else:
            story.append(Paragraph("<i>No global evidence detected.</i>", self._styles["BodyText"]))

        story.append(Spacer(1, 0.8 * cm))

        # 2. Crime Series Clusters
        story.append(Paragraph("2. Automated Crime Series Identification", self._styles["Heading2"]))
        story.append(Paragraph(
            "The following groups of cases have been automatically clustered based on shared forensic patterns. "
            "Cases within the same series are highly likely to be related.",
            self._styles["BodyText"]
        ))
        story.append(Spacer(1, 0.3 * cm))

        if data.crime_series:
            for series_id, case_ids in data.crime_series.items():
                story.append(Paragraph(f"<b>Crime Series #{series_id + 1}</b>", self._styles["Normal"]))

                case_list = []
                for cid in case_ids:
                    name = data.case_details.get(cid, "Unknown Source")
                    case_list.append(f"Case #{cid}: {name}")

                # Join as bullet points
                for item in case_list:
                    story.append(Paragraph(f"&bull; {item}", self._styles["BodyText"]))

                story.append(Spacer(1, 0.2 * cm))
        else:
            story.append(Paragraph("<i>No distinct crime series identified.</i>", self._styles["BodyText"]))

        # Footer
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph(
            f"Generated by {APP_NAME} v{data.app_version} &nbsp;|&nbsp; "
            f"Report ID: {data.report_id}",
            self._styles["BodyText"],
        ))

        doc.build(story)
        return buf.getvalue()
