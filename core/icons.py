"""Icon constants for the CrimeVision AI UI.

These are **text-only** markers (no emoji, no glyph fonts). Every
constant is the human-readable label that gets rendered as the
"icon" — the typographic look the project standardises on for a
professional, academic appearance.

Why text-only?

- Emoji render differently across Windows / macOS / Linux.
- Emoji look playful, not forensic.
- Text labels are accessible to screen readers by default.

Pages can compose the icon into a section header like::

    from core.icons import ICON_DETECTION
    st.subheader(f"{ICON_DETECTION} Detection results")

That keeps the chrome consistent without ever pulling a colourful
glyph into the UI.
"""
from __future__ import annotations

# ----------------------------------------------------------------------
# Brand
# ----------------------------------------------------------------------
BRAND_NAME: str = "CrimeVision AI"
BRAND_TAGLINE: str = "AI-Based Crime Investigation Assistant"

# ----------------------------------------------------------------------
# Page / section labels (replace emoji + freeform title strings)
# ----------------------------------------------------------------------
ICON_HOME: str = "Home"
ICON_DASHBOARD: str = "Dashboard"
ICON_INVESTIGATION: str = "Crime Scene Investigation"
ICON_IMAGE_DETECTION: str = "Image Detection"
ICON_VIDEO_PROCESSING: str = "Video Processing"
ICON_EVIDENCE_ANALYSIS: str = "Evidence Analysis"
ICON_AI_SUMMARY: str = "AI Investigation Summary"
ICON_CHAT: str = "AI Chat Assistant"
ICON_TIMELINE: str = "Crime Timeline"
ICON_PREDICTION: str = "Crime Prediction"
ICON_REPORT: str = "Investigation Report"
ICON_RECONSTRUCTION: str = "Crime Scene Reconstruction"
ICON_CASE_HISTORY: str = "Case History"
ICON_ANALYTICS: str = "Analytics"
ICON_SETTINGS: str = "Settings"
ICON_HELP: str = "Help"
ICON_CONTACT: str = "Contact"
ICON_ABOUT: str = "About"
ICON_SYSTEM_STATUS: str = "System Status"

# ----------------------------------------------------------------------
# Section-level markers (no emoji, just typographic glyphs)
# ----------------------------------------------------------------------
SECTION_KPIS: str = "Key Metrics"
SECTION_PIPELINE: str = "Pipeline Outputs"
SECTION_DETECTIONS: str = "Detection Results"
SUMMARY_EVIDENCE: str = "Evidence Summary"
SECTION_OBSERVATIONS: str = "Key Observations"
SECTION_DIAGNOSTICS: str = "Diagnostics"
SECTION_PER_FILE: str = "Per-File Results"
SECTION_COMBINED: str = "Combined Summary"
SECTION_DOWNLOADS: str = "Downloads"
SECTION_SEVERITY: str = "Severity Assessment"

# Status labels (replaces "✅" / "⚠" / "❌" with plain text)
STATUS_OK: str = "Completed"
STATUS_PENDING: str = "Pending"
STATUS_FAILED: str = "Failed"
STATUS_PROCESSING: str = "Processing"
STATUS_NOT_GENERATED: str = "Not generated yet"

# Generic action labels
ACTION_ANALYZE: str = "Analyze Evidence"
ACTION_DOWNLOAD_PDF: str = "Download PDF Report"
ACTION_DOWNLOAD_DOCX: str = "Download DOCX Report"
ACTION_RETRY: str = "Try Again"
ACTION_NEW_CASE: str = "New Case"
ACTION_VIEW_REPORTS: str = "View Reports"
ACTION_VIEW_RECONSTRUCTION: str = "View Reconstruction"

__all__ = [
    "BRAND_NAME",
    "BRAND_TAGLINE",
    "ICON_HOME",
    "ICON_DASHBOARD",
    "ICON_INVESTIGATION",
    "ICON_IMAGE_DETECTION",
    "ICON_VIDEO_PROCESSING",
    "ICON_EVIDENCE_ANALYSIS",
    "ICON_AI_SUMMARY",
    "ICON_CHAT",
    "ICON_TIMELINE",
    "ICON_PREDICTION",
    "ICON_REPORT",
    "ICON_RECONSTRUCTION",
    "ICON_CASE_HISTORY",
    "ICON_ANALYTICS",
    "ICON_SETTINGS",
    "ICON_HELP",
    "ICON_CONTACT",
    "ICON_ABOUT",
    "ICON_SYSTEM_STATUS",
    "SECTION_KPIS",
    "SECTION_PIPELINE",
    "SECTION_DETECTIONS",
    "SUMMARY_EVIDENCE",
    "SECTION_OBSERVATIONS",
    "SECTION_DIAGNOSTICS",
    "SECTION_PER_FILE",
    "SECTION_COMBINED",
    "SECTION_DOWNLOADS",
    "SECTION_SEVERITY",
    "STATUS_OK",
    "STATUS_PENDING",
    "STATUS_FAILED",
    "STATUS_PROCESSING",
    "STATUS_NOT_GENERATED",
    "ACTION_ANALYZE",
    "ACTION_DOWNLOAD_PDF",
    "ACTION_DOWNLOAD_DOCX",
    "ACTION_RETRY",
    "ACTION_NEW_CASE",
    "ACTION_VIEW_REPORTS",
    "ACTION_VIEW_RECONSTRUCTION",
]
