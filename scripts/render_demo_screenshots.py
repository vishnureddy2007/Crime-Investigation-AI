#!/usr/bin/env python3
"""
Generate labelled mockup PNGs for `docs/screenshots/`.

These are not real screenshots of the running app — they are Pillow-rendered
mockups that show the page title, subtitle, and a generic stub layout so
the GitHub-rendered USER_MANUAL.md has visual anchors next to every section.

For real screenshots, see `docs/USER_MANUAL.md` § "Capturing real
screenshots": launch the app with `streamlit run app.py` and use any OS
screenshot tool.

Usage:
    python scripts/render_demo_screenshots.py [--out docs/screenshots]

Dependencies:
    Pillow (already in requirements.txt for image I/O).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ----------------------------------------------------------------------
# Brand colours (must stay consistent with `core/theming.py` light theme)
# ----------------------------------------------------------------------
BG          = (248, 249, 251)   # #f8f9fb
SURFACE     = (255, 255, 255)   # #ffffff
BORDER      = (226, 230, 236)   # #e2e6ec
TEXT        = (17, 24, 39)      # #111827
TEXT_MUTED  = (107, 114, 128)   # #6b7280
PRIMARY     = (30, 58, 138)     # #1e3a8a (deep navy)
ACCENT      = (37, 99, 235)     # #2563eb (restrained blue accent)
PILL_OK     = (220, 252, 231)   # #dcfce7
PILL_OK_FG  = (22, 101, 52)     # #166534
PILL_WARN   = (254, 249, 195)   # #fef9c3
PILL_WARN_FG= (133, 77, 14)     # #854d0e
FAINT       = (243, 244, 246)   # #f3f4f6


# ----------------------------------------------------------------------
# Font resolution (graceful fallback to PIL default)
# ----------------------------------------------------------------------
def _resolve_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


# ----------------------------------------------------------------------
# Page spec
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PageMockup:
    """One mockup to render."""
    slug: str           # file basename, e.g. "dashboard"
    title: str          # page title (e.g. "Dashboard")
    subtitle: str       # short subtitle
    blocks: tuple[tuple[str, str], ...]  # (label, value) for the KPI strip
    rows: tuple[str, ...]                # body bullet rows for the bottom panel


PAGES: tuple[PageMockup, ...] = (
    PageMockup(
        slug="dashboard",
        title="Dashboard",
        subtitle="Overview of persisted cases and the active analysis.",
        blocks=(
            ("Total cases",       "12"),
            ("High + Critical",   "3"),
            ("Avg severity",      "47.6"),
            ("Latest case",       "2026-08-12"),
        ),
        rows=(
            "Active analysis — Severity score 71/100 (HIGH)",
            "Threat detected: a weapon is present in the active evidence.",
            "Suggested category: Armed Robbery.",
            "Detected objects: person x4, knife x1, car x1, bottle x2.",
        ),
    ),
    PageMockup(
        slug="evidence_analysis",
        title="Evidence Analysis",
        subtitle="Convert raw detections into a structured EvidenceAnalysis.",
        blocks=(
            ("Persons",       "4"),
            ("Weapons",       "1"),
            ("Vehicles",      "1"),
            ("Bags",          "2"),
        ),
        rows=(
            "Severity score: 71 / 100 (HIGH band)",
            "Key observations:",
            "- 4 persons detected at average confidence 0.81",
            "- 1 knife detected at confidence 0.92 (threat)",
            "- Image source: street_cam_frame_032.png",
        ),
    ),
    PageMockup(
        slug="ai_summary",
        title="AI Investigation Summary",
        subtitle="Generate a written investigator summary from the analysis.",
        blocks=(
            ("Source",        "street_cam_frame_032.png"),
            ("Model",         "FLAN-T5 base"),
            ("Used AI",       "yes"),
            ("Generation",    "3.2 s"),
        ),
        rows=(
            "The submitted evidence contains four persons in close proximity",
            "to a vehicle. One individual appears to be holding a bladed",
            "instrument — confidence 0.92 — consistent with an armed",
            "confrontation. The recommended investigative next step is",
            "to gather CCTV from adjacent cameras in the same time window.",
        ),
    ),
    PageMockup(
        slug="report",
        title="Investigation Report",
        subtitle="Generate a PDF or DOCX investigation report.",
        blocks=(
            ("Severity",      "71 (HIGH)"),
            ("Category",      "Armed Robbery"),
            ("Detections",    "13"),
            ("Page count",    "3"),
        ),
        rows=(
            "Includes: cover page, evidence summary, severity table,",
            "list of detections, AI-generated narrative, and a",
            "disclaimer (academic project, not legal evidence).",
            "Available formats: PDF, DOCX.",
        ),
    ),
    PageMockup(
        slug="reconstruction",
        title="Crime Scene Reconstruction",
        subtitle="Generate a 4-6 scene storyboard and a short reconstruction video.",
        blocks=(
            ("Scenes",        "5"),
            ("Duration",      "12.0 s"),
            ("Width x Height","1280 x 720"),
            ("Format",        "MP4"),
        ),
        rows=(
            "Scene 1 — Establish: empty street, no persons.",
            "Scene 2 — Approach: vehicle enters frame left.",
            "Scene 3 — Escalation: four persons converge.",
            "Scene 4 — Threat: one individual brandishes a knife.",
            "Scene 5 — Aftermath: persons disperse, vehicle departs.",
            "AI-generated probable reconstruction — not a literal depiction.",
        ),
    ),
    PageMockup(
        slug="analytics",
        title="Analytics",
        subtitle="Aggregated KPIs across every persisted investigation.",
        blocks=(
            ("Cases",         "12"),
            ("Analyses",      "34"),
            ("Reports",       "11"),
            ("Feedback",      "4"),
        ),
        rows=(
            "Category distribution: theft x4, assault x3, robbery x2,",
            "vehicle_incident x2, other x1.",
            "Severity bands: low x5, medium x4, high x2, critical x1.",
            "Export options: JSON snapshot, CSV (categories), Markdown.",
        ),
    ),
)


# ----------------------------------------------------------------------
# Rendering primitives
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 800

# Layout constants (in pixels)
PAD          = 48
SIDEBAR_W    = 220
CONTENT_W    = WIDTH - SIDEBAR_W - PAD * 3   # leave a right margin


def _draw_sidebar(draw: ImageDraw.ImageDraw) -> None:
    """Left sidebar with the CV wordmark and page-rail entries."""
    # Sidebar background
    draw.rectangle(
        [(0, 0), (SIDEBAR_W, HEIGHT)],
        fill=BG,
    )
    # Wordmark badge "CV"
    draw.rectangle(
        [(28, 28), (28 + 36, 28 + 36)],
        fill=PRIMARY,
    )
    f_badge = _resolve_font(18, bold=True)
    draw.text((28 + 18 - 7, 28 + 8), "CV", fill=(255, 255, 255), font=f_badge)
    # Brand name
    f_brand = _resolve_font(15, bold=True)
    draw.text((76, 30), "CrimeVision AI", fill=TEXT, font=f_brand)
    f_tag = _resolve_font(11)
    draw.text((76, 50), "AI-Based Crime Investigation", fill=TEXT_MUTED, font=f_tag)

    # Nav rail
    rail_top = 120
    rail_items = [
        ("Dashboard",              True),
        ("Crime Scene Investigation", False),
        ("Image Detection",        False),
        ("Video Processing",       False),
        ("Evidence Analysis",      False),
        ("AI Investigation Summary", False),
        ("Crime Timeline",         False),
        ("Crime Prediction",       False),
        ("Investigation Report",   False),
        ("Crime Scene Reconstruction", False),
        ("Case History",           False),
        ("Analytics",              False),
        ("Settings",               False),
        ("Help",                   False),
        ("About",                  False),
    ]
    f_nav = _resolve_font(13)
    f_nav_active = _resolve_font(13, bold=True)
    y = rail_top
    for label, active in rail_items:
        if active:
            draw.rectangle(
                [(12, y - 4), (SIDEBAR_W - 12, y + 22)],
                fill=(238, 242, 255),
            )
            draw.rectangle(
                [(12, y - 4), (16, y + 22)],
                fill=ACCENT,
            )
            draw.text((24, y), label, fill=PRIMARY, font=f_nav_active)
        else:
            draw.text((24, y), label, fill=TEXT_MUTED, font=f_nav)
        y += 32


def _draw_demo_badge(draw: ImageDraw.ImageDraw) -> None:
    """A small 'Demo screenshot' ribbon top-right of the content area."""
    label = "DEMO SCREENSHOT (mockup)"
    f = _resolve_font(10, bold=True)
    bbox = draw.textbbox((0, 0), label, font=f)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 10
    x0 = WIDTH - w - 20
    y0 = 28
    draw.rectangle(
        [(x0, y0), (x0 + w, y0 + h)],
        fill=PILL_WARN,
    )
    draw.text((x0 + 10, y0 + 4), label, fill=PILL_WARN_FG, font=f)


def _draw_kpi_strip(draw: ImageDraw.ImageDraw, blocks: tuple[tuple[str, str], ...]) -> int:
    """Render 4 KPI tiles in a row. Returns the y-coordinate just below the strip."""
    n = len(blocks)
    gap = 18
    total_gaps = gap * (n - 1)
    tile_w = (CONTENT_W - total_gaps) // n
    y0 = 130
    x0 = SIDEBAR_W + PAD * 2
    f_label = _resolve_font(11)
    f_value = _resolve_font(28, bold=True)
    for i, (label, value) in enumerate(blocks):
        x = x0 + i * (tile_w + gap)
        # Tile background
        draw.rectangle(
            [(x, y0), (x + tile_w, y0 + 92)],
            fill=SURFACE,
            outline=BORDER,
            width=1,
        )
        draw.text((x + 16, y0 + 14), label.upper(), fill=TEXT_MUTED, font=f_label)
        draw.text((x + 16, y0 + 36), value, fill=TEXT, font=f_value)
    return y0 + 92 + 32


def _draw_bullets(draw: ImageDraw.ImageDraw, rows: tuple[str, ...], y0: int) -> None:
    """Render the bottom content panel as bullet rows inside a card."""
    box_h = HEIGHT - y0 - 32
    draw.rectangle(
        [(SIDEBAR_W + PAD * 2, y0), (SIDEBAR_W + PAD * 2 + CONTENT_W, y0 + box_h)],
        fill=SURFACE,
        outline=BORDER,
        width=1,
    )
    f_body = _resolve_font(13)
    y = y0 + 18
    for row in rows:
        draw.text(
            (SIDEBAR_W + PAD * 2 + 18, y),
            row,
            fill=TEXT,
            font=f_body,
        )
        y += 24


# ----------------------------------------------------------------------
# Page renderer
# ----------------------------------------------------------------------
def render_page(spec: PageMockup, out_dir: Path) -> Path:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Sidebar
    _draw_sidebar(draw)

    # Demo badge
    _draw_demo_badge(draw)

    # Page title + subtitle
    title_x = SIDEBAR_W + PAD * 2
    f_title = _resolve_font(28, bold=True)
    f_sub = _resolve_font(13)
    draw.text((title_x, 36), spec.title, fill=TEXT, font=f_title)
    draw.text((title_x, 82), spec.subtitle, fill=TEXT_MUTED, font=f_sub)

    # KPI strip
    y_after_kpis = _draw_kpi_strip(draw, spec.blocks)

    # Body panel
    _draw_bullets(draw, spec.rows, y_after_kpis)

    out_path = out_dir / f"{spec.slug}.png"
    img.save(out_path, "PNG", optimize=True)
    return out_path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="docs/screenshots",
        type=Path,
        help="Output directory for the generated PNGs.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in PAGES:
        path = render_page(spec, out_dir)
        size_kb = path.stat().st_size / 1024
        print(f"  wrote {path}  ({size_kb:.1f} KB)")

    print(f"\n{len(PAGES)} mockup screenshot(s) written to {out_dir}/")
    print("Note: these are labelled mockups, not real running-app captures.")
    print("See docs/USER_MANUAL.md for the recipe to capture real screenshots.")


if __name__ == "__main__":
    main()
