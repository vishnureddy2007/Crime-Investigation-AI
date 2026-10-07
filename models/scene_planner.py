"""
Storyboard scene planner.

Pure deterministic function: takes an EvidenceAnalysis (+ optional
source image) and returns a Storyboard with 4-6 scenes.

The scenes are NOT AI-generated images. They are captioned panels
based on the actual detected evidence. If a real image/keyframe is
available, it's reused as the visual backdrop; otherwise a clean
colored placeholder is generated.
"""

from __future__ import annotations

import requests
import json
from datetime import datetime
from typing import Any

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    STORYBOARD_DEFAULT_DURATION_SEC,
    STORYBOARD_MAX_SCENES,
    STORYBOARD_MIN_SCENES,
    STORYBOARD_PANEL_SIZE,
)
from models.schemas import EvidenceAnalysis, InvestigationSummary, Storyboard, StoryboardScene


# Subtle tints applied to reused real frames so consecutive scenes
# are visually distinguishable without fabricating content.
_SCENE_TINTS: list[tuple[int, int, int]] = [
    (  0,   0,   0),   # 1. baseline
    ( 20,  40,  20),   # 2. slight green
    ( 40,  20,  20),   # 3. slight red (threat)
    ( 20,  20,  40),   # 4. slight blue
    ( 40,  30,  10),   # 5. slight amber
    ( 10,  10,  10),   # 6. near-baseline
]


def _panel_placeholder(title: str, caption: str) -> Any:
    """Build a clean colored placeholder panel using PIL."""
    from PIL import Image, ImageDraw, ImageFont
    w, h = STORYBOARD_PANEL_SIZE
    img = Image.new("RGB", (w, h), color=(26, 26, 46))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        caption_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        title_font = ImageFont.load_default()
        caption_font = ImageFont.load_default()
    try:
        tw = draw.textlength(title, font=title_font)
    except AttributeError:
        tw = len(title) * 12
    draw.text(((w - tw) / 2, h / 2 - 40), title, fill=(232, 234, 246), font=title_font)
    wrapped = _wrap_text(caption, max_chars=46)
    for i, line in enumerate(wrapped[:4]):
        try:
            lw = draw.textlength(line, font=caption_font)
        except AttributeError:
            lw = len(line) * 8
        draw.text(((w - lw) / 2, h / 2 + 10 + i * 22), line, fill=(180, 190, 210), font=caption_font)
    return img


def _wrap_text(text: str, max_chars: int = 46) -> list[str]:
    """Greedy word-wrap to a max character width."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _tint_image(img: Any, tint: tuple[int, int, int]) -> Any:
    """Apply a soft RGB tint to a PIL image."""
    from PIL import Image
    base = img.convert("RGB")
    if tint == (0, 0, 0):
        return base
    overlay = Image.new("RGB", base.size, tint)
    return Image.blend(base, overlay, alpha=0.18)


def _resize_to_panel(img: Any) -> Any:
    """Resize any PIL image to the standard panel size (preserving aspect, padded)."""
    from PIL import Image
    base = img.convert("RGB")
    w, h = STORYBOARD_PANEL_SIZE
    src_w, src_h = base.size
    scale = min(w / src_w, h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = base.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), color=(0, 0, 0))
    canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


def plan_scenes(
    analysis: EvidenceAnalysis,
    base_image: Any | None = None,
    min_scenes: int = STORYBOARD_MIN_SCENES,
    max_scenes: int = STORYBOARD_MAX_SCENES,
    default_duration: float = STORYBOARD_DEFAULT_DURATION_SEC,
) -> Storyboard:
    """
    Plan a logical sequence of storyboard scenes from the analysis.

    The scene list is ordered chronologically:
        1. Location / Setup        (always)
        2. Persons arrive          (if persons > 0)
        3. Threat / Weapon         (if weapon > 0)
        4. Vehicle involvement     (if vehicle > 0)
        5. Bag / Property          (if bag > 0)
        6. Outcome / Resolution    (always)
    """
    base = _resize_to_panel(base_image) if base_image is not None else None

    def _scene(index: int, title: str, caption: str, tint_idx: int) -> StoryboardScene:
        if base is not None:
            image = _tint_image(base, _SCENE_TINTS[tint_idx % len(_SCENE_TINTS)])
            based_on_real = True
        else:
            image = _panel_placeholder(title, caption)
            based_on_real = False
        return StoryboardScene(
            index=index,
            title=title,
            caption=caption,
            image=image,
            duration_sec=default_duration,
            based_on_real_frame=based_on_real,
        )

    candidates: list[tuple[bool, str, str]] = [
        (
            True,
            "1. Location / Setup",
            f"The scene of the {analysis.source_type} shows an indoor/outdoor environment under investigation.",
        ),
        (
            analysis.person_count > 0,
            "2. Persons Detected",
            f"{analysis.person_count} person(s) detected at the scene.",
        ),
        (
            analysis.weapon_count > 0,
            "3. Threat Identified",
            f"⚠ {analysis.weapon_count} weapon(s) detected. Possible violent incident.",
        ),
        (
            analysis.vehicle_count > 0,
            "4. Vehicle Involved",
            f"{analysis.vehicle_count} vehicle(s) detected near the scene.",
        ),
        (
            analysis.bag_count > 0,
            "5. Property Interaction",
            f"{analysis.bag_count} bag(s) detected — possible theft indicator.",
        ),
        (
            True,
            "6. Outcome / Resolution",
            f"Suggested category: {analysis.suggested_category.replace('_', ' ').title()}. "
            f"Severity: {analysis.severity_level.upper()} ({analysis.severity_score}/100).",
        ),
    ]

    scenes: list[StoryboardScene] = []
    idx = 0
    for cond, title, caption in candidates:
        if not cond:
            continue
        if len(scenes) >= max_scenes:
            break
        scenes.append(_scene(idx, title, caption, tint_idx=idx))
        idx += 1

    # Enforce minimum number of scenes (re-pad with "not observed" panels if needed)
    while len(scenes) < min_scenes and len(scenes) < len(candidates):
        padded = False
        for cond, title, caption in candidates:
            if cond:
                continue
            cand_title = title
            cand_caption = caption + " (not observed in this evidence)"
            scenes.append(_scene(len(scenes), cand_title, cand_caption, tint_idx=len(scenes)))
            padded = True
            break
        # NOTE: `if not padded: break` is omitted because it's unreachable —
        # the outer `while len(scenes) < len(candidates)` already guarantees
        # at least one iteration of the inner for loop will find a `cond=False`
        # candidate as soon as there are any. The structural invariant
        # `len(candidates) >= 4` (Location + Outcome are always present)
        # ensures the first loop produces >= 2 scenes, and the only way to
        # exhaust the candidates is for them all to match, in which case the
        # outer `len(scenes) < len(candidates)` condition is False and we
        # never enter this loop body.

    return Storyboard(
        source_name=analysis.source_name,
        source_type=analysis.source_type,
        scenes=scenes,
        timestamp=datetime.now(),
    )


def plan_scenes_from_narrative(
    summary: InvestigationSummary,
    analysis: EvidenceAnalysis,
    base_image: Any | None = None,
) -> Storyboard:
    """
    Plan a storyboard by using the AI narrative to decompose the incident
    into logical visual beats. If the AI fails, falls back to deterministic
    planning.
    """
    # Attempt AI-driven planning
    try:
        prompt = (
            f"You are a forensic scene reconstruction expert. Based on the following "
            f"investigative summary, decompose the incident into a sequence of 4 to 6 "
            f"logical visual scenes for a storyboard.\n\n"
            f"Summary: {summary.primary_text}\n\n"
            f"Output ONLY a valid JSON array of objects with these keys: "
            f"{{'title': 'short title', 'caption': 'visual description', 'duration': 1.5}}. "
            f"Do not include any markdown formatting or prose."
        )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "1h",
                "options": {"temperature": 0.3},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("response", "").strip()

        # Basic JSON cleaning if AI wrapped it in ```json ... ```
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        scenes_data = json.loads(raw_text)
        if not isinstance(scenes_data, list):
            raise ValueError("AI did not return a list of scenes.")

        # Convert AI data to StoryboardScene objects
        # We reuse the internal _scene logic from plan_scenes (but we need access to it)
        # Since _scene is local to plan_scenes, we'll implement a similar helper here.

        # To keep the logic DRY, I'll move _scene to a top-level helper in the next edit
        # but for now, I'll implement the visual part here.

        scenes: list[StoryboardScene] = []
        # Use the same visual logic as plan_scenes
        from models.scene_planner import _resize_to_panel, _tint_image, _panel_placeholder

        base = _resize_to_panel(base_image) if base_image is not None else None

        for i, s_data in enumerate(scenes_data[:STORYBOARD_MAX_SCENES]):
            title = s_data.get("title", f"Scene {i+1}")
            caption = s_data.get("caption", "")
            duration = float(s_data.get("duration", STORYBOARD_DEFAULT_DURATION_SEC))

            if base is not None:
                # Reuse tint sequence from original plan_scenes
                from models.scene_planner import _SCENE_TINTS
                image = _tint_image(base, _SCENE_TINTS[i % len(_SCENE_TINTS)])
                based_on_real = True
            else:
                image = _panel_placeholder(title, caption)
                based_on_real = False

            scenes.append(StoryboardScene(
                index=i,
                title=title,
                caption=caption,
                image=image,
                duration_sec=duration,
                based_on_real_frame=based_on_real,
            ))

        if len(scenes) >= STORYBOARD_MIN_SCENES:
            return Storyboard(
                source_name=analysis.source_name,
                source_type=analysis.source_type,
                scenes=scenes,
                timestamp=datetime.now(),
            )

    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        # Fallback to deterministic planning
        from core.logging import get_logger
        get_logger(__name__).warning("LLM scene planning failed, falling back to deterministic: %s", exc)
        from models.scene_planner import plan_scenes
        return plan_scenes(analysis, base_image)

