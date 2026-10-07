"""
Advanced Scene Planner for Crime Scene Reconstruction.

This module converts a Narrative Summary and Situation Analysis into a
Structured Scene Narrative, which is then used to plan a 3D-like visual
storyboard.
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
from models.schemas import (
    EvidenceAnalysis,
    DetailedNarrativeSummary,
    CrimeSituationAnalysis,
    StructuredSceneNarrative,
    Storyboard,
    StoryboardScene,
)

class ScenePlanner:
    """
    Plans a forensic reconstruction by translating narrative and situational
    analysis into a structured visual plan.
    """

    def __init__(
        self,
        model_name: str | None = OLLAMA_MODEL,
    ) -> None:
        self.model_name = model_name

    def plan_structured_scene(
        self,
        summary: DetailedNarrativeSummary,
        situation: CrimeSituationAnalysis,
        analysis: EvidenceAnalysis,
    ) -> StructuredSceneNarrative:
        """
        Generate a StructuredSceneNarrative using Qwen3 14B.
        """
        if not self._is_ai_available():
            return self._fallback_scene_narrative(analysis)

        try:
            prompt = self._build_scene_prompt(summary, situation, analysis)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": "1h",
                "options": {"temperature": 0.3},
            }
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            result = response.json()

            data = json.loads(result.get("response", "{}"))

            return StructuredSceneNarrative(
                environment=data.get("environment", "Crime scene environment"),
                events=[
                    {
                        "time": e.get("time", "Unknown"),
                        "description": e.get("description", "Event detail unavailable"),
                        "evidence_ids": e.get("evidence_ids", []),
                        "confidence": e.get("confidence", "Unknown"),
                        "status": e.get("status", "UNKNOWN"),
                    }
                    for e in data.get("events", [])
                ],
                objects=data.get("objects", []),
                people_count=data.get("people_count"),
                camera_plan=data.get("camera_plan", {}),
                visualization_notes=data.get("visualization_notes", []),
            )
        except Exception as e:
            from core.logging import get_logger
            get_logger(__name__).warning("Scene Planning AI Error: %s", e)
            return self._fallback_scene_narrative(analysis)

    def generate_storyboard(
        self,
        scene_narrative: StructuredSceneNarrative,
        analysis: EvidenceAnalysis,
        base_image: Any | None = None,
    ) -> Storyboard:
        """
        Convert a StructuredSceneNarrative into a visual Storyboard.
        """
        from models.scene_planner import (
            _resize_to_panel, _tint_image, _panel_placeholder, _SCENE_TINTS
        )

        base = _resize_to_panel(base_image) if base_image is not None else None
        scenes: list[StoryboardScene] = []

        # Convert structured events into storyboard scenes
        for i, event in enumerate(scene_narrative.events):
            if i >= STORYBOARD_MAX_SCENES:
                break

            title = f"Event {i+1}: {event.get('time', 'Unknown')}"
            caption = event.get("description", "")

            if base is not None:
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
                duration_sec=STORYBOARD_DEFAULT_DURATION_SEC,
                based_on_real_frame=based_on_real,
            ))

        # Ensure minimum scene count
        while len(scenes) < STORYBOARD_MIN_SCENES:
            idx = len(scenes)
            scenes.append(StoryboardScene(
                index=idx,
                title=f"Scene {idx+1}: Context",
                caption="Environmental context for investigation.",
                image=_panel_placeholder("Context", "Environmental detail"),
                duration_sec=STORYBOARD_DEFAULT_DURATION_SEC,
                based_on_real_frame=False,
            ))

        return Storyboard(
            source_name=analysis.source_name,
            source_type=analysis.source_type,
            scenes=scenes,
            timestamp=datetime.now(),
        )

    def _is_ai_available(self) -> bool:
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception as e:
            from core.logging import get_logger
            get_logger(__name__).warning("AI Availability Check Error: %s", e)
            return False

    def _fallback_scene_narrative(self, analysis: EvidenceAnalysis) -> StructuredSceneNarrative:
        return StructuredSceneNarrative(
            environment="Generic crime scene",
            events=[{"time": "T+0", "description": "Incident start", "evidence_ids": [], "confidence": "Unknown", "status": "UNKNOWN"}],
            objects=[],
            people_count=analysis.person_count,
            camera_plan={},
            visualization_notes=["Fallback narrative used due to AI unavailability."],
        )

    def _build_scene_prompt(
        self,
        summary: DetailedNarrativeSummary,
        situation: CrimeSituationAnalysis,
        analysis: EvidenceAnalysis,
    ) -> str:
        prompt = (
            "You are a 3D scene reconstruction specialist. Convert the following forensic data "
            "into a structured scene plan for visualization. \n\n"
            f"SUMMARY:\n{summary.investigation_summary}\n"
            f"SITUATION:\n{situation.likely_activity_pattern}\n"
            f"SITUATION EVENTS:\n{situation.possible_sequence_of_events}\n"
            f"VERIFIED OBJECTS: {analysis.counts_by_label}\n\n"
            "Return a JSON object with: \n"
            "- environment: description of the setting\n"
            "- events: list of {time, description, evidence_ids, confidence, status}\n"
            "- objects: list of verified objects to render\n"
            "- people_count: integer\n"
            "- camera_plan: dict with {shot_type, angle, movement}\n"
            "- visualization_notes: list of details\n"
            "Use ONLY the provided facts. Do NOT invent scene details."
        )
        return prompt
