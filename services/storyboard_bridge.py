from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from models.schemas import EvidenceAnalysis, DetailedNarrativeSummary, CrimeSituationAnalysis, StoryboardPlan, StoryboardEvent

class StoryboardGenerator:
    """
    The Bridge between AI Intelligence and 3D Rendering.
    Translates forensic narratives and predictions into actual Blender keyframes.
    """

    def __init__(self):
        self.PERSON_RADIUS = 2.0
        self.OBJ_RADIUS = 1.0

    def generate_plan(
        self,
        case_id: str,
        analysis: EvidenceAnalysis,
        summary: DetailedNarrativeSummary,
        situation: CrimeSituationAnalysis
    ) -> StoryboardPlan:
        """
        Merges forensic data and AI predictions into a structured animation plan.
        Hierarchy: Human Review > AI Detection > AI Prediction.
        """
        # 1. Define Actors (Objects and Persons)
        objects = []

        # Persons
        for i in range(analysis.person_count):
            objects.append({
                "id": f"person_{i+1}",
                "kind": "person",
                "label": f"Person {i+1}",
                "initial_pos": self._get_spread_pos(i, self.PERSON_RADIUS),
                "verified": True
            })

        # Weapons (Strictly Verified only)
        for i in range(analysis.verified_weapon_count):
            objects.append({
                "id": f"weapon_{i+1}",
                "kind": "weapon",
                "label": f"Weapon {i+1}",
                "initial_pos": self._get_spread_pos(i + 100, self.OBJ_RADIUS),
                "verified": True
            })

        # 2. Build the Narrative Timeline
        timeline = []

        # Sequence A: Verified Evidence (The "Facts")
        # We create events based on the verified detections
        for i, obj in enumerate(objects):
            timeline.append(StoryboardEvent(
                timestamp=1.0 + (i * 2.0),
                actor_id=obj["id"],
                action="HIGHLIGHT",
                target_pos=obj["initial_pos"],
                label=obj["label"],
                status="VERIFIED",
                description=f"Verified Evidence: {obj['label']} located."
            ))

        # Sequence B: Predicted Situation (The "Inference")
        # We translate the AI's predicted sequence into movements
        predicted_events = situation.possible_sequence_of_events.split(". ")
        start_offset = 10.0 # Start predictions after facts

        for i, event_text in enumerate(predicted_events):
            if not event_text.strip(): continue

            # Basic Actor Mapping
            actor_id = "person_1" # Default
            if "weapon" in event_text.lower():
                actor_id = "weapon_1" if analysis.verified_weapon_count > 0 else "person_1"
            elif "person" in event_text.lower():
                actor_id = "person_1"

            timeline.append(StoryboardEvent(
                timestamp=start_offset + (i * 4.0),
                actor_id=actor_id,
                action="move_to",
                target_pos=self._get_spread_pos(i + 200, 2.0),
                label=actor_id,
                status="PREDICTED",
                description=event_text.strip()
            ))

        # 3. Cinematic Camera Path
        camera_shots = [
            {"type": "establishing", "pos": (10, -10, 10), "target": (0, 0, 0), "duration": 5.0},
            {"type": "overview", "pos": (5, -5, 5), "target": (0, 0, 0), "duration": 5.0},
            {"type": "close_up", "pos": (2, -2, 2), "target": (0, 0, 0), "duration": 4.0},
        ]

        return StoryboardPlan(
            case_id=str(case_id),
            environment_bounds=[15.0, 15.0, 5.0],
            objects=objects,
            timeline=timeline,
            camera_shots=camera_shots,
            total_duration=start_offset + (len(predicted_events) * 4.0)
        )

    def _get_spread_pos(self, idx: int, radius: float) -> list[float]:
        import math
        angle = (2 * math.pi * idx) / 10.0
        return [round(radius * math.cos(angle), 2), round(radius * math.sin(angle), 2), 0.0]

    @staticmethod
    def calculate_state_hash(analysis: EvidenceAnalysis, summary: DetailedNarrativeSummary, situation: CrimeSituationAnalysis) -> str:
        state_string = f"{analysis.source_name}_{analysis.verified_weapon_count}_{summary.investigation_summary}_{situation.likely_activity_pattern}"
        return hashlib.sha256(state_string.encode()).hexdigest()
