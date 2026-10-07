from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from pathlib import Path

@dataclass(frozen=True)
class StoryboardObject:
    """An object that will be instantiated in the 3D scene."""
    id: str
    kind: Literal["person", "weapon", "vehicle", "bag", "generic", "environment"]
    label: str
    initial_pos: tuple[float, float, float]
    visual_props: dict[str, Any] = field(default_factory=dict) # e.g. {"color": "red", "alpha": 1.0}
    verified: bool = True

@dataclass(frozen=True)
class StoryboardEvent:
    """A dynamic change in the 3D scene anchored to the timeline."""
    timestamp: float
    actor_id: str
    action: Literal["move_to", "interact", "appear", "disappear"]
    target_pos: tuple[float, float, float] | None = None
    target_id: str | None = None
    duration: float = 1.0
    confidence_source: Literal["HUMAN_REVIEW", "AI_DETECTION", "AI_PREDICTION"] = "AI_DETECTION"
    caption: str = ""

@dataclass
class StoryboardPlan:
    """The complete declarative plan for the Blender render."""
    case_id: int
    environment_bounds: tuple[float, float, float]
    objects: list[StoryboardObject]
    timeline: list[StoryboardEvent]
    camera_shots: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)
