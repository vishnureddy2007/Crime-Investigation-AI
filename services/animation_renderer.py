"""
Forensic Animation / Video Renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import STORYBOARD_VIDEO_FPS
from models.schemas import EvidenceAnalysis, StructuredSceneNarrative
from services.investigation_video_generator import InvestigationVideoGenerator


class AnimationRenderer:
    def __init__(self, fps: int = STORYBOARD_VIDEO_FPS):
        self.fps = fps
        self.video_gen = InvestigationVideoGenerator(fps=fps)

    def render_animated_video(
        self,
        scene_narrative: StructuredSceneNarrative | None,
        analysis: EvidenceAnalysis,
        output_path: Path
    ) -> Path:
        """
        Render a 2D forensic investigation explanation video without Blender.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        case_id = getattr(analysis, "case_id", 1) or 1
        video_path, _ = self.video_gen.generate_video(
            case_id=case_id,
            force=True,
            output_dir=output_path.parent,
        )

        if video_path != output_path and video_path.exists():
            import shutil
            shutil.copy(video_path, output_path)

        return output_path


