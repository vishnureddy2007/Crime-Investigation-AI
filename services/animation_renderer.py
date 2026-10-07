"""
Forensic Animation Renderer.
"""

from __future__ import annotations

import subprocess
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from datetime import datetime

from config import (
    STORYBOARD_VIDEO_FPS,
    STORYBOARD_VIDEO_CODEC,
)
from models.schemas import StructuredSceneNarrative, EvidenceAnalysis
from models.scene_3d import Scene3DGenerator, Scene3DPlan

class AnimationRenderer:
    def __init__(self, fps: int = STORYBOARD_VIDEO_FPS):
        self.fps = fps
        project_root = Path(__file__).resolve().parent.parent
        self.blender_script_path = project_root / "scripts" / "blender_render.py"

    def render_animated_video(
        self,
        scene_narrative: StructuredSceneNarrative | None,
        analysis: EvidenceAnalysis,
        output_path: Path
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from services.storyboard_bridge import StoryboardGenerator
        from services.crime_situation import CrimeSituationAnalyzer
        from models.summary_generator import SummaryGenerator

        sum_gen = SummaryGenerator()
        summary = sum_gen.generate(analysis)

        sit_analyzer = CrimeSituationAnalyzer()
        situation = sit_analyzer.analyze(analysis, summary_text=summary.investigation_summary)

        sb_gen = StoryboardGenerator()
        plan = sb_gen.generate_plan(1, analysis, summary, situation)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            json_path = temp_path / "scene_plan.json"
            with open(json_path, "w") as f:
                def json_serial(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")

                import dataclasses
                f.write(json.dumps(dataclasses.asdict(plan), indent=2, default=json_serial))

            frame_dir = temp_path / "frames"
            frame_dir.mkdir()
            self._invoke_blender(json_path, frame_dir)

            self._invoke_ffmpeg(frame_dir, output_path, scene_narrative)

        return output_path

    def _invoke_blender(self, json_path: Path, frame_dir: Path):
        if not self.blender_script_path.exists():
            raise RuntimeError(f"Blender script not found at {self.blender_script_path}")

        cmd = [
            "blender",
            "-b",
            "-P", str(self.blender_script_path),
            "--",
            str(json_path),
            str(frame_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        # Blender often prints warnings to stdout. Only fail if returncode != 0
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown Blender error"
            raise RuntimeError(f"Blender render failed with exit code {result.returncode}: {error_msg}")
        
        # Verification: check if frames were actually created
        if not any(frame_dir.glob("*.png")):
            raise RuntimeError("Blender executed but no frames were rendered to the output directory.")

    def _invoke_ffmpeg(self, frame_dir: Path, output_path: Path, narrative: StructuredSceneNarrative):
        input_pattern = str(frame_dir / "frame_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(self.fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

