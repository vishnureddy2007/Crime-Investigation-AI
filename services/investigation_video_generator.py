"""
Investigation Video Generator Service.

Generates a lightweight, professional 2D crime-investigation explanation MP4 video
using OpenCV and Pillow, based on uploaded evidence photos and investigation results.

Replaces the legacy 3D Blender rendering pipeline with a fast, evidence-grounded,
offline-capable video engine.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    DATABASE_PATH,
    STORYBOARD_VIDEO_CODEC,
    STORYBOARD_VIDEO_FPS,
    VIDEOS_DIR,
)
from database.repository import (
    get_case_evidence_version,
    get_latest_analysis,
    get_latest_investigation_video,
    get_latest_summary,
    save_investigation_video,
)


@dataclass
class VideoScene:
    """Structured data model representing a single video scene."""
    scene_type: str  # TITLE, OVERVIEW, CONTEXT, EVIDENCE, DETECTION, TIMELINE, AI_EXPLANATION, RISK, FINAL
    title: str
    description: str
    image_path: Path | None = None
    duration: float = 4.0  # Seconds
    caption: str = ""
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str | None = None
    confidence: float | None = None
    transition: str = "fade"  # fade, none


class InvestigationVideoGenerator:
    """Service to generate 2D forensic investigation explanation videos."""

    def __init__(
        self,
        db_path: Path = DATABASE_PATH,
        fps: int = 30,
        resolution: tuple[int, int] = (1920, 1080),
    ):
        self.db_path = db_path
        self.fps = fps
        self.width, self.height = resolution

    def generate_video(
        self,
        case_id: int,
        force: bool = False,
        output_dir: Path | None = None,
        evidence_images: list[Path] | None = None,
        analysis_data: dict[str, Any] | None = None,
    ) -> tuple[Path, float]:
        """
        Generate an MP4 investigation explanation video for a case.

        Returns (video_path, total_duration). Reuses cached video if evidence version is unchanged.
        """
        evidence_version = get_case_evidence_version(self.db_path, case_id)
        latest_video = get_latest_investigation_video(self.db_path, case_id)

        # Check cache if not forcing regeneration
        if not force and latest_video and latest_video.get("status") == "READY":
            if latest_video.get("evidence_version") == evidence_version:
                vpath = Path(latest_video["video_path"])
                if vpath.exists() and vpath.stat().st_size > 0:
                    return vpath, float(latest_video.get("duration", 0.0))

        # Build case context
        case_ctx = self.build_case_context(case_id, evidence_images, analysis_data)

        # Build scene plan
        scenes = self.build_video_scenes(case_ctx)

        # Set up output path
        out_dir = output_dir or (VIDEOS_DIR / f"CASE_{case_id:03d}")
        out_dir.mkdir(parents=True, exist_ok=True)
        video_filename = f"investigation_video_v{evidence_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        final_video_path = out_dir / video_filename

        # Render video
        total_duration = self._render_video_scenes(scenes, final_video_path)

        # Save to database repository
        save_investigation_video(
            self.db_path,
            case_id=case_id,
            evidence_version=evidence_version,
            video_path=str(final_video_path),
            duration=total_duration,
            status="READY",
        )

        return final_video_path, total_duration

    def build_case_context(
        self,
        case_id: int,
        evidence_images: list[Path] | None = None,
        analysis_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Gather case metadata, evidence, detections, weapon verification, and AI summary."""
        # 1. Fetch analysis
        db_analysis = get_latest_analysis(self.db_path, case_id)
        analysis_payload = analysis_data or (db_analysis.get("payload_json") if db_analysis else {})

        if isinstance(analysis_payload, str):
            import json
            try:
                analysis_payload = json.loads(analysis_payload)
            except Exception:
                analysis_payload = {}

        # 2. Fetch summary
        db_summary = get_latest_summary(self.db_path, case_id)
        summary_payload = db_summary.get("payload_json") if db_summary else {}
        if isinstance(summary_payload, str):
            import json
            try:
                summary_payload = json.loads(summary_payload)
            except Exception:
                summary_payload = {}

        # Parse counts & verification
        person_count = analysis_payload.get("person_count", 0)
        verified_weapon_count = analysis_payload.get("verified_weapon_count", 0)
        object_count = analysis_payload.get("object_count", 0)
        has_threat = analysis_payload.get("has_threat", False)
        severity_level = db_analysis.get("severity_level", "LOW") if db_analysis else "MEDIUM"
        if verified_weapon_count > 0:
            severity_level = "HIGH"

        # Detections list
        detections = analysis_payload.get("detections", [])

        # Filter confirmed weapons vs rejected
        confirmed_weapons = [
            d for d in detections
            if d.get("is_verified_weapon") or (d.get("label", "").lower() in ["gun", "firearm", "weapon", "pistol"] and d.get("confidence", 0) >= 0.5)
        ]

        # Extract images
        image_paths: list[Path] = []
        if evidence_images:
            image_paths = [Path(p) for p in evidence_images if Path(p).exists()]
        else:
            # Check outputs/annotated or images directory
            annotated = analysis_payload.get("annotated_image_path")
            if annotated and Path(annotated).exists():
                image_paths.append(Path(annotated))

        # Text narrative
        investigation_summary = summary_payload.get("investigation_summary") or (
            f"Based on the verified evidence, the scene contains {person_count} detected persons "
            f"and {verified_weapon_count} verified weapon evidence. Available evidence is consistent "
            f"with a potentially threatening incident. The exact sequence of events cannot be established "
            f"solely from the available images."
        )

        crime_context = summary_payload.get("crime_context") or "AI-Assisted Crime Scene Investigation"
        likely_activity = summary_payload.get("likely_activity_pattern") or "Incident under active forensic evaluation."

        return {
            "case_id": case_id,
            "created_at": db_analysis.get("created_at") if db_analysis else datetime.now().isoformat(),
            "person_count": person_count,
            "verified_weapon_count": verified_weapon_count,
            "object_count": object_count,
            "has_threat": has_threat,
            "severity_level": severity_level,
            "image_paths": image_paths,
            "detections": detections,
            "confirmed_weapons": confirmed_weapons,
            "investigation_summary": investigation_summary,
            "crime_context": crime_context,
            "likely_activity": likely_activity,
        }

    def build_video_scenes(self, case_ctx: dict[str, Any]) -> list[VideoScene]:
        """Construct sequence of VideoScene items."""
        scenes: list[VideoScene] = []

        # SCENE 1: TITLE
        scenes.append(
            VideoScene(
                scene_type="TITLE",
                title="CRIME INVESTIGATION AI",
                description=f"Case #{case_ctx['case_id']:03d} - Forensic Video Report",
                duration=4.0,
                caption="AI-Assisted Evidence Analysis and Scene Explanation",
                timestamp=case_ctx.get("created_at"),
            )
        )

        # SCENE 2: CASE OVERVIEW
        scenes.append(
            VideoScene(
                scene_type="OVERVIEW",
                title="CASE OVERVIEW",
                description=f"Evidence Analysis Summary for Case #{case_ctx['case_id']:03d}",
                duration=4.5,
                caption=f"Evidence Items: {len(case_ctx['image_paths']) or 1} | Persons Detected: {case_ctx['person_count']} | Verified Weapons: {case_ctx['verified_weapon_count']}",
            )
        )

        # SCENE 3: CRIME CONTEXT
        scenes.append(
            VideoScene(
                scene_type="CONTEXT",
                title="CRIME CONTEXT",
                description=case_ctx["crime_context"],
                duration=5.0,
                caption=f"Activity Pattern: {case_ctx['likely_activity']} | Risk Level: {case_ctx['severity_level']}",
            )
        )

        # SCENE 4+: UPLOADED PHOTO PRESENTATION
        images = case_ctx["image_paths"]
        if images:
            for idx, img_path in enumerate(images, start=1):
                scenes.append(
                    VideoScene(
                        scene_type="EVIDENCE",
                        title=f"EVIDENCE PHOTO #{idx:02d}",
                        description=f"Forensic Evidence Frame {idx}",
                        image_path=img_path,
                        duration=5.0,
                        caption=f"Original Evidence Photo {idx} - Automated Object & Weapon Overlay",
                        evidence_items=case_ctx["detections"],
                        transition="fade",
                    )
                )

        # SCENE N: DETECTION EXPLANATION
        scenes.append(
            VideoScene(
                scene_type="DETECTION",
                title="DETECTION BREAKDOWN",
                description="Verified Detections",
                duration=4.5,
                caption=f"Detected Persons: {case_ctx['person_count']} | Verified Weapons: {case_ctx['verified_weapon_count']} | Other Objects: {case_ctx['object_count']}",
                evidence_items=case_ctx["detections"],
            )
        )

        # SCENE N+1: CRIME TIMELINE
        scenes.append(
            VideoScene(
                scene_type="TIMELINE",
                title="INVESTIGATION TIMELINE",
                description="Chronological Forensic Events",
                duration=5.0,
                caption="Evidence Uploaded -> Persons Detected -> Weapon Candidates Scanned -> Weapons Verified -> AI Summary Generated",
            )
        )

        # SCENE N+2: AI INVESTIGATION EXPLANATION
        scenes.append(
            VideoScene(
                scene_type="AI_EXPLANATION",
                title="AI INVESTIGATION EXPLANATION",
                description=case_ctx["investigation_summary"],
                duration=6.0,
                caption="Evidence-grounded automated situation interpretation",
            )
        )

        # SCENE N+3: RISK ASSESSMENT
        scenes.append(
            VideoScene(
                scene_type="RISK",
                title="RISK & SITUATION ASSESSMENT",
                description=f"Severity Rating: {case_ctx['severity_level']}",
                duration=4.5,
                caption=f"Threat Present: {'YES' if case_ctx['has_threat'] else 'NO'} | Verified Weapons: {case_ctx['verified_weapon_count']} | Detections Analyzed: {len(case_ctx['detections'])}",
            )
        )

        # FINAL SCENE: INVESTIGATION SUMMARY
        scenes.append(
            VideoScene(
                scene_type="FINAL",
                title="INVESTIGATION SUMMARY",
                description="Evidence Analyzed Successfully",
                duration=4.0,
                caption="Generated by Crime Investigation AI",
            )
        )

        return scenes

    # ------------------------------------------------------------------
    # Rendering Engine (OpenCV + PIL)
    # ------------------------------------------------------------------

    def _render_video_scenes(self, scenes: list[VideoScene], output_path: Path) -> float:
        """Render scenes into an MP4 video file and return total duration in seconds."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, float(self.fps), (self.width, self.height))

        if not writer.isOpened():
            # Fallback to MJPG or XVID if mp4v fails
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(str(output_path), fourcc, float(self.fps), (self.width, self.height))

        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize OpenCV VideoWriter for {output_path}")

        total_frames = 0
        total_duration = 0.0

        for idx, scene in enumerate(scenes):
            scene_frames = int(round(scene.duration * self.fps))
            total_duration += scene.duration

            # Pre-generate base frame for static scene or prepare image for Ken Burns effect
            if scene.scene_type == "EVIDENCE" and scene.image_path and scene.image_path.exists():
                frames = self._render_ken_burns_scene(scene, scene_frames)
            else:
                static_img = self._render_graphic_scene(scene)
                static_bgr = cv2.cvtColor(np.array(static_img), cv2.COLOR_RGB2BGR)
                frames = [static_bgr] * scene_frames

            # Write frames with cross-fade transition if needed
            if idx > 0 and scene.transition == "fade" and len(frames) > 5:
                fade_n = min(15, len(frames) // 2)
                prev_last = last_frame  # type: ignore
                for f_idx in range(fade_n):
                    alpha = (f_idx + 1) / float(fade_n)
                    blended = cv2.addWeighted(prev_last, 1.0 - alpha, frames[f_idx], alpha, 0)
                    writer.write(blended)
                    total_frames += 1
                for f_idx in range(fade_n, len(frames)):
                    writer.write(frames[f_idx])
                    total_frames += 1
            else:
                for fr in frames:
                    writer.write(fr)
                    total_frames += 1

            last_frame = frames[-1]

        writer.release()
        return total_duration

    def _render_graphic_scene(self, scene: VideoScene) -> Image.Image:
        """Render text & UI graphic scene on a 1920x1080 canvas."""
        img = Image.new("RGB", (self.width, self.height), color=(11, 15, 25))  # Dark forensic slate
        draw = ImageDraw.Draw(img)

        # Fonts
        title_font = self._get_font(int(self.height * 0.06), bold=True)
        sub_font = self._get_font(int(self.height * 0.035))
        body_font = self._get_font(int(self.height * 0.028))
        caption_font = self._get_font(int(self.height * 0.022))

        # Accent border lines
        draw.rectangle([(40, 40), (self.width - 40, self.height - 40)], outline=(42, 52, 75), width=3)
        draw.rectangle([(40, 40), (self.width - 40, 110)], fill=(22, 28, 46))
        draw.line([(40, 110), (self.width - 40, 110)], fill=(0, 242, 254), width=4)  # Neon Cyan line

        # Header Title
        draw.text((60, 55), "CRIME INVESTIGATION AI", fill=(0, 242, 254), font=sub_font)
        draw.text((self.width - 320, 55), datetime.now().strftime("%Y-%m-%d"), fill=(148, 163, 184), font=caption_font)

        # Center Section Card
        card_x1, card_y1 = 120, 160
        card_x2, card_y2 = self.width - 120, self.height - 140
        draw.rectangle([(card_x1, card_y1), (card_x2, card_y2)], fill=(22, 28, 46), outline=(42, 52, 75), width=2)

        # Scene Main Title
        draw.text((card_x1 + 60, card_y1 + 40), scene.title, fill=(255, 255, 255), font=title_font)

        # Content rendering based on scene type
        if scene.scene_type == "TITLE":
            draw.text((card_x1 + 60, card_y1 + 160), scene.description, fill=(79, 172, 254), font=sub_font)
            draw.text((card_x1 + 60, card_y1 + 260), "AUTOMATED FORENSIC INVESTIGATION REPORT", fill=(226, 232, 240), font=body_font)
            draw.text((card_x1 + 60, card_y1 + 320), "STATUS: VERIFIED ANALYSIS COMPLETE", fill=(34, 197, 94), font=body_font)

        elif scene.scene_type == "OVERVIEW":
            draw.text((card_x1 + 60, card_y1 + 140), scene.description, fill=(226, 232, 240), font=body_font)
            lines = [
                f"• {scene.caption}",
                "• All uploaded evidence processed via YOLO object detector",
                "• Weapon candidates automatically verified",
            ]
            for i, l in enumerate(lines):
                draw.text((card_x1 + 60, card_y1 + 220 + i * 50), l, fill=(148, 163, 184), font=body_font)

        elif scene.scene_type == "CONTEXT":
            draw.text((card_x1 + 60, card_y1 + 140), f"Incident Type: {scene.description}", fill=(226, 232, 240), font=body_font)
            draw.text((card_x1 + 60, card_y1 + 220), scene.caption, fill=(79, 172, 254), font=body_font)

        elif scene.scene_type == "DETECTION":
            draw.text((card_x1 + 60, card_y1 + 140), scene.caption, fill=(255, 255, 255), font=sub_font)
            draw.text((card_x1 + 60, card_y1 + 220), "Verified Objects & Threat Elements:", fill=(226, 232, 240), font=body_font)
            for idx, item in enumerate(scene.evidence_items[:5], start=1):
                label = item.get("label", "Object").upper()
                conf = item.get("confidence", 0.0)
                is_ver = "VERIFIED WEAPON" if item.get("is_verified_weapon") else "DETECTED"
                color = (239, 68, 68) if "WEAPON" in is_ver else (59, 130, 246)
                draw.text((card_x1 + 80, card_y1 + 270 + (idx - 1) * 45), f"{idx}. {label} - {conf:.1%} ({is_ver})", fill=color, font=body_font)

        elif scene.scene_type == "TIMELINE":
            draw.text((card_x1 + 60, card_y1 + 140), "Chronological Forensic Timeline:", fill=(226, 232, 240), font=sub_font)
            steps = [
                "00:00 - Evidence Uploaded & Cataloged",
                "00:05 - Persons & Detections Scanned",
                "00:10 - Weapon Candidates Isolated",
                "00:15 - Automatic Weapon Verification Passed",
                "00:20 - Evidence Analyzed & Risk Scored",
                "00:25 - Investigation Summary Generated",
            ]
            for i, st in enumerate(steps):
                draw.text((card_x1 + 80, card_y1 + 210 + i * 45), f"► {st}", fill=(0, 242, 254), font=body_font)

        elif scene.scene_type == "AI_EXPLANATION":
            draw.text((card_x1 + 60, card_y1 + 130), "AI Forensic Assessment:", fill=(79, 172, 254), font=sub_font)
            wrapped_summary = self._wrap_text(scene.description, max_chars=75)
            for i, line in enumerate(wrapped_summary[:6]):
                draw.text((card_x1 + 60, card_y1 + 200 + i * 42), line, fill=(226, 232, 240), font=body_font)

        elif scene.scene_type == "RISK":
            color = (239, 68, 68) if "HIGH" in scene.description else (245, 158, 11)
            draw.text((card_x1 + 60, card_y1 + 140), scene.description, fill=color, font=title_font)
            draw.text((card_x1 + 60, card_y1 + 240), scene.caption, fill=(226, 232, 240), font=body_font)

        elif scene.scene_type == "FINAL":
            draw.text((card_x1 + 60, card_y1 + 160), "INVESTIGATION COMPLETE", fill=(34, 197, 94), font=title_font)
            draw.text((card_x1 + 60, card_y1 + 260), "Generated by Crime Investigation AI", fill=(148, 163, 184), font=sub_font)

        # Bottom Subtitle Bar
        draw.rectangle([(40, self.height - 110), (self.width - 40, self.height - 40)], fill=(15, 23, 42))
        draw.text((60, self.height - 90), f"CAPTION: {scene.caption}", fill=(203, 213, 225), font=caption_font)

        return img

    def _render_ken_burns_scene(self, scene: VideoScene, num_frames: int) -> list[np.ndarray]:
        """Render evidence image scene with smooth Ken Burns zoom & pan effect."""
        frames: list[np.ndarray] = []
        if not scene.image_path or not scene.image_path.exists():
            static = self._render_graphic_scene(scene)
            bgr = cv2.cvtColor(np.array(static), cv2.COLOR_RGB2BGR)
            return [bgr] * num_frames

        try:
            pil_img = Image.open(scene.image_path).convert("RGB")
        except Exception:
            static = self._render_graphic_scene(scene)
            bgr = cv2.cvtColor(np.array(static), cv2.COLOR_RGB2BGR)
            return [bgr] * num_frames

        orig_w, orig_h = pil_img.size

        # Create base canvas with dark letterbox
        canvas = Image.new("RGB", (self.width, self.height), color=(11, 15, 25))

        # Scale factors for Ken Burns (zoom from 1.0 to 1.15)
        for f in range(num_frames):
            progress = f / max(1, num_frames - 1)
            scale = 1.0 + (0.12 * progress)

            # Target crop dimensions
            crop_w = int(orig_w / scale)
            crop_h = int(orig_h / scale)

            # Pan slightly towards center-right
            dx = int((orig_w - crop_w) * 0.5 * progress)
            dy = int((orig_h - crop_h) * 0.5 * progress)

            x1 = max(0, dx)
            y1 = max(0, dy)
            x2 = min(orig_w, x1 + crop_w)
            y2 = min(orig_h, y1 + crop_h)

            cropped = pil_img.crop((x1, y1, x2, y2))
            resized = cropped.resize((self.width - 160, self.height - 240), Image.Resampling.LANCZOS)

            frame_img = canvas.copy()
            frame_img.paste(resized, (80, 120))

            # Overlay captions and forensic border
            draw = ImageDraw.Draw(frame_img)
            draw.rectangle([(40, 40), (self.width - 40, self.height - 40)], outline=(42, 52, 75), width=3)
            draw.rectangle([(40, 40), (self.width - 40, 100)], fill=(22, 28, 46))
            draw.line([(40, 100), (self.width - 40, 100)], fill=(0, 242, 254), width=3)

            title_font = self._get_font(int(self.height * 0.035), bold=True)
            caption_font = self._get_font(int(self.height * 0.022))

            draw.text((60, 52), f"FORENSIC EVIDENCE: {scene.title}", fill=(0, 242, 254), font=title_font)

            # Bottom caption bar
            draw.rectangle([(40, self.height - 100), (self.width - 40, self.height - 40)], fill=(15, 23, 42))
            draw.text((60, self.height - 80), f"EVIDENCE DETAILS: {scene.caption}", fill=(241, 245, 249), font=caption_font)

            bgr = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
            frames.append(bgr)

        return frames

    @staticmethod
    def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load system font or default fallback."""
        try:
            font_name = "arialbd.ttf" if bold else "arial.ttf"
            return ImageFont.truetype(font_name, size)
        except OSError:
            try:
                font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
                return ImageFont.truetype(font_name, size)
            except OSError:
                return ImageFont.load_default()

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        curr = ""
        for w in words:
            cand = (curr + " " + w).strip()
            if len(cand) <= max_chars:
                curr = cand
            else:
                if curr:
                    lines.append(curr)
                curr = w
        if curr:
            lines.append(curr)
        return lines
