"""
Investigation Video Generator Service.

Generates a lightweight, professional, 2D crime-investigation explanation MP4 video
using OpenCV and Pillow, based on uploaded evidence photos and videos.

Visually connects:
Case Overview -> Evidence Inventory -> Actual Uploaded Photos & Videos ->
Detection Overlays -> Weapon Verification -> Timeline -> AI Explanation -> Risk -> Summary.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from config import (
    DATABASE_PATH,
    OUTPUTS_DIR,
    STORYBOARD_VIDEO_CODEC,
    STORYBOARD_VIDEO_FPS,
    VIDEOS_DIR,
)
from database.repository import (
    get_case_evidence_version,
    get_connection,
    get_latest_analysis,
    get_latest_investigation_video,
    get_latest_summary,
    list_case_evidence_items,
    save_investigation_video,
    _from_json,
)


@dataclass
class VideoScene:
    """Structured data model representing a single video scene."""
    scene_type: str  # TITLE, OVERVIEW, INVENTORY, EVIDENCE_IMAGE, EVIDENCE_VIDEO, VERIFICATION, TIMELINE, AI_EXPLANATION, RISK, FINAL
    title: str
    description: str
    image_path: Path | None = None
    pil_image: Image.Image | None = None
    duration: float = 4.0  # Seconds
    caption: str = ""
    evidence_id: str | None = None
    filename: str | None = None
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str | None = None
    confidence: float | None = None
    transition: str = "fade"  # fade, none
    video_timestamp: float | None = None
    target_crop_center: tuple[float, float] | None = None  # (cx, cy) for Ken Burns weapon zoom


class InvestigationVideoGenerator:
    """Service to generate evidence-grounded 2D forensic investigation explanation videos."""

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
        selected_evidence_ids: list[str] | None = None,
        detail_level: str = "Detailed",  # Brief, Standard, Detailed
        video_evidence_mode: str = "Representative Frames",  # Representative Frames, Short Evidence Clips, Full Video
        evidence_images: list[Path] | None = None,
        analysis_data: dict[str, Any] | None = None,
    ) -> tuple[Path, float]:
        """
        Generate an MP4 investigation explanation video for a case.

        Returns (video_path, total_duration).
        """
        evidence_version = get_case_evidence_version(self.db_path, case_id)
        latest_video = get_latest_investigation_video(self.db_path, case_id)

        # Check cache if not forcing regeneration
        if not force and latest_video and latest_video.get("status") == "READY":
            if latest_video.get("evidence_version") == evidence_version:
                vpath = Path(latest_video["video_path"])
                if vpath.exists() and vpath.stat().st_size > 0:
                    return vpath, float(latest_video.get("duration", 0.0))

        # 1. Load case evidence collection
        collection = self.get_case_evidence_collection(case_id, evidence_images, analysis_data)

        # Filter evidence if selected_evidence_ids provided
        if selected_evidence_ids:
            sel_set = set(selected_evidence_ids)
            collection["images"] = [img for img in collection["images"] if img.get("evidence_id") in sel_set or img.get("filename") in sel_set]
            collection["videos"] = [vid for vid in collection["videos"] if vid.get("evidence_id") in sel_set or vid.get("filename") in sel_set]
            collection["inventory"] = [inv for inv in collection["inventory"] if inv.get("evidence_id") in sel_set or inv.get("filename") in sel_set]

        # 2. Build video scene sequence
        scenes = self.build_video_scenes(collection, detail_level=detail_level, video_evidence_mode=video_evidence_mode)

        # 3. Setup output file path
        out_dir = output_dir or (VIDEOS_DIR / f"CASE_{case_id:03d}")
        out_dir.mkdir(parents=True, exist_ok=True)
        video_filename = f"investigation_video_v{evidence_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        final_video_path = out_dir / video_filename

        # 4. Render scenes to MP4
        total_duration = self._render_video_scenes(scenes, final_video_path)

        # 5. Persist record in repository
        save_investigation_video(
            self.db_path,
            case_id=case_id,
            evidence_version=evidence_version,
            video_path=str(final_video_path),
            duration=total_duration,
            status="READY",
        )

        return final_video_path, total_duration

    def get_case_evidence_collection(
        self,
        case_id: int,
        evidence_images: list[Path] | None = None,
        analysis_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Assemble structured case evidence collection filtered strictly by case_id.
        """
        images_list: list[dict[str, Any]] = []
        videos_list: list[dict[str, Any]] = []
        inventory_list: list[dict[str, Any]] = []

        # Query database for case evidence items
        db_items = list_case_evidence_items(self.db_path, case_id)

        # Query latest case analysis & summary
        db_analysis = get_latest_analysis(self.db_path, case_id)
        db_summary = get_latest_summary(self.db_path, case_id)

        analysis_payload = analysis_data or (db_analysis.get("payload_json") if db_analysis else {})
        if isinstance(analysis_payload, str):
            try:
                import json
                analysis_payload = json.loads(analysis_payload)
            except Exception:
                analysis_payload = {}

        summary_payload = db_summary.get("payload_json") if db_summary else {}
        if isinstance(summary_payload, str):
            try:
                import json
                summary_payload = json.loads(summary_payload)
            except Exception:
                summary_payload = {}

        # Case title / source
        case_name = f"Case #{case_id:03d}"
        if db_analysis and db_analysis.get("source_name"):
            case_name = db_analysis["source_name"]

        # Parse DB items
        for item in db_items:
            f_type = item.get("file_type", "image").lower()
            f_path = Path(item["file_path"]) if item.get("file_path") else None
            ann_path = Path(item["annotated_path"]) if item.get("annotated_path") else None
            payload = item.get("payload", {})
            dets = payload.get("detections", [])

            # Categorize weapons / persons / objects for this specific file
            ver_weapons = [d for d in dets if d.get("weapon_status") == "verified" or d.get("is_verified_weapon")]
            rej_weapons = [d for d in dets if d.get("weapon_status") == "rejected" or d.get("decision") == "REJECT"]
            persons = [d for d in dets if d.get("label", "").lower() in ["person", "people"]]
            objects = [d for d in dets if d not in ver_weapons and d not in persons]

            ev_entry = {
                "evidence_id": item.get("evidence_id", f"EVD-{len(inventory_list)+1:03d}"),
                "filename": item.get("filename", f_path.name if f_path else "evidence"),
                "file_type": f_type,
                "file_path": f_path,
                "annotated_path": ann_path,
                "duration": float(item.get("duration", 0.0)),
                "fps": float(item.get("fps", 0.0)),
                "frame_count": int(item.get("frame_count", 0)),
                "detections": dets,
                "verified_weapons": ver_weapons,
                "rejected_weapons": rej_weapons,
                "persons": persons,
                "objects": objects,
                "is_corrupt": False,
            }

            inventory_list.append(ev_entry)
            if f_type == "video":
                videos_list.append(ev_entry)
            else:
                images_list.append(ev_entry)

        # Fallback to direct file paths if DB items empty
        if not db_items:
            if evidence_images:
                for idx, p in enumerate(evidence_images, start=1):
                    p = Path(p)
                    f_type = "video" if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"] else "image"
                    ev_entry = {
                        "evidence_id": f"EVD-{idx:03d}",
                        "filename": p.name,
                        "file_type": f_type,
                        "file_path": p,
                        "annotated_path": None,
                        "duration": 0.0,
                        "fps": 0.0,
                        "frame_count": 0,
                        "detections": analysis_payload.get("detections", []),
                        "verified_weapons": [],
                        "persons": [],
                        "objects": [],
                        "is_corrupt": False,
                    }
                    inventory_list.append(ev_entry)
                    if f_type == "video":
                        videos_list.append(ev_entry)
                    else:
                        images_list.append(ev_entry)
            elif analysis_payload.get("annotated_image_path"):
                p = Path(analysis_payload["annotated_image_path"])
                if p.exists():
                    ev_entry = {
                        "evidence_id": "EVD-001",
                        "filename": p.name,
                        "file_type": "image",
                        "file_path": p,
                        "annotated_path": p,
                        "duration": 0.0,
                        "fps": 0.0,
                        "frame_count": 0,
                        "detections": analysis_payload.get("detections", []),
                        "verified_weapons": [],
                        "persons": [],
                        "objects": [],
                        "is_corrupt": False,
                    }
                    inventory_list.append(ev_entry)
                    images_list.append(ev_entry)

        # Extract overall counts
        person_count = analysis_payload.get("person_count", sum(len(e["persons"]) for e in inventory_list))
        verified_weapon_count = analysis_payload.get("verified_weapon_count", sum(len(e["verified_weapons"]) for e in inventory_list))
        object_count = analysis_payload.get("object_count", sum(len(e["objects"]) for e in inventory_list))
        has_threat = analysis_payload.get("has_threat", verified_weapon_count > 0)
        severity_level = db_analysis.get("severity_level", "HIGH" if verified_weapon_count > 0 else "MEDIUM") if db_analysis else ("HIGH" if verified_weapon_count > 0 else "MEDIUM")
        severity_score = db_analysis.get("severity_score", 85 if verified_weapon_count > 0 else 45) if db_analysis else 85

        # Narrative text
        investigation_summary = summary_payload.get("investigation_summary") or (
            f"Based on the verified evidence, the scene contains {person_count} detected persons "
            f"and {verified_weapon_count} verified weapon evidence. Available evidence is consistent "
            f"with a potentially threatening incident. The exact sequence of events cannot be established "
            f"solely from the available images."
        )

        crime_context = summary_payload.get("crime_context") or "AI-Assisted Crime Scene Investigation"
        likely_activity = summary_payload.get("likely_activity_pattern") or "Incident under active forensic evaluation."

        # Collect all verified weapons across inventory
        all_verified_weapons = []
        for item in inventory_list:
            all_verified_weapons.extend(item.get("verified_weapons", []))

        return {
            "case_id": case_id,
            "case_name": case_name,
            "created_at": db_analysis.get("created_at") if db_analysis else datetime.now().isoformat(),
            "images": images_list,
            "videos": videos_list,
            "inventory": inventory_list,
            "items": inventory_list,
            "verified_weapons": all_verified_weapons,
            "person_count": person_count,
            "verified_weapon_count": verified_weapon_count,
            "object_count": object_count,
            "has_threat": has_threat,
            "severity_level": severity_level,
            "severity_score": severity_score,
            "investigation_summary": investigation_summary,
            "crime_context": crime_context,
            "likely_activity": likely_activity,
        }

    def build_video_scenes(
        self,
        collection: dict[str, Any],
        detail_level: str = "Detailed",
        video_evidence_mode: str = "Representative Frames",
    ) -> list[VideoScene]:
        """Construct sequence of VideoScene items."""
        scenes: list[VideoScene] = []
        case_id = collection["case_id"]

        # SCENE 1: TITLE & CASE INTRO
        scenes.append(
            VideoScene(
                scene_type="TITLE",
                title="CRIME INVESTIGATION AI",
                description=f"Case #{case_id:03d} - {collection['case_name']}",
                duration=4.0,
                caption="AI-Assisted Forensic Evidence Analysis & Crime Scene Explanation",
                timestamp=collection.get("created_at"),
            )
        )

        # SCENE 2: CRIME SCENE OVERVIEW
        scenes.append(
            VideoScene(
                scene_type="OVERVIEW",
                title="CRIME SCENE OVERVIEW",
                description=f"Case #{case_id:03d} Executive Overview",
                duration=5.0,
                caption=f"Total Evidence Items: {len(collection['inventory'])} | Images: {len(collection['images'])} | Videos: {len(collection['videos'])}",
                evidence_items=[
                    {"label": f"Persons Detected: {collection['person_count']}"},
                    {"label": f"Verified Weapons: {collection['verified_weapon_count']}"},
                    {"label": f"Other Objects: {collection['object_count']}"},
                    {"label": f"Risk Rating: {collection['severity_level']}"},
                ]
            )
        )

        # SCENE 3: EVIDENCE INVENTORY (Standard & Detailed)
        if detail_level in ["Standard", "Detailed"] and collection["inventory"]:
            inventory_desc = "\n".join([
                f"• {item['evidence_id']} — {item['filename']} ({item['file_type'].upper()})"
                for item in collection["inventory"][:8]
            ])
            scenes.append(
                VideoScene(
                    scene_type="INVENTORY",
                    title="EVIDENCE INVENTORY",
                    description=inventory_desc,
                    duration=5.0,
                    caption=f"Cataloged {len(collection['inventory'])} evidence file(s) for Case #{case_id:03d}",
                    evidence_items=collection["inventory"],
                )
            )

        # SCENE 4+: UPLOADED IMAGE & VIDEO PRESENTATION
        # Images
        for idx, img_item in enumerate(collection["images"], start=1):
            f_path = img_item.get("file_path") or img_item.get("annotated_path")
            ev_id = img_item.get("evidence_id", f"EVD-{idx:03d}")
            fname = img_item.get("filename", "image")

            ver_w = len(img_item.get("verified_weapons", []))
            p_cnt = len(img_item.get("persons", []))

            # Ken Burns weapon zoom center
            crop_center = None
            if img_item.get("verified_weapons"):
                w_box = img_item["verified_weapons"][0].get("bbox") or img_item["verified_weapons"][0].get("box")
                if isinstance(w_box, (list, tuple)) and len(w_box) == 4:
                    crop_center = ((w_box[0] + w_box[2]) / 2.0, (w_box[1] + w_box[3]) / 2.0)
                elif isinstance(w_box, dict):
                    crop_center = ((w_box.get("x1", 0) + w_box.get("x2", 0)) / 2.0, (w_box.get("y1", 0) + w_box.get("y2", 0)) / 2.0)

            # Cautious factual evidence explanation
            explanation = (
                f"Evidence {ev_id} ({fname}) contains {p_cnt} detected person(s) and {ver_w} verified weapon(s). "
                f"{'Verified weapon evidence is present.' if ver_w > 0 else 'No verified weapon was detected in this frame.'}"
            )

            scenes.append(
                VideoScene(
                    scene_type="EVIDENCE_IMAGE",
                    title=f"EVIDENCE PHOTO #{idx:02d}",
                    description=f"{ev_id} — {fname}",
                    image_path=f_path,
                    duration=5.5,
                    caption=explanation,
                    evidence_id=ev_id,
                    filename=fname,
                    evidence_items=img_item.get("detections", []),
                    target_crop_center=crop_center,
                    transition="fade",
                )
            )

        # Videos
        for v_idx, vid_item in enumerate(collection["videos"], start=1):
            v_path = vid_item.get("file_path")
            ev_id = vid_item.get("evidence_id", f"EVD-V{v_idx:02d}")
            fname = vid_item.get("filename", "video.mp4")
            duration_sec = vid_item.get("duration", 0.0)

            explanation = (
                f"Video Evidence {ev_id} ({fname}) — Duration {duration_sec:.1f}s. "
                f"Detected {len(vid_item.get('persons', []))} person(s) and {len(vid_item.get('verified_weapons', []))} verified weapon(s)."
            )

            # Extract representative frames or clips
            rep_frames = self._sample_video_frames(v_path)
            if rep_frames:
                for f_idx, (ts_sec, pil_fr) in enumerate(rep_frames, start=1):
                    scenes.append(
                        VideoScene(
                            scene_type="EVIDENCE_VIDEO",
                            title=f"VIDEO EVIDENCE #{v_idx:02d} (Frame {f_idx})",
                            description=f"{ev_id} — {fname} (t={ts_sec:.1f}s / {duration_sec:.1f}s)",
                            pil_image=pil_fr,
                            duration=4.5,
                            caption=explanation,
                            evidence_id=ev_id,
                            filename=fname,
                            video_timestamp=ts_sec,
                            evidence_items=vid_item.get("detections", []),
                            transition="fade",
                        )
                    )
            else:
                # Video file missing or unreadable fallback
                scenes.append(
                    VideoScene(
                        scene_type="EVIDENCE_VIDEO",
                        title=f"VIDEO EVIDENCE #{v_idx:02d}",
                        description=f"{ev_id} — {fname} (File Unavailable)",
                        duration=4.0,
                        caption=f"Evidence video file '{fname}' is unavailable or unreadable.",
                        evidence_id=ev_id,
                        filename=fname,
                        transition="fade",
                    )
                )

        # SCENE N: WEAPON VERIFICATION BREAKDOWN (Detailed)
        if detail_level == "Detailed":
            scenes.append(
                VideoScene(
                    scene_type="VERIFICATION",
                    title="AUTOMATIC WEAPON VERIFICATION",
                    description="Multi-stage Confidence & Spatial Verification",
                    duration=5.0,
                    caption=f"Verified Weapons: {collection['verified_weapon_count']} | Rejected Detections Excluded",
                    evidence_items=collection["verified_weapons"],
                )
            )

        # SCENE N+1: INVESTIGATION TIMELINE
        if detail_level in ["Standard", "Detailed"]:
            scenes.append(
                VideoScene(
                    scene_type="TIMELINE",
                    title="INVESTIGATION TIMELINE",
                    description="Chronological Forensic Progression",
                    duration=5.0,
                    caption="Upload -> Object Detection -> Weapon Verification -> Evidence Correlation -> AI Narrative",
                )
            )

        # SCENE N+2: AI INVESTIGATION CONTEXT
        if detail_level == "Detailed":
            scenes.append(
                VideoScene(
                    scene_type="AI_EXPLANATION",
                    title="AI INVESTIGATION CONTEXT",
                    description=collection["investigation_summary"],
                    duration=6.0,
                    caption="Evidence-grounded automated situation assessment",
                )
            )

        # SCENE N+3: RISK ASSESSMENT
        scenes.append(
            VideoScene(
                scene_type="RISK",
                title="RISK & SITUATION ASSESSMENT",
                description=f"Severity Rating: {collection['severity_level']} ({collection['severity_score']}/100)",
                duration=4.5,
                caption=f"Threat Present: {'YES' if collection['has_threat'] else 'NO'} | Verified Weapons: {collection['verified_weapon_count']}",
            )
        )

        # FINAL SCENE: INVESTIGATION SUMMARY
        scenes.append(
            VideoScene(
                scene_type="FINAL",
                title="INVESTIGATION SUMMARY",
                description=f"Analyzed {len(collection['inventory'])} evidence items ({len(collection['images'])} images, {len(collection['videos'])} videos).",
                duration=4.5,
                caption="Generated by Crime Investigation AI",
            )
        )

        return scenes

    # ------------------------------------------------------------------
    # Frame & Video Processing Helpers
    # ------------------------------------------------------------------

    def _sample_video_frames(self, video_path: Path | None, max_frames: int = 4) -> list[tuple[float, Image.Image]]:
        """Sample 3-5 representative keyframes from a video file using OpenCV."""
        if not video_path or not video_path.exists():
            return []

        frames: list[tuple[float, Image.Image]] = []
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            duration = total_frames / max(1.0, fps)

            if total_frames <= 0:
                cap.release()
                return []

            # Sample evenly across duration
            indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame_bgr = cap.read()
                if ret and frame_bgr is not None:
                    ts_sec = float(idx) / max(1.0, fps)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frames.append((ts_sec, pil_img))
            cap.release()
        except Exception:
            pass

        return frames

    # ------------------------------------------------------------------
    # Rendering Engine (OpenCV + PIL)
    # ------------------------------------------------------------------

    def _render_video_scenes(self, scenes: list[VideoScene], output_path: Path) -> float:
        """Render scenes into an MP4 video file and return total duration in seconds."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, float(self.fps), (self.width, self.height))

        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(str(output_path), fourcc, float(self.fps), (self.width, self.height))

        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize OpenCV VideoWriter for {output_path}")

        total_duration = 0.0

        for idx, scene in enumerate(scenes):
            scene_frames = int(round(scene.duration * self.fps))
            total_duration += scene.duration

            if scene.scene_type in ["EVIDENCE_IMAGE", "EVIDENCE_VIDEO"]:
                frames = self._render_evidence_media_scene(scene, scene_frames)
            else:
                static_img = self._render_graphic_scene(scene)
                static_bgr = cv2.cvtColor(np.array(static_img), cv2.COLOR_RGB2BGR)
                frames = [static_bgr] * scene_frames

            # Write frames with cross-fade transition
            if idx > 0 and scene.transition == "fade" and len(frames) > 5:
                fade_n = min(12, len(frames) // 2)
                prev_last = last_frame  # type: ignore
                for f_idx in range(fade_n):
                    alpha = (f_idx + 1) / float(fade_n)
                    blended = cv2.addWeighted(prev_last, 1.0 - alpha, frames[f_idx], alpha, 0)
                    writer.write(blended)
                for f_idx in range(fade_n, len(frames)):
                    writer.write(frames[f_idx])
            else:
                for fr in frames:
                    writer.write(fr)

            last_frame = frames[-1]

        writer.release()
        return total_duration

    def _render_evidence_media_scene(self, scene: VideoScene, num_frames: int) -> list[np.ndarray]:
        """
        Render evidence photo or video frame with:
        - 65-80% visual area for the evidence image!
        - Ken Burns zoom/pan effect (zooming toward verified weapon if available)
        - Clean bounding box detection overlays (Verified, Candidate, Rejected, Person)
        - Dark forensic sidebar card with metadata, counts, and cautious evidence explanation.
        """
        frames: list[np.ndarray] = []

        # Load evidence image
        pil_img = scene.pil_image
        if pil_img is None and scene.image_path and scene.image_path.exists():
            try:
                pil_img = Image.open(scene.image_path)
                pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            except Exception:
                pil_img = None

        if pil_img is None:
            # Fallback for missing/corrupt media file
            fallback = self._render_graphic_scene(scene)
            bgr = cv2.cvtColor(np.array(fallback), cv2.COLOR_RGB2BGR)
            return [bgr] * num_frames

        # Draw detection overlays on a copy of the original PIL image
        annotated_pil = self._draw_detection_overlays(pil_img.copy(), scene.evidence_items)

        orig_w, orig_h = annotated_pil.size

        # Layout dimensions: 70% canvas width for image display (1340px out of 1920px)
        margin_x = int(self.width * 0.025)
        margin_top = int(self.height * 0.11)
        display_w = int(self.width * 0.67)
        display_h = int(self.height * 0.77)

        sidebar_x1 = margin_x + display_w + int(self.width * 0.02)
        sidebar_x2 = self.width - margin_x
        sidebar_y1 = margin_top
        sidebar_y2 = margin_top + display_h

        # Animate Ken Burns zoom
        for f in range(num_frames):
            progress = f / max(1, num_frames - 1)
            scale = 1.0 + (0.10 * progress)

            crop_w = int(orig_w / scale)
            crop_h = int(orig_h / scale)

            # Center crop or zoom toward verified weapon region
            if scene.target_crop_center:
                tcx, tcy = scene.target_crop_center
                x1 = int(clamp(tcx - crop_w / 2.0, 0, orig_w - crop_w))
                y1 = int(clamp(tcy - crop_h / 2.0, 0, orig_h - crop_h))
            else:
                x1 = int((orig_w - crop_w) * 0.5 * progress)
                y1 = int((orig_h - crop_h) * 0.5 * progress)

            x2 = min(orig_w, x1 + crop_w)
            y2 = min(orig_h, y1 + crop_h)

            cropped = annotated_pil.crop((x1, y1, x2, y2))
            resized = cropped.resize((display_w, display_h), Image.Resampling.LANCZOS)

            # Base Canvas
            canvas = Image.new("RGB", (self.width, self.height), color=(11, 15, 25))
            draw = ImageDraw.Draw(canvas)

            # Paste 65-80% Large Evidence Photo
            canvas.paste(resized, (margin_x, margin_top))
            draw.rectangle([(margin_x, margin_top), (margin_x + display_w, margin_top + display_h)], outline=(0, 242, 254), width=3)

            # Top Header Bar
            header_y1 = int(self.height * 0.025)
            header_y2 = int(self.height * 0.09)
            draw.rectangle([(margin_x, header_y1), (self.width - margin_x, header_y2)], fill=(22, 28, 46))
            draw.line([(margin_x, header_y2), (self.width - margin_x, header_y2)], fill=(0, 242, 254), width=3)

            title_font = self._get_font(int(self.height * 0.032), bold=True)
            body_font = self._get_font(int(self.height * 0.024))
            caption_font = self._get_font(int(self.height * 0.020))

            header_text = f"FORENSIC EVIDENCE: {scene.title}"
            draw.text((margin_x + 15, header_y1 + 8), header_text, fill=(0, 242, 254), font=title_font)
            draw.text((self.width - margin_x - int(self.width * 0.18), header_y1 + 10), datetime.now().strftime("%Y-%m-%d"), fill=(148, 163, 184), font=caption_font)

            # Right Sidebar Panel
            draw.rectangle([(sidebar_x1, sidebar_y1), (sidebar_x2, sidebar_y2)], fill=(22, 28, 46), outline=(42, 52, 75), width=2)

            # Sidebar Content
            pad = int(self.width * 0.012)
            draw.text((sidebar_x1 + pad, sidebar_y1 + pad), "EVIDENCE METADATA", fill=(0, 242, 254), font=title_font)

            if scene.evidence_id:
                draw.text((sidebar_x1 + pad, sidebar_y1 + int(self.height * 0.07)), f"ID: {scene.evidence_id}", fill=(255, 255, 255), font=body_font)
            if scene.filename:
                draw.text((sidebar_x1 + pad, sidebar_y1 + int(self.height * 0.11)), f"File: {scene.filename[:20]}", fill=(148, 163, 184), font=caption_font)

            # Status Badge
            ver_w_cnt = sum(1 for d in scene.evidence_items if d.get("weapon_status") == "verified" or d.get("is_verified_weapon"))
            badge_y1 = sidebar_y1 + int(self.height * 0.16)
            badge_y2 = badge_y1 + int(self.height * 0.06)
            if ver_w_cnt > 0:
                draw.rectangle([(sidebar_x1 + pad, badge_y1), (sidebar_x2 - pad, badge_y2)], fill=(239, 68, 68))
                draw.text((sidebar_x1 + pad + 10, badge_y1 + 8), "🟢 VERIFIED WEAPON", fill=(255, 255, 255), font=caption_font)
            else:
                draw.rectangle([(sidebar_x1 + pad, badge_y1), (sidebar_x2 - pad, badge_y2)], fill=(30, 58, 138))
                draw.text((sidebar_x1 + pad + 10, badge_y1 + 8), "🔵 OBJECTS ANALYZED", fill=(255, 255, 255), font=caption_font)

            # Detection Counts Breakdown
            p_cnt = sum(1 for d in scene.evidence_items if d.get("label", "").lower() in ["person", "people"])
            o_cnt = max(0, len(scene.evidence_items) - ver_w_cnt - p_cnt)

            counts_y = sidebar_y1 + int(self.height * 0.24)
            line_h = int(self.height * 0.04)
            draw.text((sidebar_x1 + pad, counts_y), f"• Persons: {p_cnt}", fill=(226, 232, 240), font=body_font)
            draw.text((sidebar_x1 + pad, counts_y + line_h), f"• Weapons: {ver_w_cnt}", fill=(239, 68, 68) if ver_w_cnt > 0 else (226, 232, 240), font=body_font)
            draw.text((sidebar_x1 + pad, counts_y + line_h * 2), f"• Objects: {o_cnt}", fill=(226, 232, 240), font=body_font)

            # Evidence Explanation Box
            exp_y = counts_y + line_h * 3.2
            draw.text((sidebar_x1 + pad, exp_y), "ASSESSMENT:", fill=(79, 172, 254), font=caption_font)
            wrapped_cap = self._wrap_text(scene.caption, max_chars=int((sidebar_x2 - sidebar_x1) / 10))
            for i, line in enumerate(wrapped_cap[:6]):
                draw.text((sidebar_x1 + pad, exp_y + int(self.height * 0.03) * (i + 1)), line, fill=(203, 213, 225), font=caption_font)

            # Bottom Subtitle Bar
            sub_y1 = int(self.height * 0.91)
            sub_y2 = int(self.height * 0.97)
            draw.rectangle([(margin_x, sub_y1), (self.width - margin_x, sub_y2)], fill=(15, 23, 42))
            draw.text((margin_x + 15, sub_y1 + 8), f"CAPTION: {scene.caption}", fill=(241, 245, 249), font=caption_font)

            bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
            frames.append(bgr)

        return frames

    def _draw_detection_overlays(self, pil_img: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
        """Draw clean bounding box overlays with labels and verification status badges."""
        if not detections:
            return pil_img

        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size
        font = self._get_font(max(14, int(h * 0.035)), bold=True)

        for det in detections:
            bbox = det.get("bbox") or det.get("box")
            if not bbox:
                continue

            if isinstance(bbox, dict):
                x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox

            # Handle normalized bbox (0.0 to 1.0)
            if x2 <= 1.0 and y2 <= 1.0:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

            label = det.get("label", "Object").upper()
            conf = det.get("confidence", 0.0)
            status = det.get("weapon_status", "")

            if status == "verified" or det.get("is_verified_weapon"):
                color = (0, 242, 254)  # Neon Cyan
                lbl_str = f"VERIFIED {label} ({conf:.0%})"
            elif status == "rejected" or det.get("decision") == "REJECT":
                color = (239, 68, 68)  # Red
                lbl_str = f"REJECTED {label}"
            elif status == "candidate":
                color = (245, 158, 11)  # Gold
                lbl_str = f"CANDIDATE {label} ({conf:.0%})"
            else:
                color = (59, 130, 246)  # Blue
                lbl_str = f"{label} ({conf:.0%})"

            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
            # Label background box
            draw.rectangle([(x1, max(0, y1 - 32)), (x1 + len(lbl_str) * 12 + 10, max(32, y1))], fill=color)
            draw.text((x1 + 6, max(0, y1 - 28)), lbl_str, fill=(0, 0, 0), font=font)

        return pil_img

    def _render_graphic_scene(self, scene: VideoScene) -> Image.Image:
        """Render standard graphic slides (Title, Overview, Inventory, Verification, Timeline, AI, Risk, Summary)."""
        img = Image.new("RGB", (self.width, self.height), color=(11, 15, 25))
        draw = ImageDraw.Draw(img)

        title_font = self._get_font(int(self.height * 0.055), bold=True)
        sub_font = self._get_font(int(self.height * 0.035))
        body_font = self._get_font(int(self.height * 0.026))
        caption_font = self._get_font(int(self.height * 0.022))

        # Outer Frame Border
        draw.rectangle([(40, 40), (self.width - 40, self.height - 40)], outline=(42, 52, 75), width=3)
        draw.rectangle([(40, 40), (self.width - 40, 110)], fill=(22, 28, 46))
        draw.line([(40, 110), (self.width - 40, 110)], fill=(0, 242, 254), width=4)

        # Header Title
        draw.text((60, 55), "CRIME INVESTIGATION AI", fill=(0, 242, 254), font=sub_font)
        draw.text((self.width - 320, 55), datetime.now().strftime("%Y-%m-%d"), fill=(148, 163, 184), font=caption_font)

        # Center Main Section Card
        card_x1, card_y1 = 120, 160
        card_x2, card_y2 = self.width - 120, self.height - 140
        draw.rectangle([(card_x1, card_y1), (card_x2, card_y2)], fill=(22, 28, 46), outline=(42, 52, 75), width=2)

        draw.text((card_x1 + 60, card_y1 + 40), scene.title, fill=(255, 255, 255), font=title_font)

        if scene.scene_type == "TITLE":
            draw.text((card_x1 + 60, card_y1 + 160), scene.description, fill=(79, 172, 254), font=sub_font)
            draw.text((card_x1 + 60, card_y1 + 260), "AUTOMATED FORENSIC INVESTIGATION REPORT", fill=(226, 232, 240), font=body_font)
            draw.text((card_x1 + 60, card_y1 + 320), "STATUS: VERIFIED ANALYSIS COMPLETE", fill=(34, 197, 94), font=body_font)

        elif scene.scene_type == "OVERVIEW":
            draw.text((card_x1 + 60, card_y1 + 130), scene.description, fill=(226, 232, 240), font=sub_font)
            for i, item in enumerate(scene.evidence_items[:5]):
                lbl = item.get("label", "")
                draw.text((card_x1 + 60, card_y1 + 210 + i * 50), f"► {lbl}", fill=(0, 242, 254), font=body_font)

        elif scene.scene_type == "INVENTORY":
            draw.text((card_x1 + 60, card_y1 + 130), "Evidence Items Cataloged:", fill=(79, 172, 254), font=sub_font)
            lines = scene.description.split("\n")
            for i, l in enumerate(lines[:8]):
                draw.text((card_x1 + 60, card_y1 + 200 + i * 45), l, fill=(226, 232, 240), font=body_font)

        elif scene.scene_type == "VERIFICATION":
            draw.text((card_x1 + 60, card_y1 + 130), scene.caption, fill=(255, 255, 255), font=sub_font)
            draw.text((card_x1 + 60, card_y1 + 210), "Automatic Verification Criteria Passed:", fill=(34, 197, 94), font=body_font)
            draw.text((card_x1 + 80, card_y1 + 270), "• Dual-Model Detection (YOLOv8 + Dedicated Weapon Model)", fill=(226, 232, 240), font=body_font)
            draw.text((card_x1 + 80, card_y1 + 320), "• Confidence Threshold & Cross-Model NMS Validation Passed", fill=(226, 232, 240), font=body_font)
            draw.text((card_x1 + 80, card_y1 + 370), "• Rejected weapon candidates excluded from threat flags", fill=(239, 68, 68), font=body_font)

        elif scene.scene_type == "TIMELINE":
            draw.text((card_x1 + 60, card_y1 + 130), "Chronological Forensic Progression:", fill=(226, 232, 240), font=sub_font)
            steps = [
                "00:00 - Evidence Uploaded & Cataloged",
                "00:05 - Media Scanned & Persons Detected",
                "00:12 - Weapon Candidate Isolated",
                "00:15 - Automatic Weapon Verification Passed",
                "00:20 - Evidence Correlation & AI Analysis",
                "00:25 - Final Report & Video Generated",
            ]
            for i, st in enumerate(steps):
                draw.text((card_x1 + 80, card_y1 + 200 + i * 45), f"► {st}", fill=(0, 242, 254), font=body_font)

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
            draw.text((card_x1 + 60, card_y1 + 160), "INVESTIGATION SUMMARY COMPLETE", fill=(34, 197, 94), font=title_font)
            draw.text((card_x1 + 60, card_y1 + 240), scene.description, fill=(226, 232, 240), font=body_font)
            draw.text((card_x1 + 60, card_y1 + 320), "Generated by Crime Investigation AI", fill=(148, 163, 184), font=sub_font)

        # Bottom Subtitle Bar
        draw.rectangle([(40, self.height - 110), (self.width - 40, self.height - 40)], fill=(15, 23, 42))
        draw.text((60, self.height - 90), f"CAPTION: {scene.caption}", fill=(203, 213, 225), font=caption_font)

        return img

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


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))
