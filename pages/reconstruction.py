"""
Crime Scene Reconstruction page.

Reads the last `EvidenceAnalysis` and (if available) the last
annotated image / keyframe, plans a 4-6 scene storyboard, renders
the captioned images, and produces a short reconstruction MP4.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import STORYBOARD_DIR, VIDEOS_DIR
from core.icons import ICON_RECONSTRUCTION
from models.reconstruction import (
    render_storyboard_images,
    synthesize_reconstruction_video,
)
from models.scene_planner import plan_scenes
from models.schemas import (
    EvidenceAnalysis,
    Storyboard,
)
from pages._layout import empty_state, friendly_error, render_page_header
from pages._state import resolve_active_analysis, resolve_batch_or_detection_image
from utils.db_hooks import auto_save_last_storyboard


def _get_base_image():
    """Pick the best available annotated image from session state.

    Resolution order:
    1. Latest batch result (multi-file Crime Scene Investigation).
    2. `last_video` or `last_detection` from the single-file path.
    """
    batch_image = resolve_batch_or_detection_image()
    if batch_image is not None:
        return batch_image

    last_video = st.session_state.get("last_video")
    if last_video is not None and last_video.keyframes:
        for kf in last_video.keyframes:
            if kf.detection and kf.detection.annotated_image is not None:
                return kf.detection.annotated_image

    last_detection = st.session_state.get("last_detection")
    if last_detection is not None and last_detection.annotated_image is not None:
        return last_detection.annotated_image

    return None


def render() -> None:
    render_page_header(
        ICON_RECONSTRUCTION,
        subtitle="Generate a 4-6 scene storyboard and a short reconstruction video from the evidence.",
    )

    analysis: EvidenceAnalysis | None = resolve_active_analysis()
    if analysis is None:
        empty_state(
            "No evidence analysis found.",
            "Process evidence in the Investigation workspace first to generate a reconstruction storyboard and video.",
            action_label="Open Investigation Workspace",
            action_target="pages/investigation.py",
        )
        return

    # Check if the currently held storyboard is outdated in the DB
    db_id = st.session_state.get("last_storyboard_db_id")
    if db_id is not None:
        from config import DATABASE_PATH
        from database.repository import load_storyboard
        sb_data = load_storyboard(DATABASE_PATH, db_id)
        if sb_data and sb_data.get("is_outdated"):
            st.warning("⚠️ This reconstruction is outdated. Evidence or human reviews have changed since it was generated. Please regenerate.")

    if st.session_state.get("reconstruction_auto_trigger"):
        st.success("✨ This reconstruction was automatically generated from the AI Narrative Summary.")
        st.session_state["reconstruction_auto_trigger"] = False

    base_image = _get_base_image()
    if base_image is None:
        st.info(
            "No annotated image found. The storyboard will use clean "
            "placeholder panels instead of the actual evidence frame."
        )
    else:
        st.success("Found annotated evidence frame. It will be reused as the visual base.")

    if st.button("Build Storyboard & Video", type="primary"):
        with st.spinner("Planning scenes, rendering images, and synthesizing video..."):
            try:
                storyboard: Storyboard = plan_scenes(
                    analysis=analysis,
                    base_image=base_image,
                )

                safe_stem = "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in analysis.source_name.rsplit(".", 1)[0]
                )
                sb_dir = STORYBOARD_DIR / f"{safe_stem}_{storyboard.timestamp.strftime('%Y%m%d_%H%M%S')}"
                image_paths = render_storyboard_images(storyboard, sb_dir)
                video_path = synthesize_reconstruction_video(
                    storyboard=storyboard,
                    output_path=VIDEOS_DIR / f"reconstruction_{safe_stem}.mp4",
                )

                st.session_state["last_storyboard"] = storyboard
                st.session_state["last_storyboard_image_paths"] = image_paths
                st.session_state["last_reconstruction_video_path"] = str(video_path)
                auto_save_last_storyboard()
                st.success(
                    f"Generated {storyboard.scene_count} scenes and "
                    f"a {storyboard.total_duration_sec:.1f}s video."
                )
            except Exception as exc:
                friendly_error(exc, fallback_title="Reconstruction failed.")
                return

    storyboard: Storyboard | None = st.session_state.get("last_storyboard")
    if storyboard is None:
        return

    _render_storyboard(storyboard)


def _render_storyboard(sb: Storyboard) -> None:
    st.subheader(f"Storyboard ({sb.scene_count} scenes, {sb.total_duration_sec:.1f}s)")
    cols = st.columns(min(3, sb.scene_count) or 1)
    for scene in sb.scenes:
        col = cols[(scene.index) % len(cols)]
        with col:
            if scene.image is not None:
                st.image(
                    scene.image,
                    caption=f"#{scene.index + 1} {scene.title}",
                    use_container_width=True,
                )

    with st.expander("Scene captions"):
        for scene in sb.scenes:
            st.markdown(f"**#{scene.index + 1} {scene.title}**")
            st.markdown(f"> {scene.caption}")
            st.caption(
                f"Duration: {scene.duration_sec:.1f}s · "
                f"Based on real frame: {'yes' if scene.based_on_real_frame else 'no'}"
            )

    st.markdown(
        "> **AI-generated probable reconstruction.** This visualization is generated "
        "from available evidence and should not be treated as a confirmed factual "
        "reconstruction."
    )

    st.subheader("Reconstruction video")
    video_path_str: str | None = st.session_state.get("last_reconstruction_video_path")
    if video_path_str and Path(video_path_str).exists():
        # Read once and reuse - avoids two disk hits for the same MP4.
        video_bytes = Path(video_path_str).read_bytes()
        st.video(video_bytes)
        st.download_button(
            label="Download reconstruction video (MP4)",
            data=video_bytes,
            file_name=Path(video_path_str).name,
            mime="video/mp4",
        )
    else:
        st.warning("Video file not found on disk.")

    image_paths: list[Path] = st.session_state.get("last_storyboard_image_paths", [])
    if image_paths:
        st.subheader("Download storyboard images")
        for p in image_paths:
            # Read once per image. The download_button widget reads from
            # `data` lazily so each button gets its own bytes object.
            st.download_button(
                label=p.name,
                data=p.read_bytes(),
                file_name=p.name,
                mime="image/jpeg",
                key=f"sb_dl_{p.name}",
            )
