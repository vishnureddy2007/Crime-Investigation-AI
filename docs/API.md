# API Reference

A scannable index of every public symbol in the project. One section per
module. No prose — just signatures, types, and notes.

> **Conventions:**
> - `Path` = `pathlib.Path`
> - `Image` = `PIL.Image.Image`
> - `ndarray` = `numpy.ndarray`
> - All `auto_save_*` functions are Streamlit-aware but never raise.

---

## `config.settings`

Centralized constants. All other modules read from here — never hardcode.

| Symbol | Type | Purpose |
|---|---|---|
| `PROJECT_ROOT` | `Path` | Repo root, computed from `__file__`. |
| `DATA_DIR`, `OUTPUTS_DIR`, `IMAGES_DIR`, `MODELS_DIR`, `DATABASE_DIR`, `DATABASE_PATH`, `DOCS_DIR`, `TESTS_DIR` | `Path` | Standard folders. |
| `REPORTS_DIR`, `STORYBOARD_DIR`, `VIDEOS_DIR`, `DETECTION_DIR`, `KEYFRAMES_DIR` | `Path` | Artefact subfolders of `OUTPUTS_DIR`. |
| `YOLO_MODEL_NAME` | `str` | Default `"yolov8n.pt"`. |
| `YOLO_CONFIDENCE_THRESHOLD` | `float` | Default `0.4`. |
| `YOLO_IOU_THRESHOLD` | `float` | Default `0.45`. |
| `YOLO_CLASS_MAPPING` | `dict[str, str]` | COCO name -> crime label (e.g. `"knife" -> "knife"`, `"car" -> "vehicle"`). |
| `RELEVANT_CLASSES` | `set[str]` | Derived from the mapping keys. |
| `FRAME_EXTRACTION_INTERVAL_SEC` | `float` | Default `1.0`. |
| `MAX_VIDEO_DURATION_SEC` | `int` | Default `30`. |
| `MAX_VIDEO_SIZE_MB` | `int` | Default `100`. |
| `KEYFRAME_COUNT` | `int` | Default `5`. |
| `SEVERITY_WEIGHTS` | `dict[str, float]` | Per-label contributions to the 0-100 severity score. |
| `CATEGORY_RULES` | `list[tuple[str, set[str]]]` | Ordered; first match wins. |
| `SEVERITY_THRESHOLDS` | `list[tuple[int, str]]` | `(threshold, level)` pairs, highest threshold first. |
| `SEVERITY_LEVEL_COLORS` | `dict[str, str]` | Hex colors per level. |
| `SUMMARY_MODEL_NAME` | `str` | Default `"google/flan-t5-base"`. |
| `SUMMARY_MAX_LENGTH`, `SUMMARY_MIN_LENGTH`, `SUMMARY_MAX_NEW_TOKENS`, `SUMMARY_GENERATION_TIMEOUT_SEC` | `int` / `float` | LLM generation knobs. |
| `REPORT_TITLE`, `REPORT_AUTHOR`, `REPORT_REMARKS` | `str` | Report metadata. |
| `STORYBOARD_MIN_SCENES`, `STORYBOARD_MAX_SCENES` | `int` | Default `4`, `6`. |
| `STORYBOARD_DEFAULT_DURATION_SEC` | `float` | Default `1.5`. |
| `STORYBOARD_FADE_SEC` | `float` | Default `0.5`. |
| `STORYBOARD_VIDEO_FPS` | `int` | Default `24`. |
| `STORYBOARD_VIDEO_CODEC` | `str` | Default `"mp4v"`. |
| `STORYBOARD_PANEL_SIZE` | `tuple[int, int]` | Default `(640, 360)`. |
| `APP_NAME`, `APP_VERSION` | `str` | Default `"AI Crime Investigation Assistant"`, `"0.11.0"`. |

---

## `models.schemas`

Plain `@dataclass`es — the contract between layers.

| Symbol | Type | Purpose |
|---|---|---|
| `BoundingBox(x1, y1, x2, y2)` | `@dataclass(frozen=True)` | Pixel rectangle. Properties: `width`, `height`, `area`, `as_dict()`. |
| `Detection(class_name, label, confidence, bbox)` | `@dataclass(frozen=True)` | One detected object. `as_dict()`. |
| `DetectionResult` | `@dataclass` | Wrapper around a list of `Detection`. Properties: `count`, `classes()`, `labels()`, `counts_by_label()`, `average_confidence()`. |
| `FrameResult` | `@dataclass` | One extracted video frame + its detection. Properties: `detection_count`, `average_confidence`, `evidence_score`. |
| `VideoAnalysisResult` | `@dataclass` | All `FrameResult`s + metadata + keyframe indices. Properties: `frame_count`, `keyframes`, `total_detections()`, `aggregate_counts_by_label()`. |
| `AnalysisInput` | `@dataclass` | Adapter for either image or video. Class methods `from_image(result)` / `from_video(result)`. |
| `EvidenceAnalysis` | `@dataclass` | Severity, category, observations, counts. `as_dict()`. |
| `InvestigationSummary` | `@dataclass` | `template_text`, `ai_text`, `used_ai`, `model_name`, `generation_time_sec`, `evidence_snapshot`. Property: `primary_text`. |
| `ReportData` | `@dataclass` | Everything a report renderer needs. `as_dict()`. |
| `StoryboardScene(index, title, caption, image, duration_sec, based_on_real_frame)` | `@dataclass` | One captioned panel. |
| `Storyboard` | `@dataclass` | Ordered `list[StoryboardScene]`. Properties: `scene_count`, `total_duration_sec`, `as_dict()`. |

---

## `models.yolo_detector`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `YOLODetector(model_name, confidence, iou)` | constructor | instance | `confidence` and `iou` default to config values. |
| `YOLODetector.model` | `@property` | `YOLO` | Lazy-loaded; first call downloads weights. |
| `YOLODetector.detect_image(image, source_name, confidence, iou)` | method | `DetectionResult` | Runs YOLO, filters to `RELEVANT_CLASSES`, returns annotated image. |
| `get_detector(model_name, confidence, iou)` | function | `YOLODetector` | Convenience factory. |
| `save_annotated_image(result, output_dir, file_stem)` | function | `Path` | Writes `<stem>_annotated.jpg`. |
| `result_to_bytes(result)` | function | `bytes` | JPEG-encoded bytes for `st.download_button`. |

---

## `models.video_processor`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `VideoProcessor(detector, keyframe_count, frame_interval_sec)` | constructor | instance | Duck-typed detector. |
| `VideoProcessor.process(video_path, source_name, save_frames)` | method | `VideoAnalysisResult` | Raises `FileNotFoundError` / `RuntimeError` on bad input. |
| `extract_keyframes` | (removed) | — | Now part of `VideoProcessor.process`. |

---

## `models.evidence_analyzer`

Pure functions, no I/O.

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `analyze(input_data)` | `AnalysisInput -> EvidenceAnalysis` | populated dataclass | Never raises on empty input. |
| `_compute_severity(...)` | private | `int` | 0-100, clamped. |
| `_severity_level(score)` | private | `str` | First threshold match in `SEVERITY_THRESHOLDS`. |
| `_suggest_category(present_labels)` | private | `str` | First rule match in `CATEGORY_RULES`. |
| `_build_observations(...)` | private | `list[str]` | Short factual statements. |

---

## `models.summary_generator`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `build_template_summary(analysis)` | `EvidenceAnalysis -> str` | deterministic text | Never fails. Used as the LLM prompt. |
| `SummaryGenerator(model_name, max_new_tokens)` | constructor | instance | `model_name=None` disables LLM. |
| `SummaryGenerator.pipeline` | `@property` | HF pipeline | Lazy-loaded. |
| `SummaryGenerator.is_available()` | method | `bool` | False if no model name. |
| `SummaryGenerator.generate(analysis)` | method | `InvestigationSummary` | Falls back to template on LLM failure. |

---

## `models.report_generator`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `build_report_data(analysis, summary, report_id, title, remarks)` | function | `ReportData` | Optional `summary`; otherwise uses a fresh template summary. |
| `PDFReportGenerator()` | constructor | instance | ReportLab under the hood. |
| `PDFReportGenerator.generate(data)` | method | `bytes` | A4 PDF, 2cm margins, ~6 sections. |
| `DOCXReportGenerator()` | constructor | instance | python-docx under the hood. |
| `DOCXReportGenerator.generate(data)` | method | `bytes` | DOCX with same 6 sections. |

---

## `models.scene_planner`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `plan_scenes(analysis, base_image, min_scenes, max_scenes, default_duration)` | function | `Storyboard` | Always returns `4-6` scenes. |
| `_panel_placeholder(title, caption)` | private | `Image` | Fallback when no `base_image`. |
| `_wrap_text(text, max_chars)` | private | `list[str]` | Greedy word wrap. |
| `_tint_image(img, tint)` | private | `Image` | RGB blend. |
| `_resize_to_panel(img)` | private | `Image` | Letterboxed to `STORYBOARD_PANEL_SIZE`. |

---

## `models.reconstruction`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `draw_caption_on_image(image, title, caption, bar_ratio)` | function | `Image` | Bottom semi-transparent bar. |
| `render_storyboard_images(storyboard, output_dir)` | function | `list[Path]` | Saves `scene_NN.jpg` per scene. |
| `synthesize_reconstruction_video(storyboard, output_path, fps, fade_sec, codec)` | function | `Path` | OpenCV `VideoWriter` with `mp4v`. |

---

## `models.dashboard`

Pure aggregations for the Dashboard page. Streamlit-free.

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `compute_severity_band(score)` | `int -> str` | level | Walks `SEVERITY_THRESHOLDS`. |
| `severity_color(level)` | `str -> str` | hex | Falls back to `#888888`. |
| `severity_band_progress(score)` | `int -> tuple[int, int, str]` | `(current, 100, label)` | For `st.progress`. |
| `compute_kpis(analysis)` | `EvidenceAnalysis -> dict` | flat KPI dict | Adds `severity_color`. |
| `label_counts_series(analysis)` | `EvidenceAnalysis -> list[dict]` | `[{label, count}, ...]` | Sorted desc, no zeros. |
| `summarize_session_state(snapshot)` | `dict|None -> dict` | `{key: bool, pipeline_complete: bool}` | Reads `st.session_state` if snapshot is None. |

---

## `utils.image_io`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `ALLOWED_IMAGE_EXTS` | `set[str]` | — | `{.jpg, .jpeg, .png, .bmp, .webp}` |
| `ImageLoadError` | `Exception` | — | Raised on decode failures. |
| `load_image_from_bytes(data, source_name)` | function | `Image` | Converts to RGB. |
| `load_image_from_path(path)` | `Path|str -> Image` | `Image` | Validates extension + existence. |
| `is_allowed_image(filename)` | `str -> bool` | `bool` | Extension check. |

---

## `utils.video_io`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `ALLOWED_VIDEO_EXTS` | `set[str]` | — | `{.mp4, .avi, .mov, .mkv, .webm}` |
| `VideoLoadError` | `Exception` | — | Raised on size/decode failures. |
| `is_allowed_video(filename)` | function | `bool` | Extension check. |
| `validate_video_bytes(data, filename)` | function | `None` | Raises on too-large or undecodable. |
| `get_video_metadata(path)` | function | `dict` | fps, frame_count, w, h, duration_sec, size_mb. |
| `save_uploaded_video(data, output_dir, filename)` | function | `Path` | Writes to `<output_dir>/`. |

---

## `utils.frame_utils`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `cv2_to_pil(frame)` | `ndarray -> Image` | RGB `Image` | Raises `ValueError` on None. |
| `pil_to_cv2(image)` | `Image -> ndarray` | BGR `ndarray` | Always re-encodes to RGB first. |

---

## `utils.visualization`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `_color_for(label)` | `str -> str` | hex | Falls back to `#264653`. |
| `_font(size)` | `int -> ImageFont` | font | Tries `arial.ttf`, falls back to PIL default. |
| `draw_detections(image, detections, box_width)` | `Image, Iterable[Detection], int -> Image` | annotated `Image` | Returns a copy. |

---

## `utils.db_hooks`

Streamlit-aware wrappers. **None of these raise.**

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `auto_save_last_analysis(db_path)` | function | `int | None` | Reads `last_analysis` from `st.session_state`. |
| `auto_save_last_summary(db_path)` | function | `int | None` | Reads `last_summary`. |
| `auto_save_last_report(db_path)` | function | `int | None` | Reads `last_report_data`. |
| `auto_save_last_storyboard(db_path)` | function | `int | None` | Reads `last_storyboard`. |
| `_resolve_case_id(db_path, source_name, source_type)` | private | `int` | Reuses the latest case for `source_name`, else inserts new. |

All four `auto_save_*` functions:

- Return early with `None` if the corresponding `last_*` key is unset.
- Reuse the existing row id if the corresponding `last_*_db_id` is set
  (idempotent on Streamlit reruns).
- Surface a `st.warning` on failure (swallowed).

---

## `database.db`

| Symbol | Signature | Returns | Notes |
|---|---|---|---|
| `SCHEMA_SQL` | `str` | — | 5 tables, all FKs `ON DELETE CASCADE`. |
| `get_connection(path)` | `@contextmanager` | `Iterator[sqlite3.Connection]` | Sets `row_factory=Row`, `PRAGMA foreign_keys=ON`, commits on success. |
| `init_db(path)` | function | `None` | Idempotent; creates parent dirs. |
| `_row_to_dict(row)` | private | `dict | None` | `sqlite3.Row -> dict`. |

---

## `database.repository`

All functions take `db_path: Path` first. Pure CRUD; no Streamlit import.

### `cases`

| Symbol | Signature | Returns |
|---|---|---|
| `save_case(db_path, source_name, source_type)` | `-> int` | new `case_id` |
| `list_cases(db_path)` | `-> list[dict]` | all cases + latest severity |
| `latest_case_for_source(db_path, source_name)` | `-> int | None` | most recent case_id |
| `case_exists(db_path, case_id)` | `-> bool` | existence check |

### `analyses`

| Symbol | Signature | Returns |
|---|---|---|
| `save_analysis(db_path, case_id, analysis)` | `-> int` | new `analysis_id` |
| `load_analysis(db_path, analysis_id)` | `-> dict | None` | full payload |
| `list_analyses_for_case(db_path, case_id)` | `-> list[dict]` | metadata rows |

### `summaries`

| Symbol | Signature | Returns |
|---|---|---|
| `save_summary(db_path, case_id, summary)` | `-> int` | new `summary_id` |
| `load_summary(db_path, summary_id)` | `-> dict | None` | full payload |
| `list_summaries_for_case(db_path, case_id)` | `-> list[dict]` | metadata rows |

### `reports`

| Symbol | Signature | Returns |
|---|---|---|
| `save_report(db_path, case_id, report)` | `-> int` | new `report_id` |
| `load_report(db_path, report_id)` | `-> dict | None` | full payload |
| `list_reports_for_case(db_path, case_id)` | `-> list[dict]` | metadata rows |

### `storyboards`

| Symbol | Signature | Returns |
|---|---|---|
| `save_storyboard(db_path, case_id, storyboard)` | `-> int` | new `storyboard_id` |
| `load_storyboard(db_path, storyboard_id)` | `-> dict | None` | full payload |
| `list_storyboards_for_case(db_path, case_id)` | `-> list[dict]` | metadata rows |

> All `load_*` functions return the JSON payload as a plain dict. All
> `list_*_for_case` functions return metadata-only rows (no `payload_json`)
> for fast previews.

---

## Session-state keys

The pages communicate through `st.session_state`:

| Key | Type | Producer page | Consumer pages |
|---|---|---|---|
| `last_detection` | `DetectionResult` | Image Detection | Evidence Analysis, AI Summary, Dashboard |
| `last_video` | `VideoAnalysisResult` | Video Processing | Evidence Analysis, AI Summary, Dashboard |
| `last_analysis` | `EvidenceAnalysis` | Evidence Analysis | AI Summary, Report, Reconstruction, Dashboard |
| `last_summary` | `InvestigationSummary` | AI Summary | Report, Dashboard |
| `last_report_data` | `ReportData` | Report | Dashboard |
| `last_storyboard` | `Storyboard` | Reconstruction | Dashboard |
| `last_reconstruction_video_path` | `Path` | Reconstruction | Dashboard |
| `last_*_db_id` | `int` | auto-save hooks | (idempotency) |
| `reloaded_*` | various | Case History | page that produced the original `last_*` |

---

## See also

- [Installation](INSTALLATION.md)
- [Developer Guide](DEVELOPER.md)
- [Progress](PROGRESS.md)

---

## HTTP API (`app/`)

Phase 15 expands the dependency-free HTTP façade with four new
read-only endpoints. All endpoints return a JSON-safe envelope
(`{"ok": true, "status": 200, "data": ...}`). Use
`from app import dispatch`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe — `{"status": "ok"}`. |
| GET | `/health/deep` | Adds Python version, platform, PID, observability counters. |
| GET | `/routes` | Lists every registered route. |
| GET | `/cases` | Lists every persisted case (latest 50, joined with severity). |
| GET | `/analytics` | Returns the aggregated analytics snapshot (KPIs + distributions). |
| GET | `/system` | Process snapshot + ring-buffer of recent timings/errors. |
| POST | `/echo` | Echoes the JSON payload — useful for connectivity tests. |

Payload overrides: every route that touches the database accepts
`{"db_path": "/custom/path.sqlite"}` to override the default
`DATABASE_PATH`.