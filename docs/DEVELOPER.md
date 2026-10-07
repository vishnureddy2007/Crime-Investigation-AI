# Developer Guide

This document explains how the project is organised, what conventions the
codebase follows, and how to extend it safely.

> **Audience:** contributors, project evaluators, and anyone modifying the
> codebase. New here? Skim section 1, then 3, then jump to the cookbook
> (section 4 onwards).

---

## 1. Architecture at a glance

The project follows a **layered architecture** with one direction of
dependency: pages depend on services/models, services depend on
models/database/config. There are no circular imports.

```
+------------------------------------------------------------+
| Streamlit pages/  (image_detection, ai_summary, ...)       |
+--------------------------+---------------------------------+
                           | read/write st.session_state
                           v
+------------------------------------------------------------+
| app/      dependency-free HTTP façade (Phase 13)            |
+------------------------------------------------------------+
| services/ orchestration layer (chat, timeline, prediction,  |
|           analytics — Phase 13/14)                         |
+--------------------------+---------------------------------+
                           | pure functions, dataclasses
                           v
+------------------------------------------------------------+
| models/    (yolo, video, evidence, summary, report,        |
|             scene_planner, reconstruction, dashboard)      |
+--------------------------+---------------------------------+
                           | config + database
                           v
+------------------------------------------------------------+
| core/      logging, security, theming, performance,        |
|            resilience, observability, notifications        |
+------------------------------------------------------------+
| utils/     image_io, video_io, frame_utils, viz, OCR,      |
|            face_detection, i18n, export, db_hooks          |
+------------------------------------------------------------+
| database/  SQLite DDL + repository CRUD                    |
| config/    Centralised settings + env vars                 |
| outputs/   Generated artefacts                             |
+------------------------------------------------------------+
```

**Reading the diagram:**

- The **top layer** (`pages/`) is the only place `streamlit` is imported
  outside of `core/theming.py`.
- The **services layer** wraps models so pages don't reach into the
  pipeline directly. Pure functions where possible (chat assistant,
  timeline, prediction, analytics).
- The **models layer** is pure Python: `evidence_analyzer.py`,
  `dashboard.py`, `scene_planner.py`, etc. have no Streamlit import, no
  globals, no I/O. They take dataclasses and return dataclasses.
- The **`core/` package** is cross-cutting: logging, security
  (path traversal protection), theming, performance, resilience
  primitives (retry / rate-limit / circuit-breaker), observability
  (in-memory ring buffers), and notifications (UI toast helper).
- The **`app/` package** is a dependency-free HTTP dispatcher that
  exposes read-only views over the persistence + analytics layer.
- The **data layer** (`database/`, `config/`) is pure I/O and constants.

This shape lets the project run under `pytest` without any Streamlit
context — which is why we have 360+ unit tests passing today.

---

## 2. Repository layout

```
Crime-Investigation-AI/
|-- app.py                  # Streamlit entry point + sidebar nav
|-- app/                    # dependency-free HTTP façade
|   |-- __init__.py
|   `-- api.py
|-- core/                   # cross-cutting infrastructure
|   |-- logging.py          # configure_logging()
|   |-- security.py         # safe_filename / safe_join / PathTraversalError
|   |-- theming.py          # get_theme_css (light | dark, accessibility)
|   |-- performance.py      # Stopwatch, SystemSnapshot
|   |-- resilience.py       # retry, RateLimiter, CircuitBreaker
|   |-- observability.py    # RequestTimer, RequestLog ring buffer
|   `-- notifications.py    # success/error/warning/info helpers
|-- services/               # orchestration layer (pure functions)
|   |-- chat_assistant.py
|   |-- crime_timeline.py
|   |-- crime_prediction.py
|   `-- analytics.py
|-- config/                 # Centralised settings + env vars
|-- models/                 # YOLOv8 / Transformer / PDF / DOCX wrappers
|   |-- schemas.py          # Plain dataclasses (the contract)
|   |-- yolo_detector.py    # YOLOv8 wrapper with retry + observability
|   |-- video_processor.py  # OpenCV keyframe extractor
|   |-- evidence_analyzer.py# Counts/severity/category from detections
|   |-- summary_generator.py# Template + LLM (FLAN-T5) summaries
|   |-- report_generator.py # PDF (ReportLab) + DOCX (python-docx)
|   |-- scene_planner.py    # 4-6 scene storyboard planner
|   |-- reconstruction.py   # Storyboard rendering + MP4 synthesis
|   `-- dashboard.py        # Pure aggregations for Dashboard page
|-- pages/                  # Streamlit pages (one per sidebar entry)
|   |-- home.py
|   |-- image_detection.py
|   |-- video_processing.py
|   |-- evidence_analysis.py
|   |-- ai_summary.py
|   |-- chat_assistant.py
|   |-- crime_timeline.py
|   |-- crime_prediction.py
|   |-- report.py
|   |-- reconstruction.py
|   |-- dashboard.py        # read-only
|   |-- analytics.py        # cross-case KPIs + exports
|   |-- system_status.py    # platform + observability counters
|   |-- case_history.py
|   |-- settings.py
|   |-- help.py
|   |-- contact.py
|   `-- about.py
|-- utils/                  # I/O + AI helpers
|   |-- image_io.py
|   |-- video_io.py         # safe_filename in tmp paths
|   |-- frame_utils.py
|   |-- visualization.py
|   |-- db_hooks.py
|   |-- ocr.py              # pytesseract wrapper + stub fallback
|   |-- face_detection.py   # cv2 Haar cascade wrapper
|   |-- i18n.py
|   `-- export.py
|-- database/
|   |-- db.py               # SCHEMA_SQL + get_connection + init_db
|   `-- repository.py       # CRUD with window-function list_cases
|-- outputs/                # Generated artefacts
|-- images/                 # Sample inputs
|-- scripts/                # run.sh, run.bat, smoke_app.py
|-- tests/                  # pytest suite (33+ test files)
|-- docs/                   # DEVELOPER, API, ARCHITECTURE, ...
|-- .github/workflows/ci.yml# GitHub Actions matrix
|-- pytest.ini              # Test config + markers
|-- requirements.txt        # Runtime deps
|-- requirements-dev.txt    # pytest + pytest-cov
`-- README.md
```

---

## 3. Design principles

The codebase is intentionally small. Six rules govern everything:

1. **Pure functions for the core.**
   `evidence_analyzer.analyze(input) -> EvidenceAnalysis` takes a
   dataclass and returns a dataclass. Same for `dashboard.compute_kpis`,
   `scene_planner.plan_scenes`, `services/analytics.compute_snapshot`.
   No globals. No I/O. No Streamlit import.

2. **Dataclasses are the contract between layers.**
   `models/schemas.py` defines the data classes that pages, models,
   services, and the repository all share. Pages convert UI widgets into
   these objects; downstream layers consume them; pages render their
   output.

3. **Lazy model loading.**
   YOLOv8 and FLAN-T5 are heavy. `YOLODetector.model` and
   `SummaryGenerator.pipeline` are `@property`-backed caches — the model
   is loaded only on first call, never at import time. Tests run with
   no model files. (Phase 15: wrapped in `retry()` + `RequestTimer`
   from `core.resilience` + `core.observability`.)

4. **Graceful degradation.**
   If FLAN-T5 fails or is unavailable, `SummaryGenerator.generate()`
   returns a `template_text` baseline with `used_ai=False`. If Tesseract
   is missing, `utils/ocr.extract_text()` returns an empty string with
   the stub engine. If the Haar cascade is missing, `detect_faces()`
   returns `[]`. The UI never crashes on a missing dependency.

5. **Path safety.**
   All user-supplied filenames flow through `core.security.safe_filename`
   and `core.security.safe_join`. Any path-traversal escape raises
   `PathTraversalError`. `utils/video_io.validate_video_bytes` now
   sanitises its tmp filename (Phase 16 fix).

6. **Tests live next to the layer they cover.**
   `tests/test_evidence_analyzer.py` exercises the analyzer with
   hand-built `EvidenceAnalysis` instances. `tests/test_app_api.py`
   drives `app.dispatch()` with hand-built payloads. `tests/test_core_notifications.py`
   monkeypatches `sys.modules["streamlit"]` to exercise the UI helpers
   without launching the dashboard.

---

## 4. Cookbook: adding a new YOLO class

Suppose you want the detector to recognise **fire** (COCO class id 13)
as a crime-relevant object.

1. Edit `config/settings.py` — add the entry to `YOLO_CLASS_MAPPING`:
   ```python
   "fire": "fire",
   ```
   `RELEVANT_CLASSES` is derived from the mapping keys, so it picks
   the new class up automatically.

2. (Optional) Add a color for the bounding box in
   `utils/visualization.py::_BOX_COLORS`:
   ```python
   "fire": "#FF6B35",
   ```

3. Update `tests/test_evidence_analyzer.py` if you want to assert the
   new label affects severity or category rules.

4. Run `pytest -q`. All 360+ tests should still pass.

That's the entire change — no model wrapping to update, no schema
changes, no UI changes (the dashboard reads `counts_by_label`
dynamically).

---

## 5. Cookbook: adding a new HTTP route

The dependency-free dispatcher in `app/__init__.py` is the easiest way
to expose a new read-only endpoint.

```python
from app import register

@register("/my-endpoint", methods=("GET",))
def _my_endpoint(payload: dict) -> dict:
    # pure work, no Streamlit
    return {"result": "..."}
```

Returning a `dict` produces `ApiResponse(ok=True, status=200, data=...)`.
To return an error, raise one of the package exceptions:

```python
from app import NotFound

@register("/cases", methods=("GET",))
def _cases(payload):
    if not db_path.exists():
        raise NotFound("database not found")
```

The dispatcher maps `ApiError` (and its `NotFound` subclass) to the
right HTTP status; `ApiResponse.status` reflects the chosen status.

Always add a test in `tests/test_app_api.py`.

---

## 6. Cookbook: adding a new page

1. Create `pages/my_new_page.py`:
   ```python
   import streamlit as st

   def render() -> None:
       st.title("My New Page")
       # ...

   if __name__ == "__main__":
       render()
   ```

2. Register it in `app.py`:
   ```python
   PAGES.append("My New Page")
   # ...
   elif page == "My New Page":
       from pages.my_new_page import render as render_my_new
       render_my_new()
   ```

3. If the page produces an artefact that should persist across page
   switches, store it under a `last_*` key in `st.session_state`. To
   persist it across sessions, call `auto_save_last_my_artefact()`
   (see section 7).

4. Add a smoke test in `tests/test_smoke_full.py` if it imports new
   packages.

---

## 7. Cookbook: extending the database

To add a seventh table (say, `notes`):

1. Append the DDL to `SCHEMA_SQL` in `database/db.py`. Always include
   `FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE`.

2. Add pure CRUD functions in `database/repository.py`:
   ```python
   def save_note(db_path: Path, case_id: int, text: str) -> int:
       ...
   def list_notes_for_case(db_path: Path, case_id: int) -> list[dict]:
       ...
   ```
   Prefer window functions (`ROW_NUMBER() OVER (PARTITION BY ...)`) to
   correlated subqueries — see `list_cases()` for the canonical
   pattern.

3. (Optional) Add an `auto_save_last_note(db_path)` helper in
   `utils/db_hooks.py` using the same try/except + `_warn` shape as
   the existing hooks.

4. Add a test in `tests/test_database.py` covering save + list +
   cascade delete.

---

## 8. Cookbook: adding a UI notification

Use `core.notifications` instead of `st.success` / `st.error` directly
so every page renders consistent feedback (icon prefix + body style).

```python
from core.notifications import show_success, show_error, show_warning

show_success("Saved", "case #42 persisted")
show_error("Upload failed", "file exceeds size limit")
show_warning("Disk almost full", "90% used")
```

For pure-formatting helpers (no Streamlit call), use
`format_message(Severity.X, title, body)` which is fully unit-testable.

For Toast-style feedback (auto-dismiss), pass `toast=True`.

Always add a regression test in `tests/test_core_notifications.py` if
you change the render path.

---

## 9. Testing conventions

The project uses **pytest** with custom markers (see `pytest.ini`):

| Marker        | Purpose                                              |
|---------------|------------------------------------------------------|
| `slow`        | Long-running tests; deselect with `-m "not slow"`    |
| `integration` | Tests requiring optional or heavy deps               |
| `streamlit`   | Tests that mock `st.session_state`                   |

The test suite is driven by `tests/conftest.py`, which exposes shared
fixtures:

- `project_root()` — `@lru_cache` Path to the project root.
- `fixtures_dir` — session-scoped path to a fixtures dir.
- `sample_image_path` — auto-creates a 16×16 RGB PNG-as-JPG if missing.
- `sample_video_path` — auto-creates a 2-second 8×8 MP4 if missing.
- `db_path(tmp_path)` — per-test fresh SQLite file.

**Streamlit-aware tests** use one of two patterns:

```python
# 1. utils.db_hooks style — SimpleNamespace with a plain dict state
fake_streamlit = SimpleNamespace(session_state={})

# 2. core.notifications style — full module replacement
import sys, types
mock = types.ModuleType("streamlit")
mock.success = lambda p: ...  # capture
monkeypatch.setitem(sys.modules, "streamlit", mock)
```

See `tests/test_db_hooks.py` and `tests/test_core_notifications.py`.

---

## 10. CI workflow

`.github/workflows/ci.yml` runs on every push and PR to `main` on a
matrix of **Python 3.11, 3.12, 3.13**. Steps:

1. `actions/checkout@v4`
2. `actions/setup-python@v5` with pip cache for both requirement files
3. `pip install -r requirements.txt -r requirements-dev.txt`
4. `pytest --cov=app --cov=core --cov=services --cov=models --cov=utils --cov=database --cov=config --cov-report=term-missing`

Coverage is **terminal only** (no Codecov / no HTML report). A
separate `docker-build` job runs on pushes to `main`.

---

## 11. Style and quality bars

- **PEP 8** — enforced by reviewer eyes, not by a linter.
- **Type hints** — every public function has parameter + return annotations.
- **Docstrings** — module-level docstring + per-function docstring on
  every public symbol in `models/`, `core/`, `services/`, `utils/`.
- **No circular imports** — verified by `tests/test_smoke_full.py`.
- **Exception handling** — broad `try/except` only at the I/O boundary
  (utils, db_hooks, summary_generator LLM call, ocr stub, face
  detection). Pure functions raise.
- **HTTP errors** — route handlers raise `NotFound` / `ApiError`; the
  dispatcher maps them to the right status code, never to a `{"ok":
  True, "warning": ...}` envelope (Phase 17 fix).

---

## Next steps

- See the [API reference](API.md) for every public symbol with its type.
- See the [Progress table](PROGRESS.md) for milestone-by-milestone notes.
- See the [Architecture diagram](ARCHITECTURE.md) for the mermaid view.