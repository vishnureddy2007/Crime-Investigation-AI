# Progress

Per-milestone record of what was shipped, including key deliverables,
tests added, and status. Final two rows are planned.

---

## Milestone-by-milestone

| #  | Milestone | Goal | Key deliverables | Tests added | Status |
|----|-----------|------|------------------|-------------|--------|
| 1  | Project Setup | Bootstrap a runnable Python project with a Streamlit skeleton. | `app.py` with sidebar nav, `config/settings.py`, `requirements.txt`, `pages/home.py`, ASCII tree in README. | 0 | ✅ done |
| 2  | YOLOv8 Image Detection | Detect crime-relevant objects in uploaded images. | `models/yolo_detector.py` (Ultralytics wrapper), `utils/visualization.py` (bounding-box drawing), `pages/image_detection.py`, `YOLO_CLASS_MAPPING` (10 COCO classes). | 5 | ✅ done |
| 3  | Video Processing | Extract keyframes from short CCTV clips and run per-frame detection. | `models/video_processor.py` (OpenCV), `models/schemas.FrameResult` + `VideoAnalysisResult`, `utils/frame_utils.py`, `pages/video_processing.py`. | 8 | ✅ done |
| 4  | Evidence Analysis | Turn raw detections into a structured `EvidenceAnalysis` (severity, category, observations). | `models/evidence_analyzer.py` (pure functions), `SEVERITY_WEIGHTS`, `CATEGORY_RULES`, `SEVERITY_THRESHOLDS`, `pages/evidence_analysis.py`. | 14 | ✅ done |
| 5  | AI Investigation Summary | Generate investigator-style summaries with FLAN-T5. | `models/summary_generator.py` (template + LLM), graceful fallback when LLM unavailable, `pages/ai_summary.py`. | 6 | ✅ done |
| 6  | Report Generation | Produce PDF + DOCX investigation reports. | `models/report_generator.py` (ReportLab + python-docx), `pages/report.py`, `REPORT_TITLE/REPORT_REMARKS`. | 8 | ✅ done |
| 7  | Crime Scene Reconstruction | Build a logical storyboard + short MP4 video. | `models/scene_planner.py` (4-6 scenes), `models/reconstruction.py` (OpenCV `mp4v` writer), `models/schemas.Storyboard*`, `pages/reconstruction.py`. | 14 | ✅ done |
| 8  | Dashboard | Aggregate last-pipeline results into KPIs, charts, severity bands. | `models/dashboard.py` (6 pure functions), `SEVERITY_LEVEL_COLORS`, `pages/dashboard.py` (read-only), `compute_severity_band` + `summarize_session_state`. | 19 | ✅ done |
| 9  | Database + Case History | Persist analyses, summaries, reports, storyboards in SQLite; browse + reload past cases. | `database/db.py` (`SCHEMA_SQL`, `get_connection`, `init_db`), `database/repository.py` (5-table normalized CRUD with cascade delete), `utils/db_hooks.py` (auto-save hooks), `pages/case_history.py`. | 28 | ✅ done |
| 10 | Testing + CI | Real QA layer: pytest config, conftest fixtures, ~24 new tests, GitHub Actions matrix. | `pytest.ini` (markers), `requirements-dev.txt`, `tests/conftest.py` (5 fixtures), 4 new test files (visualization, utils_io, db_hooks, smoke_full), `.github/workflows/ci.yml` (3.11+3.12). | 28 | ✅ done |
| 11 | Documentation | Plain-Markdown docs for reviewers: install, dev guide, API ref, progress. | `docs/INSTALLATION.md`, `docs/DEVELOPER.md`, `docs/API.md`, `docs/PROGRESS.md`, `docs/README.md` (index). | 0 | ✅ done |
| 12 | Deployment | One-command local launchers, single-stage Dockerfile, optional env-var overrides, CI docker-build smoke job. | `scripts/run.sh` + `scripts/run.bat`, `Dockerfile`, `.dockerignore`, `Procfile`, `runtime.txt`, `docs/DEPLOYMENT.md`, env-var block in `config/settings.py`, docker-build job in CI. | 17 | ✅ done |
| 13 | Production polish & expansion | Hardening + missing features: dark theme, security helpers, chat, timeline, prediction, OCR, face detection, contact form, docker-compose. | `core/{logging,security,theming}.py`, `services/{chat_assistant,crime_timeline,crime_prediction}.py`, `utils/{ocr,face_detection}.py`, `app/__init__.py` + `app/api.py`, `pages/{_layout,settings,help,contact,chat_assistant,crime_timeline,crime_prediction}.py`, `docker-compose.yml`, `.env.example`, ARCHITECTURE.md, USER_MANUAL.md. | +90 | ✅ done |
| 14 | Enterprise expansion v2 | Analytics page + System Status page; i18n + export utilities; resilience primitives (retry / rate-limit / circuit-breaker); accessibility CSS (skip link, focus ring, reduced-motion); new env vars LOG_LEVEL / LOG_FILE. | `core/{performance,resilience}.py`, `services/analytics.py`, `utils/{i18n,export}.py`, `pages/{analytics,system_status}.py`, accessibility chunk in `core/theming.py`, expanded `config/settings.py`. | +57 | ✅ done |
| 15 | Observability + API expansion | Ring-buffer request timing; notifications helper; HTTP API endpoints for cases / analytics / system / health/deep; YOLO detector wrapped in retry + observability. | `core/{observability,notifications}.py`, 4 new HTTP routes in `app/__init__.py`, resilience wrappers in `models/yolo_detector.py`, 4 new YOLO unit tests. | +20 | ✅ done |
| 16 | Audit + fix pass | Severity-band drift between `services/analytics` and `config.SEVERITY_THRESHOLDS`; path-traversal in `utils/video_io` tmp filename; OCR stub crash on numpy arrays; LOG_FILE drift between `core.logging` and `config.settings`. All fixed + locked with 8 new regression tests. | `services/analytics.py`, `utils/{video_io,ocr}.py`, `core/logging.py`, `tests/test_phase16_fixes.py`. | +8 | ✅ done |
| 17 | HTTP envelope tightening + N+1 fix | `/cases` and `/analytics` now return 404 NotFound (ok=False) when the DB is missing instead of masking the failure as `{"ok": True, "warning": ...}`. `list_cases` replaced its two correlated subqueries with a single `LEFT JOIN ... ROW_NUMBER()` window function. New tests cover notifications with mocked streamlit, OCR with mocked pytesseract, and face detection with mocked cv2. DEVELOPER.md refreshed. | `app/__init__.py`, `database/repository.py`, `tests/test_app_api.py`, `tests/test_database.py`, `tests/test_core_notifications.py`, `tests/test_utils_ocr.py`, `tests/test_utils_face.py`, `docs/DEVELOPER.md`, `docs/INSTALLATION.md`. | +11 | ✅ done |
| 18 | Coverage expansion + dead-code sweep | Removed unused `_SLUG_RE` + `_slug()` from `app/__init__.py`. Covered the previously unreachable keyframe save branch in `models/video_processor.py` (lines 127-132). Covered the FLAN-T5 success path in `services/chat_assistant.py` (lines 185-206). Covered `/routes` route handler and `_resolve_db_path` config fallback. `models/video_processor.py` 80%→99%, `services/chat_assistant.py` 75%→97%, `app/__init__.py` 83%→88%. | `app/__init__.py`, `tests/{test_app_api,test_services_chat,test_video_processor}.py`, `docs/PROGRESS.md`, `README.md`. | +16 | ✅ done |
| 19 | Defensive coverage pass | Pushed `database/db.py` to 100% (rollback + `_row_to_dict`), `models/dashboard.py` to 100% (empty-thresholds fallback, streamlit-less `summarize_session_state`), `models/yolo_detector.py` to 96% (`save_annotated_image` / `result_to_bytes` None branches, `get_detector`), `utils/db_hooks.py` to 96% (every hook's exception swallow path, `_resolve_path`, `_resolve_case_id` reuse), `core/performance.py` to 91% (psutil fallback, corrupt startup marker), `app/__init__.py` to 93% (`ApiError` custom status, `/cases` + `/analytics` happy paths). | `tests/{test_database,test_dashboard,test_yolo_detector,test_db_hooks,test_core_performance,test_app_api}.py`. | +24 | ✅ done |
| 20 | Audit + fix pass v2 | **Bug fix:** `services/analytics.py` `severity_counts` no longer hardcodes `"medium"` — it now derives its bucket labels from `SEVERITY_BANDS`, so the canonical `"moderate"` label from `config.SEVERITY_THRESHOLDS` is preserved end-to-end. **Dead code:** removed the `@contextmanager Stopwatch()` function in `core/performance.py` that was shadowed by the actual `Stopwatch` class — same pattern as the Phase 18 `_slug` removal. 3 new regression tests lock in the moderate bucket behaviour. | `services/analytics.py`, `core/performance.py`, `tests/test_services_analytics.py`, `docs/PROGRESS.md`, `README.md`. | +3 | ✅ done |
| 21 | Defensive coverage pass v2 | Pushed `models/reconstruction.py` to 97% (long-caption wrap, `None`-image scene skip, placeholder-only video, video-writer-open failure), `models/scene_planner.py` to 92% (padding loop, candidate-exhausted break, long-caption placeholder, `_wrap_text` edge cases, `_tint_image` black-tint skip, `_resize_to_panel` letterbox), `utils/visualization.py` to 93% (`_color_for` unknown-label fallback, `_font` happy path, `draw_detections` textlength AttributeError fallback, unknown-label default color, empty-list idempotence), `utils/image_io.py` to 96% (JPEG/RGBA round-trips, missing-file raise), `utils/ocr.py` to 91% (stub truncation on huge arrays), `utils/face_detection.py` to 93% (grayscale PIL input). | `tests/{test_reconstruction,test_scene_planner,test_visualization,test_image_io,test_utils_ocr,test_utils_face}.py`, `docs/PROGRESS.md`, `README.md`. | +23 | ✅ done |
| 22 | Audit + fix pass v3 | **Bug fix:** `pages/analytics.py` Severity bands bar chart hardcoded the legacy label `"medium"` — the "moderate" bar was therefore always 0 even after Phase 20 fixed `services/analytics.py`. Now derives band labels from `config.SEVERITY_THRESHOLDS` (canonical source), so the chart can never drift from the snapshot again. **Bug fix:** `pages/analytics.py` JSON export used `datetime.utcnow()` (deprecated in 3.12, removed in 3.14+) — switched to `datetime.now(timezone.utc)` for an RFC 3339 `+00:00`-suffixed ISO timestamp. 5 new source-grep + behavioural tests lock both fixes in. | `pages/analytics.py`, `tests/test_phase22_fixes.py`, `docs/PROGRESS.md`, `README.md`. | +5 | ✅ done |
| 23 | Coverage push on lowest-tier modules | Pushed `utils/ocr.py` 91%→100% (added regression tests for `_stub_extract` confidence=0.0 fallback when `tobytes()` raises, and `extract_text` confidence=0.0 fallback when `image_to_data` raises). Pushed `models/scene_planner.py` 92%→100% (added regression tests for `_panel_placeholder` OSError fallback when Arial is missing, and `textlength` AttributeError fallback for old Pillow). **Dead code:** removed the unreachable `if not padded: break` defensive branch at `models/scene_planner.py:201` — proved by invariant analysis that `len(scenes) < len(candidates)` already guards the loop exit, making line 201 impossible to fire in any input (same pattern as the Phase 18 `_slug` and Phase 20 `@contextmanager Stopwatch` removals). | `models/scene_planner.py`, `tests/{test_scene_planner,test_utils_ocr}.py`, `docs/PROGRESS.md`, `README.md`. | +5 | ✅ done |
| 24 | Defensive branch + module hygiene | Pushed `utils/export.py` 95%→100% (3 tests: bool→"true"/"false" CSV cells, empty markdown columns, None markdown cell rendering). Pushed `models/summary_generator.py` 93%→100% (4 tests: multi-weapon template plural branch, low-confidence cross-check disclaimer, `pipeline` RuntimeError when `model_name=None`, empty-evidence "appears clear" message). Pushed `services/chat_assistant.py` 97%→100% (1 test: `pipeline` RuntimeError when `model_name=None`). Pushed `utils/video_io.py` 95%→98% (4 tests: oversize buffer, empty buffer, missing file, non-video file). All 12 new tests live in existing test files. | `tests/{test_utils_export,test_summary_generator,test_services_chat,test_video_io}.py`, `docs/PROGRESS.md`, `README.md`. | +12 | ✅ done |
| 25 | End-of-roadmap verification | **Project-completion audit.** Verified all four recurring audit patterns continue to hold: (1) no severity/category label drift — `"medium"` only appears in regression tests; (2) no dead code since Phase 23's `if not padded: break` removal; (3) no path-traversal hazards — `core/security.py:52`'s `os.path.basename` use is part of the *defence*, not a hazard; (4) no deprecated APIs — `datetime.utcnow()` gone since Phase 22. No paid / cloud dependencies confirmed (grep for openai/anthropic/google returns nothing). Pushed `models/summary_generator.py` 100% (added `test_generator_pipeline_lazily_loads_and_caches` covering the `hf_pipeline()` call site). Pushed `services/chat_assistant.py` 97%→100% (added `test_chat_assistant_pipeline_lazily_loads_and_caches`). Both use `monkeypatch.setitem(sys.modules, "transformers", fake)` to inject a fake module so the lazy-load branch fires without pulling the real 250 MB transformers dep. | `tests/{test_summary_generator,test_services_chat}.py`, `docs/PROGRESS.md`, `README.md`. | +2 | ✅ done |
| 26 | Audit-script automation | Codified the four-pattern sweep (label drift, dead code, path safety, deprecated APIs) that was being run by hand across Phases 16–25 into a single `scripts/audit.py` script with a regex-driven `AuditReport`. The script is dependency-free (stdlib only), prints "✅ audit clean" on a healthy codebase and exits non-zero when a finding is added — wired into the test suite as `test_audit_clean_on_current_codebase`. 6 tests in `tests/test_audit_script.py` cover the clean path, detection of each pattern family, the `allow_in_files` exemption (including off-tree fixtures under `tmp_path`), the `line_filter` comment-skip heuristic, and report grouping by pattern_id. | `scripts/audit.py`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`. | +6 | ✅ done |
| 27 | Audit-script hardening | Two new patterns added to `scripts/audit.py` (now 6 total): **logging hygiene** (5a: `print()` calls outside docstrings → use `core.logging`; 5b: bare `except:` and `except Exception:` → narrow the type and log) and **SQL safety** (`.execute(...)` with f-strings or string concat → must use `?` placeholders). The 18 `except Exception:` clauses already in the codebase were narrowed to specific exception types (`ImportError`, `AttributeError`, `RuntimeError`, `OSError`, `sqlite3.Error`, `KeyError`, `ValueError`, etc.) — each one a documented resilience wrapper around an optional dependency or streamlit-stale API. The audit script is now wired into `.github/workflows/ci.yml` as a pre-test gate (fails the job before pytest runs if any pattern regresses). 5 new tests in `tests/test_audit_script.py` cover: print detection outside docstrings, print skip inside docstrings (the `core/observability.py:103` example), bare-except detection, f-string `.execute()` detection, and parameterised `.execute()` is NOT flagged. | `scripts/audit.py`, `.github/workflows/ci.yml`, `app/__init__.py`, `core/{logging,notifications,performance}.py`, `database/db.py`, `models/{dashboard,summary_generator,yolo_detector}.py`, `pages/_layout.py`, `services/{analytics,chat_assistant}.py`, `utils/{db_hooks,face_detection,ocr}.py`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`. | +5 | ✅ done |
| 28 | Audit-script CI artifact + JSON output | `scripts/audit.py` now emits a deterministic JSON report (`--json` or `--json-out <path>`) alongside the human-readable output. The schema is versioned via `JSON_SCHEMA_VERSION = 1` so downstream consumers can diff between runs. Clean runs produce `{"clean": true, "finding_count": 0, ...}` so CI dashboards can detect new findings on a green job. CI now uploads `AUDIT.json` as a workflow artifact on every push (14-day retention). 4 new tests lock in the JSON shape (deterministic byte-identical serialisation, clean-run document, off-tree fixture paths, subprocess `--json` writes the artifact). | `scripts/audit.py`, `.github/workflows/ci.yml`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`. | +4 | ✅ done |
| 29 | Audit-script pattern registry + `--list-patterns` | Added a declarative `PATTERNS` registry (`tuple[AuditPattern, ...]`) to `scripts/audit.py` — adding a new audit pattern is now one row of data instead of one function-call site. The existing 10 patterns are all listed, plus a new `AuditPattern` dataclass with `pattern_id / severity / regex / short_summary / allow_in_files / use_docstring_filter` fields. New CLI flag `--list-patterns` prints every active pattern_id + severity and exits 0 — lets contributors see what's covered without reading the source. New `list_patterns()` function returns a JSON-friendly snapshot for tests. 3 new tests lock in: the registry covers every active pattern_id, `--list-patterns` subprocess output contains every ID, and `list_patterns()` row shape is stable. | `scripts/audit.py`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`. | +3 | ✅ done |
| 30 | Pre-commit hook + CONTRIBUTING guide | Added `.pre-commit-config.yaml` (a `repo: local` hook with `id: project-audit` that runs `python scripts/audit.py` on every commit — `language: system`, `pass_filenames: false`, `always_run: true`). Added `CONTRIBUTING.md` documenting local setup, the audit script, the active patterns, how to add a new pattern (one row in the `PATTERNS` registry), how to use the `allow_in_files` exemption, custom line filters, and coding conventions (no paid deps, no cloud calls, parameterised SQL, narrow excepts, validated inputs). 3 new tests lock in: pre-commit config exists with the right structural fields, the hook command (`python scripts/audit.py`) actually exits 0 on the current codebase, and CONTRIBUTING.md references the audit script, the registry, and all six pattern families. | `.pre-commit-config.yaml`, `CONTRIBUTING.md`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`. | +3 | ✅ done |
| 31 | Audit-script `--diff` mode | Added `--diff BEFORE AFTER` to `scripts/audit.py` — compares two `AUDIT.json` artifacts and prints only the findings introduced between them. New `DiffReport` dataclass with `from_json_files` + `from_dicts` classmethods; findings are keyed by `(file, line, pattern_id)` so the same offender in the same place is the same finding across runs. `has_regressions` is True only when a finding was added (resolving a finding is informational, not a failure). Schema-version mismatch raises `ValueError` so we never silently compare reports of different shapes. Exit code 0 = no new findings, 1 = regression detected, 2 = config error (file not found / schema mismatch). 5 new tests lock in: clean diff, new-finding diff, resolved-finding diff, schema-version mismatch raises, subprocess-level test that plants an offender and verifies the diff surfaces it. | `scripts/audit.py`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`, `CONTRIBUTING.md`. | +5 | ✅ done |
| 32 | Audit-script CI PR-diff job | Added a new `pr-audit-diff` job to `.github/workflows/ci.yml` that runs only on pull_request events. The job checks out the PR base ref and the PR head into separate directories, runs `python scripts/audit.py --json` in each, then runs `python scripts/audit.py --diff base/AUDIT.json head/AUDIT.json`. Exit 0 = no regressions; exit 1 = new findings (job fails). Uploaded as an `audit-diff` artifact with 14-day retention so reviewers can re-download and re-diff locally. CLI now exits 2 (not a stack trace) when a file is missing — locked in by `test_audit_diff_cli_handles_missing_file_gracefully`. 3 new tests: ci.yml defines the pr-audit-diff job, ci.yml gates it on pull_request events only, and the missing-file path exits 2 with a clear message. | `.github/workflows/ci.yml`, `scripts/audit.py`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`. | +3 | ✅ done |
| 33 | Audit-script `--scope` flag | Added `--scope <dir>` to `scripts/audit.py` so contributors can spot-check a single layer (`core`, `services`, `models`, `app`, `utils`, `database`, `pages`) without scanning the whole tree. Defaults to `all` (current behaviour). Threaded `scope` through `_iter_source_files`, `_scan`, `_scan_filtered`, and every `check_*` function. Per-test monkey-patches of `_iter_source_files` in `tests/test_audit_script.py` were updated to accept the new `scope` keyword argument so the existing 9 detector tests stay green. argparse rejects unknown values with exit 2 (silently coercing to `all` would defeat the spot-check purpose). 6 new tests lock in: `--scope` is in the help text, `--scope core` returns a strict subset of the full sweep, unknown scope raises `ValueError`, `--scope all` matches the no-flag default, subprocess rejects a bogus value with exit 2, and end-to-end subprocess on `--scope core` runs clean. CONTRIBUTING.md now shows a `--scope core` example next to the full-sweep command. | `scripts/audit.py`, `tests/test_audit_script.py`, `CONTRIBUTING.md`, `docs/PROGRESS.md`, `README.md`. | +6 | ✅ done |
| 34 | Audit `--fail-on` flag + severity summary | Added `--fail-on <severity>` to `scripts/audit.py` so CI can decide which severity bucket makes the process exit 1. Default is `smell` (every finding above `info` is fatal). New `--strict` flag is a shorthand for `--fail-on info` (every finding is fatal). Locked the canonical severity order in a `SEVERITY_ORDER` tuple (`info < smell < drift < bug`); unknown severities are treated as the highest rank so a typo can never silently downgrade a finding. New `AuditReport.count_by_severity()` + `findings_meeting(severity)` + `has_findings_meeting(severity)` helpers drive both the exit-code logic and the new `[INFO] severity totals: bug=N, drift=N, smell=N` summary line. `--list-patterns` now prints per-severity counts in the header and the active fail-on threshold. 11 new tests lock in: severity order is preserved, unknown severity is ranked highest, `count_by_severity()` is correct, `findings_meeting` threshold semantics, default `--fail-on smell` ignores info findings, `--fail-on bug` ignores smell findings, `--strict` is accepted and equals `--fail-on info`, subprocess exits 0 on a clean tree with `--fail-on smell`, argparse rejects bogus `--fail-on` values with exit 2, `--list-patterns` shows the per-severity breakdown + threshold, and `--strict` propagates to `--list-patterns`. | `scripts/audit.py`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`, `CONTRIBUTING.md`. | +11 | ✅ done |
| 35 | Audit `--diff` honours `--fail-on` | The `--diff` short-circuit now consults `--fail-on` so the same threshold drives both the live sweep and the PR-diff sweep. Without this wiring, a smell-level regression would fail the diff even when the rest of the pipeline was tuned with `--fail-on bug`. New `DiffReport.regressions_meeting(severity)` + `has_regressions_meeting(severity)` helpers mirror the Phase 34 report helpers. CLI now prints `[INFO] diff introduced N finding(s) but all are below fail-on threshold: <sev>` when regressions exist but none meets the threshold, so CI logs make the exit-0 reason obvious. 5 new tests lock in: regressions_meeting filters by severity correctly, clean diff returns empty list, `--fail-on bug` ignores smell regressions in the diff subprocess, `--fail-on smell` (default) still catches smell regressions, and `--strict` catches info-level regressions. | `scripts/audit.py`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`, `CONTRIBUTING.md`. | +5 | ✅ done |
| 36 | Audit pattern: emoji-in-print | New `logging-hygiene:emoji-in-print` smell flags any non-ASCII emoji glyph inside `print(...)` so CI logs stay grep-friendly and terminal-rendering-safe. Streamlit UI code that legitimately uses emoji via `st.markdown` / button labels is untouched because the regex is scoped to print calls only. Covers the common Unicode blocks: dingbats (`U+2600`-`U+27BF`, ✅/❌/⚠) and supplementary multilingual plane emoji (`U+1F300`-`U+1FAFF`, 🔍/🚨). `scripts/smoke_app.py` is allow-listed (its `OK` lines sometimes round-trip through terminals that inject glyphs). 3 new tests lock in: positive detection on a UTF-8 file with five common emoji, `allow_in_files` actually exempts, and emoji outside print calls (Streamlit UI, logger messages, string literals) do NOT trigger. Active pattern count is now 11 (smell=5). | `scripts/audit.py`, `tests/test_audit_script.py`, `docs/PROGRESS.md`, `README.md`, `CONTRIBUTING.md`. | +3 | ✅ done |
| 37 | Utils coverage push (visualization + face_detection) | Both `utils/visualization.py` and `utils/face_detection.py` were at 93% line coverage; pushed both to 100% with 3 small targeted tests, zero production code changes. Visualization: monkeypatched `utils.visualization.ImageFont.truetype` to raise `OSError` on the first call (delegating subsequent calls back to the original so `load_default()` can resolve a font), asserting `_font()` returns a non-None font — exercises lines 32-33. Face detection: (1) passed a numpy array (no `.convert` method) to `detect_faces()` to exercise line 73 (the array-input branch), with a stub `cv2.CascadeClassifier` returning one box; (2) monkeypatched `cv2.data.haarcascades` to raise `OSError` so `_cascade_path()` returns `None` — exercises lines 50-51 and the cascade-missing fallback in `detect_faces()`. Both modules now read 100% covered. | `tests/test_visualization.py`, `tests/test_utils_face.py`, `docs/PROGRESS.md`, `README.md`, `CONTRIBUTING.md`. | +3 | ✅ done |
| 38 | Utils coverage push (ocr) | `utils/ocr.py` was at 94% line coverage (3 missed lines: 43-44, 82, 100); pushed to 100% with 5 small targeted tests, zero production code changes. (1) `is_ocr_available()` returns False when `import pytesseract` raises `AttributeError` — monkeypatched `builtins.__import__` to raise; exercises the `(ImportError, AttributeError)` clause on line 43. (2) `extract_text()` short-circuits to `_stub_extract()` when `is_ocr_available()` is False — exercises line 82. (3) `extract_text()` falls back to the stub engine when `pytesseract.image_to_string` raises `OSError` — exercises line 100. (4) `_stub_extract()` with a numpy array routes through the `.tobytes()` + `.nbytes` branch — exercises lines 63-64. (5) `_stub_extract()` with a plain list (no `.size`+`.convert`, no `.tobytes`) routes through the `str(...)` fallback — exercises line 66. Module is now 100% covered (47 stmts / 0 missed). | `tests/test_utils_ocr.py`, `docs/PROGRESS.md`, `README.md`, `CONTRIBUTING.md`. | +5 | ✅ done |

---

## Test suite growth

| After milestone | Test functions | Test files |
|---|---|---|
| M1  | 0  | 0 |
| M2  | 5  | 1 |
| M3  | 13 | 3 |
| M4  | 27 | 5 |
| M5  | 33 | 7 |
| M6  | 41 | 8 |
| M7  | 55 | 10 |
| M8  | 74 | 11 |
| M9  | 102 | 13 |
| M10 | 147 | 17 |
| M11 | 147 | 17 |
| M12 | 165 | 17 |
| M13 | 275 | 26 |
| M14 | 332 | 31 |
| M15 | 352 | 33 |
| M16 | 360 | 34 |
| M17 | 371 | 36 |
| M18 | 392 | 37 |
| M19 | 416 | 37 |
| M20 | 419 | 37 |
| M21 | 442 | 37 |
| M22 | 448 | 38 |
| M23 | 453 | 38 |
| M24 | 465 | 38 |
| M25 | 467 | 38 |
| M26 | 473 | 39 |
| M27 | 478 | 39 |
| M28 | 482 | 39 |
| M29 | 485 | 39 |
| M30 | 488 | 39 |
| M31 | 493 | 39 |
| M32 | 496 | 39 |
| M33 | 502 | 39 |
| M34 | 513 | 39 |
| M35 | 541 | 39 |
| M36 | 543 | 39 |
| M37 | 546 | 39 |
| M38 | 551 | 39 |

> **Today:** 551 tests across 39 files, ~98% line coverage on the
> `app`, `core`, `services`, `models`, `utils`, `database`, and
> `config` packages.

---

## What was shipped overall

By the end of Milestone 13, the project has:

- **16 user-facing pages** — `Home`, `Dashboard`, `Image Detection`,
  `Video Processing`, `Evidence Analysis`, `AI Summary`, `Chat Assistant`,
  `Crime Timeline`, `Crime Prediction`, `Report`, `Reconstruction`,
  `Case History`, `Settings`, `Help`, `Contact`, `About`.
- **Cross-cutting infrastructure** in `core/` (`logging`, `security`,
  `theming`) used by every page.
- **Service layer** in `services/` (`chat_assistant`,
  `crime_timeline`, `crime_prediction`) keeping business flows
  out of the UI.
- **HTTP façade** in `app/` with a tiny dependency-free dispatcher
  (handy for notebooks, scripts, and future REST).
- **AI utilities** in `utils/` — `ocr` and `face_detection`
  alongside the original image/video I/O.
- **11 models** under `models/` covering detection, video,
  evidence, summary, report, scene planning, reconstruction,
  dashboard, schemas, plus the new image-save helper.
- **2 database modules** — DDL/connection + CRUD repository,
  covering a 6-table normalized schema with cascade delete (added
  the `feedback` table in Phase 13).
- **3 CI jobs** — Python 3.11 + 3.12 + 3.13 on every push and PR,
  plus the docker-build smoke job on `main` only.
- **Deployment story** — `scripts/run.sh` + `scripts/run.bat`
  launchers, single-stage `Dockerfile`, `docker-compose.yml`,
  `.env.example`, and `docs/DEPLOYMENT.md`.
- **Documentation** — `INSTALLATION.md`, `DEVELOPER.md`, `API.md`,
  `PROGRESS.md`, `ARCHITECTURE.md` (mermaid), `USER_MANUAL.md`,
  and `DEPLOYMENT.md`.
- **Coverage targets** — terminal report for `app`, `core`,
  `services`, `models`, `utils`, `database`, `config`. Current line
  coverage ~89%.
- **0 paid dependencies**, **0 cloud calls**.

The codebase is small enough that one developer can read it in a
sitting and large enough to demonstrate every requested AI
concept (detection, NLP, generative reconstruction, structured
reasoning, prediction, RAG-style chat, OCR, face detection, and
persistence).

---

## See also

- [Installation](INSTALLATION.md) — how to run the app.
- [Developer Guide](DEVELOPER.md) — how to extend it.
- [API Reference](API.md) — every public symbol.

---

## Phase 40 — Professional UI polish

The brief was to transform the project from a student/demo look into a
professional, clean, modern, academic, forensic-investigation AI
application suitable for a final-year B.Tech demonstration. Achieved
without changing any data flows or functionality.

### Changes

- **`core/theming.py`** — tightened radii, introduced `Inter` / system
  UI typography stack, added `.brand-mark`, `.brand-tag`, `.cv-empty`,
  `.cv-pill` (ok / warn / danger) CSS classes.
- **`core/icons.py`** — new module. Replaces every emoji-as-icon with
  text-only constants (`ICON_*`, `SECTION_*`, `STATUS_*`, `ACTION_*`).
- **`pages/_layout.py`** — sidebar header now renders a clean
  `CV` wordmark + brand name + tagline (no skull/gun/emoji).
  Added three new helpers:
  - `empty_state(title, body, action_label=None, action_target=None)`
  - `friendly_error(exc, *, fallback_title, detail=False)`
  - `status_pill(text, *, kind="ok")`
- **`pages/_state.py`** — removed the `🔄` spinner glyph.
- **All 19 pages** — switch to `render_page_header()` for consistent
  page-title + subtitle chrome. Removed emoji from sub-headers, button
  labels, and ad-hoc `st.success("...")` strings.
- **`pages/dashboard.py`** — redesigned from scratch. Replaces the old
  3×2 readiness checkbox grid with a compact 6-cell status strip
  (`status_pill`). Adds a DB-backed KPI strip + a recent-cases table
  sourced directly from `database.repository.list_cases`. The
  pipeline-outputs expanders were dropped because they duplicated the
  Summary, Report, and Reconstruction pages. Per-session "active
  analysis" block retained for live session work.
- **`pages/ai_summary.py`, `pages/chat_assistant.py`,
  `pages/crime_timeline.py`, `pages/crime_prediction.py`,
  `pages/report.py`, `pages/reconstruction.py`** — top-level
  "No evidence analysis found" warnings converted to
  `empty_state("...", action_target="pages/evidence_analysis.py")`.
- **`pages/crime_scene_investigation.py`** — all sub-header and
  warning strings de-emoji'd; banner separations standardized.
- **`pages/system_status.py`** — removed the developer-only
  "Resilience helpers (live demo)" playground so the page no longer
  exposes internals in the academic demo. Now purely diagnostic.

### Tests

- 17 new layout-helper tests
  (`tests/test_layout_helpers.py`) cover icon constants, no-emoji
  invariant, empty_state card / link / plain / no-action,
  friendly_error mapping for every handled exception type + retry
  button, and status_pill rendering for ok / warn / danger.
- Full suite: **591 passed**, 1 pre-existing OCR fail unrelated to UI.
- `python scripts/smoke_app.py` → "All imports OK."

---

## Phase 41 — Demo-readiness & Repo Polish

Four small, independent deliverables. No architectural changes.
Closes the last four gaps between "functional" and "presentation-ready"
for a B.Tech final-year demo.

### Changes

- **`utils/ocr.py`** — broadened the `extract_text()` exception clause
  to also catch `TypeError` and `ValueError`. Previously,
  `pytesseract.image_to_string(None)` raised
  `TypeError("Unsupported image object")` from upstream, which slipped
  through the existing `except (ImportError, AttributeError, OSError,
  RuntimeError)` clause and failed the test. Now every failure —
  missing binary, malformed image, `None`, dtype mismatch — falls back
  to the deterministic stub. `extract_text()` truly never raises.
- **`LICENSE`** — new file at the repo root. MIT License, copyright
  "Vishnu Reddy" 2026. GitHub now auto-detects MIT and renders the
  license badge.
- **`scripts/render_demo_screenshots.py`** — new Pillow-based generator.
  Renders six labelled 1280×800 mockups (`dashboard.png`,
  `evidence_analysis.png`, `ai_summary.png`, `report.png`,
  `reconstruction.png`, `analytics.png`) into `docs/screenshots/`. Each
  mockup shows the real page title + subtitle + 4-tile KPI strip +
  body content + a yellow **"DEMO SCREENSHOT (mockup)"** ribbon so it
  cannot be mistaken for a live capture. Uses DejaVu/Segoe UI fonts
  with a graceful fallback to the PIL default.
- **`docs/screenshots/README.md`** — new file explaining that the PNGs
  are labelled mockups and pointing readers at the capture recipe.
- **`docs/USER_MANUAL.md`** — added a "Screenshots" section near the
  top (six images rendered from the mockups) and an "Appendix:
  screenshots" with a 5-line recipe for replacing the mockups with
  real `streamlit run app.py` captures.
- **`README.md`** — replaced all 38 `✅ done` entries in the features
  table with `[Done]`. Removes the only remaining emoji chrome in the
  README and matches the Phase 40 "no emoji chrome" decision.

### Tests

- Full suite: **592 passed**, 0 failed (was 591 + 1 OCR fail = 591).
- `python scripts/smoke_app.py` → "All imports OK."
- `python scripts/render_demo_screenshots.py` → six PNGs written.

---

## Phase 42 — CI coverage truth

### Context

The README badge claimed **98% line coverage**, and CI's
`pytest --cov=...` line listed
`app`, `core`, `services`, `models`, `utils`, `database`, `config`
— but **not** `pages`. The 19 Streamlit page modules were silently
excluded, so CI told green lies. Running the matrix with pages
included reveals the true number: **73%**.

This phase does not add or remove tests; it makes CI honest.

### Changes

- **`.github/workflows/ci.yml`** — added `--cov=pages` to the
  terminal-coverage step. Subsequent CI runs will report the
  full picture across every code directory.
- **`README.md`** — test-count badge updated from 551 → **592
  passed**, coverage badge updated from 98% → **73%**. The new
  numbers match what CI now prints on every push.

### Why we did not chase 90%+

`pages/` coverage is structurally low (12–43%) because every page is
`render()` that calls into ~30+ Streamlit APIs in sequence. Driving a
full Streamlit session from a unit test requires stubbing
`st.session_state`, `st.file_uploader`, `st.button`, every
`st.markdown` call, etc. — and the value of that work for a B.Tech
demo is marginal: the pages are thin renderers over well-tested
`models/`, `services/`, and `database/` modules, all of which sit at
96–100%. An optional Phase 43 could add a small
`tests/test_pages_render.py` that exercises each page's `render()`
through a shared MagicMock fixture.

### Verification

- `python -m pytest --cov=app --cov=core --cov=services --cov=models
  --cov=utils --cov=database --cov=config --cov=pages -q`
  → **592 passed, 73% coverage** (matches the badge).
- `python scripts/smoke_app.py` → "All imports OK."
- `python scripts/audit.py --json` → still clean.

---

## Phase 43 — README badge cleanup

The README contained two occurrences of a CI badge URL with literal
`<OWNER>/<REPO>` placeholders (lines 9 and 229). GitHub silently
404s those, so the rendered README displayed a broken image and a
"Replace `<OWNER>/<REPO>` in the badge URL after the first push"
note — a leftover from when the project was a template.

### Changes

- **`README.md`** — removed both `[!\\[CI\\]](https://github.com/<OWNER>/<REPO>/...)`
  badges. The remaining shields.io badges (Python version, tests,
  coverage, license) already convey project health and don't need an
  owner-specific URL.
- **`README.md`** — replaced the badge + "Replace `<OWNER>/<REPO>`"
  instruction in the Continuous Integration section with a short
  prose paragraph that points readers at `.github/workflows/ci.yml`
  for the workflow file itself.

### Verification

- `grep -n "<OWNER>\\|<REPO>" README.md` → no matches.
- README renders cleanly on GitHub with no broken images.

---

## Phase 44 — Evaluator demo runbook

The project is presentation-ready. This phase records the exact
five-minute demo flow so it lives in the repo, not just in chat.

### Pre-demo (the night before)

Run on the evaluator's laptop — or your own — so the demo doesn't
start with multi-minute model downloads:

```bash
git clone <your-repo-url>
cd Crime-Investigation-AI

# macOS / Linux
./scripts/run.sh

# Pre-cache the AI models (YOLO ~6 MB, FLAN-T5 ~990 MB).
# The first run of scripts/run.sh will create .venv and install
# requirements.txt; activate it before pre-caching:
source .venv/bin/activate
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM;
  AutoTokenizer.from_pretrained('google/flan-t5-base');
  AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"
deactivate
```

### Demo day

| Time | Page | Action |
|---|---|---|
| 0:00 | Dashboard | Show KPI strip + recent cases (real DB data) |
| 0:30 | Image Detection | Upload a sample image; run YOLO |
| 1:30 | Evidence Analysis | Show severity, person/weapon counts |
| 2:00 | AI Summary | Toggle FLAN-T5 on; click Generate |
| 2:30 | Report | Download PDF |
| 3:00 | Reconstruction | Generate 4-scene storyboard + MP4 |
| 4:00 | Case History | Show the case now persisted |
| 4:30 | Settings | Switch theme light → dark |

### What NOT to do

- Do not run `docker compose up` from scratch — adds ~3 minutes.
- Do not let "FLAN-T5 is loading" appear in the UI — pre-cache it.
- Do not upload a >5 MB video — first keyframe extraction will demo
  the spinner longer than you'd like.

### Verification

- `scripts/run.sh` / `scripts/run.bat` exist and exec correctly.
- `docker --version` available on the demo machine (optional).
- Python ≥ 3.11 on the demo machine.