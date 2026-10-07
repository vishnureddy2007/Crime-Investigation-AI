# Architecture

This document describes the runtime architecture of the
**AI-Based Crime Investigation Assistant** (Phase 13 release).

## High-level diagram

```mermaid
flowchart LR
    subgraph Browser
        UI[Streamlit UI]
    end

    subgraph App[app.py + pages/]
        ROUTER[st.sidebar.radio router]
        P_HOME[Home]
        P_DASH[Dashboard]
        P_IMG[Image Detection]
        P_VID[Video Processing]
        P_EVD[Evidence Analysis]
        P_SUM[AI Summary]
        P_CHAT[Chat Assistant]
        P_TIME[Crime Timeline]
        P_PRED[Crime Prediction]
        P_RPT[Report]
        P_REC[Reconstruction]
        P_CASE[Case History]
        P_SET[Settings]
        P_HLP[Help]
        P_CON[Contact]
        P_ABT[About]
    end

    subgraph Services[services/]
        S_CHAT[ChatAssistant]
        S_TIME[build_timeline]
        S_PRED[predict_risk]
    end

    subgraph Models[models/]
        M_YOLO[YOLODetector]
        M_VID[VideoProcessor]
        M_EVD[EvidenceAnalyzer]
        M_SUM[SummaryGenerator]
        M_RPT[PDF + DOCX Generators]
        M_REC[ScenePlanner + Reconstruction]
        M_DASH[Dashboard helpers]
    end

    subgraph Core[core/]
        C_SEC[security\npath-traversal, sanitisation]
        C_LOG[logging]
        C_THM[theming\nlight/dark CSS]
    end

    subgraph Utils[utils/]
        U_IMG[image_io]
        U_VID[video_io]
        U_FRM[frame_utils]
        U_VIZ[visualization]
        U_OCR[ocr]
        U_FACE[face_detection]
        U_DB[db_hooks]
    end

    subgraph Data
        DB[(SQLite\ncases / analyses /\nsummaries / reports /\nstoryboards / feedback)]
        OUT[outputs/\nreports, keyframes,\nstoryboard, videos]
    end

    UI --> ROUTER
    ROUTER --> P_HOME & P_DASH & P_IMG & P_VID & P_EVD & P_SUM & P_CHAT & P_TIME & P_PRED & P_RPT & P_REC & P_CASE & P_SET & P_HLP & P_CON & P_ABT

    P_IMG --> M_YOLO
    P_VID --> M_VID
    P_EVD --> M_EVD
    P_SUM --> M_SUM
    P_CHAT --> S_CHAT
    P_TIME --> S_TIME
    P_PRED --> S_PRED
    P_RPT --> M_RPT
    P_REC --> M_REC
    P_DASH --> M_DASH

    M_YOLO --> U_VIZ
    M_VID --> U_FRM
    M_SUM --> M_EVD
    M_RPT --> M_EVD
    M_REC --> M_EVD

    S_CHAT --> M_EVD
    S_TIME --> M_EVD
    S_PRED --> M_EVD

    P_IMG --> U_IMG
    P_VID --> U_VID
    P_RPT --> U_OCR
    P_DASH --> U_FACE

    U_DB --> DB
    P_RPT --> DB
    P_REC --> OUT
    M_RPT --> OUT
    M_VID --> OUT
```

## Request lifecycle — “image uploaded → report downloaded”

1. The user picks **Image Detection** in the sidebar.
2. `pages/image_detection.py` validates the upload with
   `utils.image_io.load_image_from_bytes()`.
3. The shared `YOLODetector` singleton from
   `pages/_state.get_yolo_detector()` runs the model and returns
   a `DetectionResult`.
4. The page writes `last_detection` into `st.session_state` and
   the auto-save hook (`utils/db_hooks.auto_save_last_analysis()`)
   persists a row in `analyses` and (if needed) `cases`.
5. The user proceeds to **Evidence Analysis** →
   `models.evidence_analyzer.analyze()` produces an
   `EvidenceAnalysis` from the detection.
6. **AI Summary** → `models.summary_generator.SummaryGenerator`
   builds the deterministic template and (optionally) rewrites it
   with FLAN-T5.
7. **Report** → `models.report_generator.build_report_data()`
   + `PDFReportGenerator` / `DOCXReportGenerator` produce the file
   and `utils/db_hooks.auto_save_last_report()` persists the row.

Every cross-layer contract is a frozen dataclass in
`models/schemas.py`, so no module reaches into another module's
internals.

## Layer responsibilities

| Layer | Responsibility | Streamlit-aware? |
|---|---|---|
| `app/` | HTTP façade (no UI use) | no |
| `core/` | logging, security, theming | no |
| `services/` | orchestration, business flows | no |
| `models/` | detection / generation / analysis | no |
| `utils/` | I/O + small helpers | no |
| `database/` | SQLite repository | no |
| `pages/` | Streamlit UI per feature | yes |
| `app.py` | bootstrap + routing | yes |

The “Streamlit-aware” column is the contract: everything in
`pages/` may import `streamlit`; nothing else may.

## Configuration

Every value in `config/settings.py` either:

- has a constant default, **or**
- is read from an environment variable with a safe fallback
  (e.g. `APP_PORT`, `APP_THEME`, `LOG_LEVEL`).

`config/__init__.py` re-exports every public name so callers
import `from config import APP_NAME` instead of reaching into the
sub-module.

## Security posture

- `core.security.safe_filename` strips path components and unsafe
  characters from every user-supplied filename.
- `core.security.safe_join` is the only function allowed to combine
  a server-side directory with a user-supplied filename; it raises
  `PathTraversalError` on escapes.
- Every SQL statement uses parameterised queries; no string
  concatenation reaches the database.
- File uploads are size- and extension-checked before they hit disk
  (`utils.image_io`, `utils.video_io`).
- All free-text input is length-bounded and stripped of control
  characters (`core.security.validate_text_input`).

## Performance notes

- Heavy models (YOLOv8, FLAN-T5) are loaded **once** per session
  via Streamlit’s `cache_resource` pattern (see
  `pages/_state.get_yolo_detector()` and
  `pages/ai_summary._get_generator()`).
- `core.theming` keeps a single string literal per theme and
  injects it once per page render — no per-element style
  recomputation.
- `app.api.dispatch` is a small constant-time registry; adding a
  new endpoint never regresses existing ones.

## Where to look for a feature

| If you’re looking for… | Open this file |
|---|---|
| A new model wrapper | `models/` |
| A new business rule | `services/` |
| A new sidebar page | `pages/` (mirror `pages/about.py`) |
| A new env var | `config/settings.py` |
| A new DB table | `database/db.py` (DDL) + `database/repository.py` (CRUD) |
| A new theme token | `core/theming.py` |
| A new HTTP route | `app/__init__.py` (`@register(...)`) |
