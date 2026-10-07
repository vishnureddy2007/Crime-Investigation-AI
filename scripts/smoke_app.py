"""
Import-time smoke check.

Imports every project package in dependency order and prints one
``OK <module>`` line per success. Used by CI / Docker / manual sanity
checks to catch broken imports without launching the Streamlit
server.

Usage:
    python scripts/smoke_app.py

Exit codes:
    0 -> all imports succeeded
    1 -> at least one import raised
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# Ensure the repo root is on sys.path when invoked as
# `python scripts/smoke_app.py` so the project packages resolve
# without a `pip install -e .` step.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Order matters: leaf packages first, then parents, then pages.
# pages/* import streamlit at module scope, which is fine when run
# from outside Streamlit (they only USE st inside render()).
_IMPORT_ORDER: tuple[str, ...] = (
    "config",
    "config.settings",
    "database",
    "database.db",
    "database.repository",
    "core",
    "core.logging",
    "core.security",
    "core.theming",
    "core.performance",
    "core.resilience",
    "services",
    "services.chat_assistant",
    "services.crime_timeline",
    "services.crime_prediction",
    "services.analytics",
    "utils",
    "utils.image_io",
    "utils.video_io",
    "utils.frame_utils",
    "utils.visualization",
    "utils.db_hooks",
    "utils.ocr",
    "utils.face_detection",
    "utils.i18n",
    "utils.export",
    "models",
    "models.schemas",
    "models.yolo_detector",
    "models.video_processor",
    "models.evidence_analyzer",
    "models.summary_generator",
    "models.report_generator",
    "models.scene_planner",
    "models.reconstruction",
    "models.dashboard",
    "app",
    "app.api",
)


def main() -> int:
    failures: list[tuple[str, str]] = []
    for name in _IMPORT_ORDER:
        try:
            importlib.import_module(name)
            print(f"OK  {name}")
        except Exception as exc:  # pragma: no cover - smoke harness
            failures.append((name, f"{exc.__class__.__name__}: {exc}"))
            print(f"FAIL  {name}: {exc}", file=sys.stderr)

    # Optional: also try the page modules (these touch streamlit).
    print("\n(streamlit-aware pages; warnings are expected if not launched)")
    for page in (
        "pages.home",
        "pages.dashboard",
        "pages.about",
        "pages.chat_assistant",
        "pages.crime_timeline",
        "pages.crime_prediction",
        "pages.settings",
        "pages.help",
        "pages.contact",
        "pages.analytics",
        "pages.system_status",
    ):
        try:
            importlib.import_module(page)
            print(f"OK  {page}")
        except Exception as exc:  # pragma: no cover
            failures.append((page, f"{exc.__class__.__name__}: {exc}"))
            print(f"FAIL  {page}: {exc}", file=sys.stderr)

    if failures:
        print(
            f"\n{len(failures)} import(s) failed:\n"
            + "\n".join(f"  - {n}: {e}" for n, e in failures)
        )
        return 1

    print("\nAll imports OK.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
