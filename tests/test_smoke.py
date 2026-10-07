"""
Smoke test for Milestone 1.

Verifies the basic project structure loads without errors.
"""

from __future__ import annotations

from config.settings import (
    APP_NAME,
    APP_VERSION,
    PROJECT_ROOT,
    REPORTS_DIR,
    STORYBOARD_DIR,
    VIDEOS_DIR,
)


def test_project_root_exists() -> None:
    """The project root directory should exist on disk."""
    assert PROJECT_ROOT.exists(), f"Missing project root: {PROJECT_ROOT}"
    assert PROJECT_ROOT.is_dir()


def test_app_metadata_loaded() -> None:
    """Configuration values should be non-empty strings."""
    assert isinstance(APP_NAME, str) and APP_NAME
    assert isinstance(APP_VERSION, str) and APP_VERSION


def test_required_folders_exist() -> None:
    """All Milestone 1 folders should exist."""
    for folder in [
        "config", "models", "utils", "database",
        "outputs", "images", "tests", "docs",
    ]:
        path = PROJECT_ROOT / folder
        assert path.exists(), f"Missing folder: {folder}"
        assert path.is_dir(), f"Not a directory: {folder}"


def test_output_subdirs_exist() -> None:
    """Output subdirectories should exist (even if empty)."""
    for sub in (REPORTS_DIR, STORYBOARD_DIR, VIDEOS_DIR):
        assert sub.exists(), f"Missing output dir: {sub}"
        assert sub.is_dir(), f"Not a directory: {sub}"
