"""
Whole-project import smoke tests (Milestone 10).

Catches "I renamed a module and forgot to update a page" early and
pins the current app version so every milestone that bumps the
version must update this test deliberately.
"""

from __future__ import annotations

import importlib
import pkgutil

import database
import models
import pages as pages_pkg
import utils
from config import APP_NAME, APP_VERSION


def _iter_modules(pkg) -> list[str]:
    return [m.name for m in pkgutil.iter_modules(pkg.__path__)]


def test_all_models_import() -> None:
    for name in _iter_modules(models):
        importlib.import_module(f"models.{name}")


def test_all_utils_import() -> None:
    for name in _iter_modules(utils):
        importlib.import_module(f"utils.{name}")


def test_all_pages_import() -> None:
    for name in _iter_modules(pages_pkg):
        importlib.import_module(f"pages.{name}")


def test_database_modules_import() -> None:
    for name in _iter_modules(database):
        importlib.import_module(f"database.{name}")


def test_config_exports_app_name_and_version() -> None:
    assert isinstance(APP_NAME, str) and APP_NAME
    assert isinstance(APP_VERSION, str) and APP_VERSION


def test_app_version_is_pinned_to_1_0_0() -> None:
    """Bump intentionally on each release; update this in lockstep."""
    assert APP_VERSION == "1.0.0"


def test_no_circular_imports_in_clean_order() -> None:
    """Force a deterministic import order; raises if any module cycles."""
    for mod in ("config", "database", "utils", "models"):
        importlib.import_module(mod)
