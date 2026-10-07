"""
Tests for the deployment-time env-var overrides in config/settings.py.

These cover Milestone 12's "optional env vars with config defaults"
contract:

    APP_PORT     -> int   (default 8501)
    APP_ADDRESS  -> str   (default "localhost")
    APP_HEADLESS -> bool  (default False)
    APP_THEME    -> str   (default "light")
    HF_HOME      -> Path  (default ~/.cache/huggingface)

config.settings reads these at *import time*, so we use
importlib.reload after monkey-patching the environment. A snapshot/
restore block at the start of each test keeps them hermetic.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

import config.settings as cfg


# Snapshot the current environment so each test can mutate freely and
# restore on teardown without leaking to siblings.
@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("APP_PORT", "APP_ADDRESS", "APP_HEADLESS", "APP_THEME", "HF_HOME"):
        monkeypatch.delenv(key, raising=False)


def _reload_settings() -> None:
    """Reload config.settings so module-level reads re-run."""
    importlib.reload(cfg)


# ----------------------------------------------------------------------
# Default behaviour (no env vars set)
# ----------------------------------------------------------------------
class TestDefaults:
    def test_app_port_default(self) -> None:
        _reload_settings()
        assert cfg.APP_PORT == 8501

    def test_app_address_default(self) -> None:
        _reload_settings()
        assert cfg.APP_ADDRESS == "localhost"

    def test_app_headless_default(self) -> None:
        _reload_settings()
        assert cfg.APP_HEADLESS is False

    def test_app_theme_default(self) -> None:
        _reload_settings()
        assert cfg.APP_THEME == "light"

    def test_hf_home_default_points_under_user_cache(self) -> None:
        _reload_settings()
        # Default is ~/.cache/huggingface (we check the suffix).
        assert cfg.HF_HOME == Path.home() / ".cache" / "huggingface"
        assert "huggingface" in str(cfg.HF_HOME).lower()


# ----------------------------------------------------------------------
# Explicit overrides
# ----------------------------------------------------------------------
class TestOverrides:
    def test_app_port_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_PORT", "9000")
        _reload_settings()
        assert cfg.APP_PORT == 9000
        assert isinstance(cfg.APP_PORT, int)

    def test_app_address_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ADDRESS", "0.0.0.0")
        _reload_settings()
        assert cfg.APP_ADDRESS == "0.0.0.0"

    def test_app_headless_truthy_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for truthy in ("1", "true", "True", "TRUE", "yes", "on"):
            monkeypatch.setenv("APP_HEADLESS", truthy)
            _reload_settings()
            assert cfg.APP_HEADLESS is True, f"APP_HEADLESS={truthy!r} should be True"

    def test_app_headless_falsy_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for falsy in ("0", "false", "False", "no", "off", "anything-else"):
            monkeypatch.setenv("APP_HEADLESS", falsy)
            _reload_settings()
            assert cfg.APP_HEADLESS is False, f"APP_HEADLESS={falsy!r} should be False"

    def test_app_theme_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_THEME", "dark")
        _reload_settings()
        assert cfg.APP_THEME == "dark"

    def test_hf_home_override_to_explicit_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        custom = tmp_path / "my_hf_cache"
        monkeypatch.setenv("HF_HOME", str(custom))
        _reload_settings()
        assert cfg.HF_HOME == Path(str(custom))


# ----------------------------------------------------------------------
# Robustness: malformed inputs must fall back (or raise predictably),
# never silently produce a wrong type
# ----------------------------------------------------------------------
class TestRobustness:
    def test_app_port_invalid_string_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_PORT", "not-a-number")
        # The contract is documented as int; an invalid string cannot be
        # silently converted. ValueError is the acceptable failure mode.
        with pytest.raises(ValueError):
            _reload_settings()


# ----------------------------------------------------------------------
# Cross-test cleanliness
# ----------------------------------------------------------------------
def test_module_reloads_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting an env var, reloading, and reloading back must restore."""
    monkeypatch.setenv("APP_PORT", "7777")
    _reload_settings()
    assert cfg.APP_PORT == 7777

    monkeypatch.delenv("APP_PORT", raising=False)
    _reload_settings()
    assert cfg.APP_PORT == 8501
