"""Tests for ``core.performance`` — SystemSnapshot, Stopwatch, startup marker."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.performance import (
    Stopwatch,
    SystemSnapshot,
    capture_snapshot,
    mark_startup,
    read_startup,
    uptime_seconds,
)


def test_capture_snapshot_returns_dataclass() -> None:
    snap = capture_snapshot()
    assert isinstance(snap, SystemSnapshot)
    assert snap.python_version.count(".") >= 2
    assert snap.pid > 0
    assert snap.cpu_count >= 1
    assert snap.captured_at.endswith("+00:00") or "T" in snap.captured_at


def test_capture_snapshot_as_dict_is_json_safe() -> None:
    snap = capture_snapshot()
    d = snap.as_dict()
    assert d["python_version"] == snap.python_version
    assert d["platform"] == snap.platform
    assert d["extras"]["hostname"]


def test_stopwatch_measures_elapsed() -> None:
    with Stopwatch("sleep") as sw:
        time.sleep(0.05)
    assert sw.elapsed_ms >= 40.0
    assert sw.label == "sleep"


def test_stopwatch_zero_before_exit() -> None:
    sw = Stopwatch("noop")
    assert sw.elapsed_ms == 0.0
    with sw:
        pass
    assert sw.elapsed_ms >= 0.0


def test_mark_and_read_startup_roundtrip(tmp_path: Path) -> None:
    marker = mark_startup(tmp_path)
    assert marker.exists()
    parsed = read_startup(tmp_path)
    assert parsed is not None


def test_read_startup_missing_returns_none(tmp_path: Path) -> None:
    assert read_startup(tmp_path / "nope") is None


def test_uptime_seconds_returns_value(tmp_path: Path) -> None:
    mark_startup(tmp_path)
    secs = uptime_seconds(tmp_path)
    assert secs is not None
    assert secs >= 0.0


def test_uptime_seconds_missing_returns_none(tmp_path: Path) -> None:
    assert uptime_seconds(tmp_path / "missing") is None


def test_stopwatch_repr_is_friendly() -> None:
    sw = Stopwatch("demo")
    with sw:
        pass
    text = repr(sw)
    assert "Stopwatch" in text
    assert "demo" in text

# ----------------------------------------------------------------------
# Phase 19 — defensive coverage
# ----------------------------------------------------------------------


def test_rss_mb_falls_back_when_resource_unavailable(monkeypatch) -> None:
    """If `resource.getrusage` raises, _rss_mb tries psutil then 0.0."""
    import core.performance as perf
    import builtins

    # Force `resource.getrusage` to raise ImportError on Windows path.
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("forced: no resource module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Also ensure psutil import fails (it likely isn't installed).
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)

    # Even if both paths fail, _rss_mb returns 0.0.
    assert perf._rss_mb() == 0.0


def test_read_startup_returns_none_on_corrupt_file(tmp_path: Path) -> None:
    """A startup marker with garbage content returns None."""
    marker = tmp_path / ".startup"
    marker.write_text("not a date", encoding="utf-8")
    assert read_startup(tmp_path) is None
