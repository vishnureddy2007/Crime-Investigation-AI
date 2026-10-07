"""Tests for ``core.observability``."""

from __future__ import annotations

import pytest

from core import observability as obs


@pytest.fixture(autouse=True)
def _clear_logs() -> None:
    """Each test starts with a fresh buffer."""
    obs.reset()


def test_request_timer_records_success() -> None:
    with obs.RequestTimer("yolo") as ctx:
        result = sum(range(100))
    assert result == 4950
    assert ctx["success"] is True
    assert ctx["elapsed_ms"] >= 0.0
    snap = obs.request_log.snapshot()
    assert snap["counts"]["timings_total"] == 1
    assert snap["counts"]["errors_total"] == 0


def test_request_timer_records_failure() -> None:
    with pytest.raises(RuntimeError):
        with obs.RequestTimer("explode"):
            raise RuntimeError("boom")
    snap = obs.request_log.snapshot()
    assert snap["counts"]["timings_total"] == 1
    assert snap["counts"]["errors_total"] == 1


def test_timing_report_returns_recent_first() -> None:
    for i in range(5):
        with obs.RequestTimer(f"tag-{i}"):
            pass
    recent = obs.timing_report(limit=3)
    assert [r["tag"] for r in recent] == ["tag-4", "tag-3", "tag-2"]


def test_error_report_captures_message() -> None:
    try:
        with obs.RequestTimer("bad"):
            raise ValueError("nope")
    except ValueError:
        pass
    errs = obs.error_report()
    assert errs[0]["tag"] == "bad"
    assert "ValueError" in errs[0]["message"]


def test_reset_clears_buffers() -> None:
    with obs.RequestTimer("x"):
        pass
    obs.reset()
    snap = obs.request_log.snapshot()
    assert snap["counts"]["timings_total"] == 0
    assert snap["counts"]["errors_total"] == 0


def test_ring_buffer_is_bounded() -> None:
    for i in range(obs.MAX_TIMINGS + 50):
        with obs.RequestTimer(f"t-{i}"):
            pass
    snap = obs.request_log.snapshot()
    assert snap["counts"]["timings_total"] == obs.MAX_TIMINGS