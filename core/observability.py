"""
Lightweight observability primitives — request timings + error log.

Pure stdlib. Provides:

- :class:`RequestTimer` — context manager that records elapsed-ms and
  a tag into an in-memory ring buffer.
- :class:`ErrorRing` — bounded ring buffer of the most recent
  exceptions (timestamp, tag, message).
- :func:`request_log` — module-level singleton + :func:`timing_report`
  returning a serialisable snapshot of recent timings.

Used by the **System Status** page to render the live request trace.
Not thread-safe — Streamlit is single-threaded per session, so a
``list`` + ``deque`` is sufficient.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterator

MAX_TIMINGS: int = 100
MAX_ERRORS: int = 50


@dataclass
class TimingRecord:
    """A single captured request timing."""

    tag: str
    elapsed_ms: float
    captured_at: str
    success: bool
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ErrorRecord:
    """A single captured error."""

    tag: str
    message: str
    captured_at: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RequestLog:
    """Singleton-ish in-memory log of timings + errors."""

    timings: deque[TimingRecord] = field(default_factory=lambda: deque(maxlen=MAX_TIMINGS))
    errors: deque[ErrorRecord] = field(default_factory=lambda: deque(maxlen=MAX_ERRORS))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, record: TimingRecord) -> None:
        with self._lock:
            self.timings.append(record)

    def record_error(self, record: ErrorRecord) -> None:
        with self._lock:
            self.errors.append(record)

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe snapshot of the buffers."""
        with self._lock:
            return {
                "timings": [r.as_dict() for r in self.timings],
                "errors":  [e.as_dict() for e in self.errors],
                "counts": {
                    "timings_total": len(self.timings),
                    "errors_total":  len(self.errors),
                },
            }

    def clear(self) -> None:
        with self._lock:
            self.timings.clear()
            self.errors.clear()


# Default shared log used by helpers below.
request_log: RequestLog = RequestLog()


@contextmanager
def RequestTimer(tag: str, *, detail: str = "") -> Iterator[dict[str, object]]:
    """Time the wrapped block; record success/failure into ``request_log``.

    Example:
        with RequestTimer("yolo.detect") as ctx:
            detections = yolo.detect(image)
        print(ctx["elapsed_ms"])
    """
    record_box: dict[str, object] = {"elapsed_ms": 0.0, "success": True}
    started = time.perf_counter()
    try:
        yield record_box
    except Exception as exc:
        record_box["success"] = False
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        request_log.record(
            TimingRecord(
                tag=tag,
                elapsed_ms=elapsed_ms,
                captured_at=_now_iso(),
                success=False,
                detail=f"{exc.__class__.__name__}: {exc}"[:200],
            )
        )
        request_log.record_error(
            ErrorRecord(
                tag=tag,
                message=f"{exc.__class__.__name__}: {exc}"[:200],
                captured_at=_now_iso(),
            )
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record_box["elapsed_ms"] = elapsed_ms
        request_log.record(
            TimingRecord(
                tag=tag,
                elapsed_ms=elapsed_ms,
                captured_at=_now_iso(),
                success=True,
                detail=detail,
            )
        )


def timing_report(*, limit: int = 20) -> list[dict[str, object]]:
    """Return the most-recent ``limit`` timings, newest first."""
    snap = request_log.snapshot()
    items = list(snap["timings"])[-limit:][::-1]
    return items


def error_report(*, limit: int = 20) -> list[dict[str, object]]:
    """Return the most-recent ``limit`` errors, newest first."""
    snap = request_log.snapshot()
    items = list(snap["errors"])[-limit:][::-1]
    return items


def reset() -> None:
    """Clear all captured records."""
    request_log.clear()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()