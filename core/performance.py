"""
Lightweight performance + system introspection helpers.

Pure stdlib (no psutil) so we don't add a runtime dep just to show
memory / CPU info in the **System Status** page. Falls back gracefully
when `resource` is unavailable (e.g. Windows native Python).

Public surface
--------------
- `SystemSnapshot` — frozen dataclass with the captured readings.
- `capture_snapshot()` — collect platform / Python / process info.
- `Stopwatch` — context manager for timing blocks (`with Stopwatch() as sw: ...`)
- `startup_marker()` — write a sentinel file with the current timestamp;
  used by the **System Status** page to report "uptime since".
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_STARTUP_FILE: Path = Path("logs") / ".startup"


@dataclass(frozen=True)
class SystemSnapshot:
    """A point-in-time view of the host process and platform."""

    captured_at: str
    python_version: str
    platform: str
    machine: str
    pid: int
    cwd: str
    rss_mb: float
    cpu_count: int
    has_resource_module: bool
    extras: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict (timestamps + floats)."""
        return asdict(self)


def _rss_mb() -> float:
    """Return RSS in MB; returns 0.0 if the platform doesn't expose it."""
    try:
        import resource  # POSIX-only

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return round(usage.ru_maxrss / 1024.0, 2)
    except (ImportError, OSError, ValueError):
        # Windows: try psutil if available, else 0.0.
        try:
            import psutil  # type: ignore[import-not-found]

            return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
        except (ImportError, AttributeError):
            return 0.0


def capture_snapshot() -> SystemSnapshot:
    """Collect a snapshot of platform + process info. Cheap to call."""
    extras: dict[str, str] = {}
    try:
        extras["hostname"] = platform.node() or "unknown"
    except Exception:  # pragma: no cover - defensive
        extras["hostname"] = "unknown"

    return SystemSnapshot(
        captured_at=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.split()[0],
        platform=platform.system(),
        machine=platform.machine(),
        pid=os.getpid(),
        cwd=str(Path.cwd()),
        rss_mb=_rss_mb(),
        cpu_count=os.cpu_count() or 1,
        has_resource_module="resource" in sys.modules,
        extras=extras,
    )


class Stopwatch:
    """Stopwatch instance. Use as `with Stopwatch(...) as sw: ...`."""

    __slots__ = ("label", "_start", "_end", "_elapsed_ms")

    def __init__(self, label: str = "block") -> None:
        self.label: str = label
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed_ms: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds spent inside the `with` block (0.0 if not yet stopped)."""
        return self._elapsed_ms

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._end = time.perf_counter()
        self._elapsed_ms = (self._end - self._start) * 1000.0

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Stopwatch(label={self.label!r}, elapsed_ms={self._elapsed_ms:.2f})"


def mark_startup(log_dir: Path | str = "logs") -> Path:
    """Write the current UTC timestamp into `<log_dir>/.startup`.

    Returns the path to the sentinel file. Idempotent — overwrites any
    previous marker so the System Status page always sees the latest
    process boot.
    """
    out = Path(log_dir) / ".startup"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return out


def read_startup(log_dir: Path | str = "logs") -> datetime | None:
    """Return the datetime written by `mark_startup`, or `None` if absent."""
    path = Path(log_dir) / ".startup"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        return datetime.fromisoformat(text)
    except (OSError, ValueError):
        return None


def uptime_seconds(log_dir: Path | str = "logs") -> float | None:
    """Return seconds since `mark_startup` was last called, or `None`."""
    start = read_startup(log_dir)
    if start is None:
        return None
    delta = datetime.now(timezone.utc) - start
    return max(delta.total_seconds(), 0.0)


# Re-export the startup-marker path so tests can locate the sentinel.
STARTUP_MARKER: Path = _STARTUP_FILE