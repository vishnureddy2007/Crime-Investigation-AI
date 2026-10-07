"""
Performance Profiler & Developer Diagnostics Module.

Instruments the complete investigation pipeline (YOLO loading/inference,
evidence analysis, database operations, Ollama model connection/inference,
and rendering) and measures execution time, CPU usage, RAM RSS, and VRAM.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StageMetric:
    name: str
    duration_sec: float
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    vram_mb: float = 0.0
    details: str = ""


class PipelineProfiler:
    """Singleton profiler tracking stage execution metrics."""

    _instance: PipelineProfiler | None = None

    def __init__(self) -> None:
        self.metrics: list[StageMetric] = []
        self.start_time: float = time.perf_counter()

    @classmethod
    def get_instance(cls) -> PipelineProfiler:
        if cls._instance is None:
            cls._instance = PipelineProfiler()
        return cls._instance

    def record_stage(
        self,
        name: str,
        duration_sec: float,
        details: str = "",
        vram_mb: float = 0.0,
    ) -> StageMetric:
        ram_mb = self._get_ram_mb()
        cpu_pct = self._get_cpu_percent()
        metric = StageMetric(
            name=name,
            duration_sec=round(duration_sec, 3),
            cpu_percent=cpu_pct,
            ram_mb=ram_mb,
            vram_mb=vram_mb,
            details=details,
        )
        self.metrics.append(metric)
        return metric

    @staticmethod
    def _get_ram_mb() -> float:
        try:
            import psutil  # type: ignore
            return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
        except (ImportError, AttributeError, OSError):
            return 0.0

    @staticmethod
    def _get_cpu_percent() -> float:
        try:
            import psutil  # type: ignore
            return round(psutil.cpu_percent(interval=None), 1)
        except (ImportError, AttributeError, OSError):
            return 0.0

    def summary(self) -> dict[str, Any]:
        total_duration = sum(m.duration_sec for m in self.metrics)
        sorted_metrics = sorted(self.metrics, key=lambda m: m.duration_sec, reverse=True)
        top_bottlenecks = [
            {
                "stage": m.name,
                "duration_sec": m.duration_sec,
                "ram_mb": m.ram_mb,
                "details": m.details,
            }
            for m in sorted_metrics[:3]
        ]
        return {
            "total_duration_sec": round(total_duration, 2),
            "stage_count": len(self.metrics),
            "top_bottlenecks": top_bottlenecks,
            "metrics": [m.__dict__ for m in self.metrics],
        }

    def format_log_report(self) -> str:
        s = self.summary()
        lines = [
            "=============================",
            "PERFORMANCE DIAGNOSTIC REPORT",
            "=============================",
            f"Total Duration: {s['total_duration_sec']} sec | Stages Tracked: {s['stage_count']}",
            "",
            "Stage Breakdown:",
        ]
        for m in self.metrics:
            lines.append(
                f"  - {m.name:<30}: {m.duration_sec:>6.2f} sec | RAM: {m.ram_mb:>7.1f} MB | {m.details}"
            )
        lines.append("")
        lines.append("Top Bottlenecks:")
        for idx, b in enumerate(s["top_bottlenecks"], start=1):
            lines.append(f"  #{idx} {b['stage']}: {b['duration_sec']:.2f} sec ({b['details']})")
        lines.append("=============================")
        return "\n".join(lines)


# Global helper functions
def profile_stage(name: str, duration_sec: float, details: str = "") -> StageMetric:
    """Record a stage metric in the global pipeline profiler."""
    return PipelineProfiler.get_instance().record_stage(name, duration_sec, details)


def get_performance_report() -> str:
    """Return a formatted performance report string."""
    return PipelineProfiler.get_instance().format_log_report()
