"""Build an ordered timeline of investigative events from an analysis.

`build_timeline()` is a deterministic function that derives a list
of `TimelineEvent` objects from the latest evidence analysis. The
order is:

  1. evidence acquired (always)
  2. detection run (always)
  3. weapon observed (if present)
  4. vehicle observed (if present)
  5. bag observed (if present)
  6. analysis timestamp (always, last)

This shape is enough to power a UI timeline and a JSON dump for the
case-history API.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from models.schemas import EvidenceAnalysis


@dataclass(frozen=True)
class TimelineEvent:
    """A single point in the investigative timeline."""

    label: str
    detail: str
    timestamp: datetime
    severity: str = "info"  # "info" | "warning" | "critical"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label":     self.label,
            "detail":    self.detail,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "severity":  self.severity,
        }


@dataclass(frozen=True)
class Timeline:
    """The full timeline for a single investigation."""

    source_name: str
    events: list[TimelineEvent]

    @property
    def count(self) -> int:
        return len(self.events)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "count":       self.count,
            "events":      [e.as_dict() for e in self.events],
        }


def build_timeline(
    analysis: EvidenceAnalysis,
    summary_text: str | None = None,
) -> Timeline:
    """Construct a Timeline from `analysis` (and optional summary text).

    Events are anchored to `analysis.timestamp` and spaced at 1-minute
    intervals so they always render in a stable order regardless of the
    wall-clock at construction time.
    """
    if analysis is None:
        return Timeline(source_name="", events=[])

    base = analysis.timestamp
    events: list[TimelineEvent] = []

    # 1. Evidence acquired
    events.append(TimelineEvent(
        label="Evidence acquired",
        detail=f"{analysis.source_type.capitalize()} '{analysis.source_name}' ingested.",
        timestamp=base,
    ))

    # 2. Detection run
    events.append(TimelineEvent(
        label="Detection completed",
        detail=(
            f"{analysis.total_objects} crime-relevant object(s) across "
            f"{analysis.frame_count} frame(s); "
            f"avg confidence {analysis.average_confidence:.0%}."
        ),
        timestamp=base + timedelta(minutes=1),
    ))

    # 3. Weapon
    if analysis.weapon_count > 0:
        events.append(TimelineEvent(
            label="⚠️ Weapon detected",
            detail=f"{analysis.weapon_count} weapon(s) present — threat level raised.",
            timestamp=base + timedelta(minutes=2),
            severity="critical" if analysis.weapon_count > 1 else "warning",
        ))

    # 4. Vehicle
    if analysis.vehicle_count > 0:
        events.append(TimelineEvent(
            label="Vehicle observed",
            detail=f"{analysis.vehicle_count} vehicle(s) detected.",
            timestamp=base + timedelta(minutes=3),
        ))

    # 5. Bag
    if analysis.bag_count > 0:
        events.append(TimelineEvent(
            label="Bag(s) detected",
            detail=f"{analysis.bag_count} bag(s) — possible theft indicator.",
            timestamp=base + timedelta(minutes=4),
        ))

    # 6. Summary (optional)
    if summary_text:
        snippet = summary_text if len(summary_text) <= 140 else summary_text[:137] + "..."
        events.append(TimelineEvent(
            label="Summary recorded",
            detail=snippet,
            timestamp=base + timedelta(minutes=5),
        ))

    # 7. Severity band (always last)
    events.append(TimelineEvent(
        label=f"Severity: {analysis.severity_level.upper()}",
        detail=(
            f"Score {analysis.severity_score}/100. "
            f"Suggested category: {analysis.suggested_category.replace('_', ' ')}."
        ),
        timestamp=base + timedelta(minutes=6),
        severity="critical" if analysis.severity_level == "critical"
                  else "warning" if analysis.severity_level in ("high", "moderate")
                  else "info",
    ))

    return Timeline(source_name=analysis.source_name, events=events)