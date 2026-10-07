"""Unit tests for `services/crime_timeline.py`."""
from __future__ import annotations

from datetime import datetime

import pytest

from models.schemas import EvidenceAnalysis
from services.crime_timeline import TimelineEvent, build_timeline


def _a(**kw) -> EvidenceAnalysis:
    base = dict(
        source_name="c.png",
        source_type="image",
        counts_by_label={"person": 1, "knife": 1, "bag": 1, "vehicle": 1},
        total_objects=4,
        unique_labels=["person", "knife", "bag", "vehicle"],
        average_confidence=0.9,
        person_count=1,
        verified_weapon_count=1,
        candidate_weapon_count=0,
        weapon_count=1,
        vehicle_count=1,
        bag_count=1,
        severity_score=80,
        severity_level="high",
        suggested_category="robbery",
        key_observations=[],
        has_threat=True,
        frame_count=1,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )
    base.update(kw)
    return EvidenceAnalysis(**base)


class TestTimelineEvent:
    def test_as_dict(self) -> None:
        e = TimelineEvent(
            label="x", detail="d",
            timestamp=datetime(2026, 1, 1), severity="info",
        )
        d = e.as_dict()
        assert d["label"] == "x"
        assert d["severity"] == "info"
        assert "2026-01-01" in d["timestamp"]


class TestBuildTimeline:
    def test_empty_input_returns_empty(self) -> None:
        tl = build_timeline(None)  # type: ignore[arg-type]
        assert tl.events == []
        assert tl.source_name == ""

    def test_basic_event_count(self) -> None:
        tl = build_timeline(_a())
        # always-on: acquired, detection, severity (3)
        # present: weapon, vehicle, bag (3)
        # total: 6
        assert tl.count == 6

    def test_no_objects_means_only_basics(self) -> None:
        tl = build_timeline(_a(
            weapon_count=0, vehicle_count=0, bag_count=0,
            counts_by_label={}, total_objects=0, has_threat=False,
            severity_score=5, severity_level="low",
        ))
        # 2 always-on: acquired + detection + severity
        assert tl.count == 3

    def test_events_are_ordered(self) -> None:
        tl = build_timeline(_a())
        timestamps = [e.timestamp for e in tl.events]
        assert timestamps == sorted(timestamps)

    def test_summary_event_added_when_text(self) -> None:
        tl = build_timeline(_a(), summary_text="Investigation summary text.")
        assert any(e.label == "Summary recorded" for e in tl.events)

    def test_weapon_event_severity(self) -> None:
        tl = build_timeline(_a(weapon_count=2))
        weapon_events = [e for e in tl.events if "Weapon" in e.label]
        assert weapon_events
        assert weapon_events[0].severity == "critical"

    def test_as_dict(self) -> None:
        tl = build_timeline(_a())
        d = tl.as_dict()
        assert d["source_name"] == "c.png"
        assert d["count"] == tl.count
        assert isinstance(d["events"], list)