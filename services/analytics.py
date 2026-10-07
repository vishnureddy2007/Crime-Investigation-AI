"""
Aggregated analytics over the investigations database.

Pure-Python module — no Streamlit imports — so the **Analytics** page
can call into a tested, pure function and just render the result.

Public surface
--------------
- :class:`SeverityBand` — band label + inclusive min / exclusive max.
- :func:`severity_band` — map a numeric severity to its band.
- :class:`AnalyticsSnapshot` — KPIs + category / severity distributions.
- :func:`compute_snapshot` — run all queries and assemble the snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from database.db import get_connection


def _load_severity_bands() -> tuple[tuple[str, int, int], ...]:
    """Read the canonical severity bands from config.

    Builds a tuple of ``(label, inclusive_min, exclusive_max)`` from
    ascending thresholds so the **next** threshold becomes the
    exclusive max of the previous one. The final (highest) band gets
    an exclusive max of 101 so a score of 100 always resolves. Falls
    back to a safe default if config is unavailable so the helper
    still works under stripped-down test fixtures.
    """
    try:
        from config import SEVERITY_THRESHOLDS  # type: ignore[import-not-found]
    except (ImportError, AttributeError):
        SEVERITY_THRESHOLDS = [
            (75, "critical"),
            (50, "high"),
            (25, "moderate"),
            (0,  "low"),
        ]

    # Sort ASCENDING by min_score so threshold[i+1] is the exclusive max.
    sorted_thresholds = sorted(SEVERITY_THRESHOLDS, key=lambda t: t[0])
    bands: list[tuple[str, int, int]] = []
    for idx, (min_score, label) in enumerate(sorted_thresholds):
        if idx + 1 < len(sorted_thresholds):
            max_score = sorted_thresholds[idx + 1][0]
        else:
            max_score = 101  # exclusive upper bound for the highest band
        bands.append((label, int(min_score), int(max_score)))
    return tuple(bands)


# Severity thresholds (inclusive min, exclusive max).
SEVERITY_BANDS: tuple[tuple[str, int, int], ...] = _load_severity_bands()


@dataclass(frozen=True)
class SeverityBand:
    """Inclusive min / exclusive max band definition."""

    label: str
    min_score: int
    max_score: int

    def contains(self, score: int) -> bool:
        return self.min_score <= score < self.max_score


def severity_band(score: int) -> str:
    """Return the band label for ``score``.

    Uses the same thresholds as ``config.SEVERITY_THRESHOLDS`` so
    band labels agree across the analyzer, dashboard, and analytics
    page.
    """
    for label, lo, hi in SEVERITY_BANDS:
        if lo <= score < hi:
            return label
    return SEVERITY_BANDS[0][0] if SEVERITY_BANDS else "critical"


def _band_index(label: str) -> int:
    """Return the (descending) rank of ``label`` for sorting."""
    for idx, (band_label, _lo, _hi) in enumerate(SEVERITY_BANDS):
        if band_label == label:
            return idx
    return len(SEVERITY_BANDS)


@dataclass
class AnalyticsSnapshot:
    """Aggregated KPIs across every persisted case."""

    total_cases: int = 0
    total_analyses: int = 0
    total_reports: int = 0
    total_feedback: int = 0
    avg_severity: float = 0.0
    threats: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    recent_cases: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (dataclasses are JSON-friendly)."""
        return asdict(self)


def compute_snapshot(db_path: Path, *, recent_limit: int = 5) -> AnalyticsSnapshot:
    """Run all queries and assemble an :class:`AnalyticsSnapshot`."""
    snap = AnalyticsSnapshot()
    if not db_path.exists():
        return snap

    with get_connection(db_path) as conn:
        snap.total_cases = int(
            conn.execute("SELECT COUNT(*) AS n FROM cases").fetchone()["n"]
        )
        snap.total_analyses = int(
            conn.execute("SELECT COUNT(*) AS n FROM analyses").fetchone()["n"]
        )
        snap.total_reports = int(
            conn.execute("SELECT COUNT(*) AS n FROM reports").fetchone()["n"]
        )
        snap.total_feedback = int(
            conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
        )
        avg_row = conn.execute(
            "SELECT AVG(severity_score) AS s FROM analyses"
        ).fetchone()
        snap.avg_severity = round(float(avg_row["s"] or 0.0), 2)
        snap.threats = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM analyses WHERE has_threat = 1"
            ).fetchone()["n"]
        )

        # Category distribution.
        cat_rows = conn.execute(
            "SELECT suggested_category, COUNT(*) AS n "
            "FROM analyses GROUP BY suggested_category"
        ).fetchall()
        snap.category_counts = {r["suggested_category"]: int(r["n"]) for r in cat_rows}

        # Severity band distribution. Initialise every band that
        # SEVERITY_BANDS knows about so the snapshot is complete
        # even on sparse data (and so we don't drift from the
        # canonical label set, e.g. "moderate" vs "medium").
        severity_rows = conn.execute(
            "SELECT severity_score FROM analyses"
        ).fetchall()
        bands: dict[str, int] = {
            label: 0 for label, _lo, _hi in SEVERITY_BANDS
        }
        for row in severity_rows:
            bands[severity_band(int(row["severity_score"]))] += 1
        snap.severity_counts = bands

        # Recent cases for the timeline strip.
        recent_rows = conn.execute(
            "SELECT c.case_id, c.source_name, c.source_type, c.created_at, "
            "       (SELECT a.severity_level FROM analyses a "
            "          WHERE a.case_id = c.case_id "
            "          ORDER BY a.created_at DESC LIMIT 1) AS latest_severity_level "
            "FROM cases c ORDER BY c.created_at DESC LIMIT ?",
            (recent_limit,),
        ).fetchall()
        snap.recent_cases = [dict(r) for r in recent_rows]

    return snap


def worst_recent(db_path: Path, *, limit: int = 3) -> list[dict[str, Any]]:
    """Return the highest-severity recent cases (descending)."""
    if not db_path.exists():
        return []
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT c.case_id, c.source_name, c.created_at, "
            "       a.severity_score, a.severity_level, a.suggested_category "
            "FROM cases c JOIN analyses a ON a.case_id = c.case_id "
            "ORDER BY a.created_at DESC LIMIT 50"
        ).fetchall()
    ranked = sorted(
        rows,
        key=lambda r: (
            int(r["severity_score"]),
            _band_index(r["severity_level"] or "low"),
        ),
        reverse=True,
    )
    return [dict(r) for r in ranked[:limit]]