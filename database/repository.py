"""
Repository functions over the investigations SQLite database.

Every function takes a `db_path: Path` as its first argument so tests
can drive them with `tmp_path`. No Streamlit imports anywhere — keep
the data layer pure and trivially testable.

Each save also populates a few denormalized hot-path columns so the
Case History page can render a preview without unpacking the JSON
payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from database.db import _row_to_dict, get_connection
from models.schemas import (
    EvidenceAnalysis,
    InvestigationSummary,
    ReportData,
    Storyboard,
)


def mark_case_outdated(db_path: Path, case_id: int) -> None:
    """Mark all summaries, reports, storyboards, and animations for a case as outdated."""
    with get_connection(db_path) as conn:
        conn.execute("UPDATE summaries SET is_outdated = 1 WHERE case_id = ?", (case_id,))
        conn.execute("UPDATE reports SET is_outdated = 1 WHERE case_id = ?", (case_id,))
        conn.execute("UPDATE storyboards SET is_outdated = 1 WHERE case_id = ?", (case_id,))
        conn.execute("UPDATE animations SET is_outdated = 1 WHERE case_id = ?", (case_id,))


def _now_iso() -> str:
    # Full microsecond precision so back-to-back saves in tests
    # always sort distinctly by created_at.
    return datetime.now().isoformat()


def _to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _from_json(text: str) -> dict[str, Any]:
    return json.loads(text)


# ----------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------
def save_case(db_path: Path, source_name: str, source_type: str) -> int:
    """Insert a new `cases` row. Returns the new case_id."""
    created_at = _now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO cases (source_name, source_type, created_at) "
            "VALUES (?, ?, ?)",
            (source_name, source_type, created_at),
        )
        return int(cur.lastrowid)


def list_cases(db_path: Path) -> list[dict[str, Any]]:
    """
    List all cases ordered by created_at DESC, joined with the latest
    analysis's severity for a quick preview column.

    Uses a single LEFT JOIN with a window function (ROW_NUMBER) to
    pick the latest analysis per case — replaces the previous N+1
    correlated subqueries with one pass over `cases` plus one pass
    over `analyses`.
    """
    sql = """
        SELECT c.case_id,
               c.source_name,
               c.source_type,
               c.created_at,
               latest.severity_level  AS latest_severity_level,
               latest.severity_score  AS latest_severity_score
        FROM cases c
        LEFT JOIN (
            SELECT case_id,
                   severity_level,
                   severity_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY case_id
                       ORDER BY created_at DESC, analysis_id DESC
                   ) AS rn
            FROM analyses
        ) latest ON latest.case_id = c.case_id AND latest.rn = 1
        ORDER BY c.created_at DESC
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def latest_case_for_source(db_path: Path, source_name: str) -> int | None:
    """Return the most recent case_id for `source_name`, or None."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT case_id FROM cases WHERE source_name = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (source_name,),
        ).fetchone()
    return int(row["case_id"]) if row is not None else None


# ----------------------------------------------------------------------
# analyses
# ----------------------------------------------------------------------
def save_analysis(
    db_path: Path, case_id: int, analysis: EvidenceAnalysis,
) -> int:
    """Persist an `EvidenceAnalysis` against `case_id`. Returns new analysis_id."""
    payload = analysis.as_dict()
    created_at = _now_iso()
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO analyses (case_id, payload_json, severity_score, "
                "severity_level, suggested_category, has_threat, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    _to_json(payload),
                    int(analysis.severity_score),
                    analysis.severity_level,
                    analysis.suggested_category,
                    int(bool(analysis.has_threat)),
                    created_at,
                ),
            )
            analysis_id = int(cur.lastrowid)
    except Exception as exc:
        # Use a descriptive error to help diagnose "unable to open database file"
        raise RuntimeError(f"Database auto-save failed for analysis (Case #{case_id}): {exc}") from exc

    # Any existing summary/report is now outdated.
    mark_case_outdated(db_path, case_id)
    return analysis_id


def load_analysis(db_path: Path, analysis_id: int) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM analyses WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
    if row is None:
        return None
    return _from_json(row["payload_json"])


def list_analyses_for_case(db_path: Path, case_id: int) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT analysis_id, severity_score, severity_level, "
            "       suggested_category, has_threat, created_at "
            "FROM analyses WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------
def save_summary(
    db_path: Path, case_id: int, summary: InvestigationSummary,
) -> int:
    payload = summary.as_dict()
    created_at = _now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO summaries (case_id, payload_json, model_name, "
            "used_ai, is_outdated, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                case_id,
                _to_json(payload),
                summary.model_name,
                int(bool(summary.used_ai)),
                0,
                created_at,
            ),
        )
        return int(cur.lastrowid)


def load_summary(db_path: Path, summary_id: int) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json, is_outdated FROM summaries WHERE summary_id = ?",
            (summary_id,),
        ).fetchone()
    if row is None:
        return None
    res = _from_json(row["payload_json"])
    res["is_outdated"] = bool(row["is_outdated"])
    return res


def list_summaries_for_case(db_path: Path, case_id: int) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT summary_id, model_name, used_ai, created_at "
            "FROM summaries WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------
def save_report(db_path: Path, case_id: int, report: ReportData) -> int:
    payload = report.as_dict()
    created_at = _now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO reports (case_id, payload_json, report_uid, "
            "severity_score, severity_level, app_version, is_outdated, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                case_id,
                _to_json(payload),
                report.report_id,
                int(report.severity_score),
                report.severity_level,
                report.app_version,
                0,
                created_at,
            ),
        )
        return int(cur.lastrowid)


def load_report(db_path: Path, report_id: int) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json, is_outdated FROM reports WHERE report_id = ?",
            (report_id,),
        ).fetchone()
    if row is None:
        return None
    res = _from_json(row["payload_json"])
    res["is_outdated"] = bool(row["is_outdated"])
    return res


def list_reports_for_case(db_path: Path, case_id: int) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT report_id, report_uid, severity_score, severity_level, "
            "       app_version, created_at "
            "FROM reports WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------
# storyboards
# ----------------------------------------------------------------------
def save_storyboard(db_path: Path, case_id: int, storyboard: Storyboard) -> int:
    payload = storyboard.as_dict()
    created_at = _now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO storyboards (case_id, payload_json, scene_count, "
            "total_duration_sec, is_outdated, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                case_id,
                _to_json(payload),
                int(storyboard.scene_count),
                float(storyboard.total_duration_sec),
                0,
                created_at,
            ),
        )
        return int(cur.lastrowid)


def save_animation(db_path: Path, case_id: int, video_path: str, payload: dict[str, Any]) -> int:
    """Persist an animation video reference against `case_id`."""
    created_at = _now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO animations (case_id, payload_json, video_path, is_outdated, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, _to_json(payload), video_path, 0, created_at),
        )
        return int(cur.lastrowid)


def load_animation(db_path: Path, animation_id: int) -> dict[str, Any] | None:
    """Load an animation reference by ID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json, video_path, is_outdated FROM animations WHERE animation_id = ?",
            (animation_id,),
        ).fetchone()
    if row is None:
        return None
    res = _from_json(row["payload_json"])
    res["video_path"] = row["video_path"]
    res["is_outdated"] = bool(row["is_outdated"])
    return res


def list_animations_for_case(db_path: Path, case_id: int) -> list[dict[str, Any]]:
    """List all animations for a case, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT animation_id, video_path, created_at "
            "FROM animations WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_storyboard(db_path: Path, storyboard_id: int) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json, is_outdated FROM storyboards WHERE storyboard_id = ?",
            (storyboard_id,),
        ).fetchone()
    if row is None:
        return None
    res = _from_json(row["payload_json"])
    res["is_outdated"] = bool(row["is_outdated"])
    return res


def list_storyboards_for_case(db_path: Path, case_id: int) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT storyboard_id, scene_count, total_duration_sec, created_at "
            "FROM storyboards WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------
# feedback (Phase 13 — Contact page)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FeedbackEntry:
    """A single user-submitted feedback row."""
    name: str
    email: str
    subject: str
    body: str
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name":       self.name,
            "email":      self.email,
            "subject":    self.subject,
            "body":       self.body,
            "created_at": self.created_at,
        }


def save_feedback(
    db_path: Path,
    name: str,
    email: str,
    subject: str,
    body: str,
) -> int:
    """Persist a contact-form submission. Returns the new feedback_id."""
    created_at = _now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO feedback (name, email, subject, body, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, email, subject, body, created_at),
        )
        return int(cur.lastrowid)


def list_feedback(db_path: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent `limit` feedback rows, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT feedback_id, name, email, subject, body, created_at "
            "FROM feedback ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------
def case_exists(db_path: Path, case_id: int) -> bool:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM cases WHERE case_id = ?", (case_id,),
        ).fetchone()
    return row is not None


# ----------------------------------------------------------------------
# human reviews (Phase 51 — explicit human-in-the-loop audit trail)
# ----------------------------------------------------------------------
VALID_REVIEW_DECISIONS: frozenset[str] = frozenset({"CONFIRM", "REJECT", "UNCERTAIN"})


def save_human_review(
    db_path: Path,
    case_id: int,
    detection_key: str,
    label: str,
    confidence: float,
    decision: str,
    reviewer: str = "",
    note: str = "",
    original_status: str = "candidate",
    analysis_id: int | None = None,
) -> int:
    """Persist a human-review decision against a detection.

    `decision` must be one of `CONFIRM`, `REJECT`, `UNCERTAIN`. Anything
    else raises ValueError so we never store a typo. `detection_key`
    is a stable per-detection string the application constructs (e.g.
    ``"weapon@0.62:frame_001"``) so the same detection can be reviewed
    repeatedly without producing duplicates.
    """
    if decision not in VALID_REVIEW_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(VALID_REVIEW_DECISIONS)}, got {decision!r}"
        )
    created_at = _now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO human_reviews (case_id, analysis_id, detection_key, "
            "label, confidence, decision, reviewer, note, original_status, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                case_id,
                analysis_id,
                detection_key,
                label,
                float(confidence),
                decision,
                reviewer,
                note,
                original_status,
                created_at,
            ),
        )
        review_id = int(cur.lastrowid)

    # Any existing summary/report is now outdated because the truth has changed.
    mark_case_outdated(db_path, case_id)
    return review_id


def list_human_reviews(db_path: Path, case_id: int) -> list[dict[str, Any]]:
    """All human-review decisions for a case, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT review_id, detection_key, label, confidence, decision, "
            "       reviewer, note, original_status, created_at "
            "FROM human_reviews WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_human_review_stats(db_path: Path, case_id: int) -> dict[str, int]:
    """Aggregate human review statistics for a case."""
    reviews = list_human_reviews(db_path, case_id)
    total = len(reviews)
    confirmed = sum(1 for r in reviews if r["decision"] == "CONFIRM")
    rejected = sum(1 for r in reviews if r["decision"] == "REJECT")
    uncertain = sum(1 for r in reviews if r["decision"] == "UNCERTAIN")

    return {
        "human_total_detections": total,
        "human_confirmed_count": confirmed,
        "human_rejected_count": rejected,
        "human_pending_count": uncertain,
    }


def human_review_for(db_path: Path, case_id: int, detection_key: str) -> dict[str, Any] | None:
    """Return the latest review for a detection_key, or None."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT review_id, detection_key, label, confidence, decision, "
            "       reviewer, note, original_status, created_at "
            "FROM human_reviews WHERE case_id = ? AND detection_key = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (case_id, detection_key),
        ).fetchone()
    return dict(row) if row is not None else None


# ----------------------------------------------------------------------
# Link Analysis (Intelligence Layer)
# ----------------------------------------------------------------------

def get_global_label_distribution(db_path: Path) -> dict[str, int]:
    """Aggregate total counts of every label across all cases."""
    total_dist: dict[str, int] = {}
    with get_connection(db_path) as conn:
        # Get latest analysis for every case
        rows = conn.execute("""
            SELECT payload_json FROM analyses a
            JOIN (
                SELECT case_id, ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY created_at DESC) as rn
                FROM analyses
            ) latest ON a.case_id = latest.case_id AND latest.rn = 1
        """).fetchall()

        for row in rows:
            payload = _from_json(row["payload_json"])
            counts = payload.get("counts_by_label", {})
            for label, count in counts.items():
                total_dist[label] = total_dist.get(label, 0) + count
    return total_dist


def find_cases_by_label(db_path: Path, label: str) -> list[dict[str, Any]]:
    """Find all cases that have a specific object label detected."""
    matching_cases: list[dict[str, Any]] = []
    with get_connection(db_path) as conn:
        # Get latest analysis for every case
        rows = conn.execute("""
            SELECT c.case_id, c.source_name, a.payload_json
            FROM cases c
            JOIN analyses a ON c.case_id = a.case_id
            JOIN (
                SELECT case_id, ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY created_at DESC) as rn
                FROM analyses
            ) latest ON a.case_id = latest.case_id AND latest.rn = 1
        """).fetchall()

        for row in rows:
            payload = _from_json(row["payload_json"])
            if label in payload.get("counts_by_label", {}):
                matching_cases.append({
                    "case_id": row["case_id"],
                    "source_name": row["source_name"]
                })
    return matching_cases


def find_linked_cases(db_path: Path, case_id: int, exclude_labels: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Find all other cases that share evidence (labels) with the given case.
    Returns a list of {'case_id': ..., 'shared_count': ...} sorted by count DESC.
    """
    # 1. Get target case labels
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM analyses a "
            "JOIN (SELECT case_id, ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY created_at DESC) as rn FROM analyses) latest "
            "ON a.case_id = latest.case_id AND latest.rn = 1 "
            "WHERE a.case_id = ?",
            (case_id,),
        ).fetchone()

    if not row:
        return []

    target_payload = _from_json(row["payload_json"])
    target_labels = set(target_payload.get("counts_by_label", {}).keys())

    if exclude_labels:
        target_labels -= set(exclude_labels)

    if not target_labels:
        return []

    # 2. Find other cases sharing any of these labels
    links: dict[int, int] = {}
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT c.case_id, a.payload_json
            FROM cases c
            JOIN analyses a ON c.case_id = a.case_id
            JOIN (
                SELECT case_id, ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY created_at DESC) as rn
                FROM analyses
            ) latest ON a.case_id = latest.case_id AND latest.rn = 1
            WHERE c.case_id != ?
        """, (case_id,)).fetchall()

        for row in rows:
            other_id = row["case_id"]
            other_payload = _from_json(row["payload_json"])
            other_labels = set(other_payload.get("counts_by_label", {}).keys())

            shared = target_labels.intersection(other_labels)
            if shared:
                links[other_id] = len(shared)

    # Return sorted list
    return sorted(
        [{"case_id": cid, "shared_count": count} for cid, count in links.items()],
        key=lambda x: x["shared_count"],
        reverse=True,
    )

# ----------------------------------------------------------------------
# Spatial Intelligence
# ----------------------------------------------------------------------

def update_case_location(db_path: Path, case_id: int, lat: float, lon: float) -> None:
    """Set the geographic coordinates for a case."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE cases SET latitude = ?, longitude = ? WHERE case_id = ?",
            (lat, lon, case_id),
        )

def get_cases_with_locations(db_path: Path) -> list[dict[str, Any]]:
    """List all cases that have spatial data."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT case_id, source_name, latitude, longitude FROM cases WHERE latitude IS NOT NULL AND longitude IS NOT NULL",
        ).fetchall()
    return [dict(r) for r in rows]

# ----------------------------------------------------------------------
# Temporal Intelligence
# ----------------------------------------------------------------------

def get_evidence_trends(db_path: Path) -> list[dict[str, Any]]:
    """
    Get a time-series of object detections.
    Returns a list of: {'date': 'YYYY-MM-DD', 'labels': {'person': 5, 'weapon': 1}}
    """
    with get_connection(db_path) as conn:
        # Get latest analysis for every case
        rows = conn.execute("""
            SELECT c.created_at, a.payload_json
            FROM cases c
            JOIN (
                SELECT case_id, ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY created_at DESC) as rn
                FROM analyses
            ) latest ON c.case_id = latest.case_id AND latest.rn = 1
        """).fetchall()

        trends: dict[str, dict[str, int]] = {}
        for row in rows:
            # Strip time from ISO timestamp for daily grouping
            date = row["created_at"].split("T")[0]
            payload = _from_json(row["payload_json"])
            counts = payload.get("counts_by_label", {})

            if date not in trends:
                trends[date] = {}

            for label, count in counts.items():
                trends[date][label] = trends[date].get(label, 0) + count

        # Convert to sorted list
        sorted_dates = sorted(trends.keys())
        return [{"date": d, "labels": trends[d]} for d in sorted_dates]
