"""
SQLite connection helper and schema bootstrap.

Provides:
- `SCHEMA_SQL`: the DDL for all 5 tables.
- `get_connection(path)`: a context manager that yields a connection
  with `row_factory=sqlite3.Row` and `PRAGMA foreign_keys=ON`.
- `init_db(path)`: idempotent schema bootstrap (safe to call repeatedly).

This module is pure I/O + DDL. All business logic lives in
`database.repository`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


# Schema for the normalized investigations database.
# 5 core tables: cases (parent) + analyses / summaries / reports / storyboards.
# Plus a `feedback` table for the Contact page (Phase 13).
# ON DELETE CASCADE keeps child rows tidy when a case is removed.
SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS cases (
    case_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name  TEXT    NOT NULL,
    source_type  TEXT    NOT NULL,
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            INTEGER NOT NULL,
    payload_json       TEXT    NOT NULL,
    severity_score     INTEGER NOT NULL,
    severity_level     TEXT    NOT NULL,
    suggested_category TEXT    NOT NULL,
    has_threat         INTEGER NOT NULL,
    created_at         TEXT    NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS summaries (
    summary_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL,
    payload_json TEXT    NOT NULL,
    model_name   TEXT    NOT NULL,
    used_ai      INTEGER NOT NULL,
    is_outdated  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reports (
    report_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id        INTEGER NOT NULL,
    payload_json   TEXT    NOT NULL,
    report_uid     TEXT    NOT NULL,
    severity_score INTEGER NOT NULL,
    severity_level TEXT    NOT NULL,
    app_version    TEXT    NOT NULL,
    is_outdated    INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT    NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS storyboards (
    storyboard_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            INTEGER NOT NULL,
    payload_json       TEXT    NOT NULL,
    scene_count        INTEGER NOT NULL,
    total_duration_sec REAL    NOT NULL,
    is_outdated        INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT    NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS animations (
    animation_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            INTEGER NOT NULL,
    payload_json       TEXT    NOT NULL,
    video_path        TEXT    NOT NULL,
    is_outdated        INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT    NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    email       TEXT    NOT NULL,
    subject     TEXT    NOT NULL,
    body        TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS human_reviews (
    review_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id           INTEGER NOT NULL,
    analysis_id       INTEGER,
    detection_key     TEXT    NOT NULL,
    label             TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    decision          TEXT    NOT NULL,
    reviewer          TEXT,
    note              TEXT,
    original_status   TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    FOREIGN KEY (case_id)     REFERENCES cases(case_id) ON DELETE CASCADE,
    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_human_reviews_case
    ON human_reviews(case_id, created_at);
"""


@contextmanager
def get_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """
    Yield a `sqlite3.Connection` to the database at `path`.

    Sets `row_factory=sqlite3.Row` (so queries return dict-like rows)
    and enables `PRAGMA foreign_keys=ON` so cascade deletes work.
    Commits on success, always closes.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except (sqlite3.Error, RuntimeError):
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path) -> None:
    """
    Create the database file (and parent dirs) and ensure all tables exist.

    Idempotent — safe to call on every app start or page render.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(path) as conn:
        conn.executescript(SCHEMA_SQL)
        # Migration: Ensure spatial columns exist in cases table
        try:
            conn.execute("ALTER TABLE cases ADD COLUMN latitude REAL")
            conn.execute("ALTER TABLE cases ADD COLUMN longitude REAL")
        except sqlite3.OperationalError:
            # Columns already exist
            pass

        # Migration: Ensure is_outdated exists in child tables
        try:
            conn.execute("ALTER TABLE summaries ADD COLUMN is_outdated INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE reports ADD COLUMN is_outdated INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE storyboards ADD COLUMN is_outdated INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass



def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a single sqlite3.Row to a plain dict, or None if missing."""
    return dict(row) if row is not None else None
