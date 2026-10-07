"""Unit tests for the new `feedback` repository."""
from __future__ import annotations

from database.db import init_db
from database.repository import list_feedback, save_feedback


def test_save_and_list_feedback(db_path) -> None:
    init_db(db_path)
    fid = save_feedback(
        db_path,
        name="Alice",
        email="alice@example.com",
        subject="Hello",
        body="Just saying hi.",
    )
    assert fid >= 1
    rows = list_feedback(db_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "Alice"
    assert r["email"] == "alice@example.com"
    assert r["subject"] == "Hello"
    assert r["body"] == "Just saying hi."
    assert r["created_at"]


def test_list_feedback_respects_limit(db_path) -> None:
    init_db(db_path)
    for i in range(5):
        save_feedback(db_path, name=f"u{i}", email=f"u{i}@x.com",
                      subject="s", body="b")
    rows = list_feedback(db_path, limit=2)
    assert len(rows) == 2


def test_list_feedback_orders_newest_first(db_path) -> None:
    init_db(db_path)
    save_feedback(db_path, "a", "a@x.com", "first", "body")
    save_feedback(db_path, "b", "b@x.com", "second", "body")
    rows = list_feedback(db_path)
    assert rows[0]["subject"] == "second"
    assert rows[1]["subject"] == "first"