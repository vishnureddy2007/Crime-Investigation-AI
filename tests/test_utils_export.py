"""Tests for ``utils.export``."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

import pytest

from utils.export import to_csv, to_json, to_jsonl, to_markdown


@dataclass
class _Sample:
    name: str
    score: int


def test_to_json_with_dataclass() -> None:
    text = to_json(_Sample("knife", 8))
    payload = json.loads(text)
    assert payload == {"name": "knife", "score": 8}


def test_to_json_with_list() -> None:
    rows = [{"a": 1}, {"a": 2}]
    payload = json.loads(to_json(rows))
    assert payload == rows


def test_to_json_handles_non_serialisable() -> None:
    class _Opaque:
        def __str__(self) -> str:
            return "opaque!"

    payload = json.loads(to_json({"v": _Opaque()}))
    assert payload["v"] == "opaque!"


def test_to_csv_writes_header_and_rows() -> None:
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    text = to_csv(rows)
    reader = csv.DictReader(io.StringIO(text))
    assert list(reader) == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_to_csv_handles_missing_keys() -> None:
    rows = [{"a": 1, "b": 2}, {"a": 3}]
    text = to_csv(rows)
    assert text.splitlines()[0] == "a,b"
    assert text.splitlines()[1] == "1,2"
    assert text.splitlines()[2] == "3,"


def test_to_csv_empty_returns_empty_string() -> None:
    assert to_csv([]) == ""


def test_to_jsonl_one_per_line() -> None:
    text = to_jsonl([{"a": 1}, {"a": 2}])
    lines = text.split("\n")
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"a": 2}


def test_to_markdown_renders_table() -> None:
    rows = [{"name": "knife", "score": 8}, {"name": "gun", "score": 9}]
    text = to_markdown(rows)
    assert "| name | score |" in text
    assert "| knife | 8 |" in text
    assert "| gun | 9 |" in text


def test_to_markdown_uses_explicit_columns() -> None:
    rows = [{"a": 1, "b": 2}]
    text = to_markdown(rows, columns=["b", "a"])
    assert "| b | a |" in text
    assert "| 2 | 1 |" in text


def test_to_markdown_empty() -> None:
    assert to_markdown([]) == ""


def test_to_markdown_escapes_pipes_and_newlines() -> None:
    rows = [{"x": "with | pipe\nand newline"}]
    text = to_markdown(rows)
    assert "\\|" in text
    assert "and newline" in text

# ----------------------------------------------------------------------
# Phase 24 — defensive coverage
# ----------------------------------------------------------------------


def test_to_csv_bool_renders_as_true_false() -> None:
    """_csv_safe converts bool to 'true'/'false' so CSV consumers get
    a portable, deterministic representation rather than 'True'/'False'
    Python repr."""
    rows = [{"flag_true": True, "flag_false": False, "value": 3}]
    text = to_csv(rows)
    assert "flag_true" in text
    assert "true" in text
    assert "false" in text
    # Python's str(True) = "True" → must NOT be in the CSV
    assert ",True," not in text
    assert ",False," not in text


def test_to_markdown_no_columns_returns_empty() -> None:
    """When the first row has no keys, columns ends up empty and
    to_markdown returns '' — defensive branch at line 96."""
    rows = [{}]
    assert to_markdown(rows) == ""


def test_to_markdown_none_cell_renders_as_empty() -> None:
    """A None cell value renders as an empty markdown cell
    (not the literal string 'None'). Defensive branch at line 111."""
    rows = [{"a": 1, "b": None, "c": "x"}]
    text = to_markdown(rows)
    # The '| |' (empty cell) should appear, not '| None |'
    assert "None" not in text
    assert "|  |" in text
