"""
Export helpers — convert dataclasses / dicts into portable text formats.

Used by the **Analytics** page (and the **Case History** page when
exporting single rows). All helpers are pure functions with no
Streamlit / IO dependencies so they're trivial to test.

Public surface
--------------
- :func:`to_json` — pretty-printed JSON.
- :func:`to_csv` — RFC 4180 CSV (one row per dict in the iterable).
- :func:`to_markdown` — a small Markdown table with a header row.
- :func:`to_jsonl` — newline-delimited JSON (one object per line).
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


def _coerce(obj: Any) -> Any:
    """Convert dataclass instances to dicts; leave everything else alone."""
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def to_json(obj: Any, *, indent: int = 2) -> str:
    """Serialise ``obj`` (or list of objects) to pretty-printed JSON."""
    return json.dumps(_coerce(obj), indent=indent, default=str, ensure_ascii=False)


def to_jsonl(items: Iterable[Any]) -> str:
    """Serialise an iterable of objects, one per line (JSONL)."""
    lines = [json.dumps(_coerce(item), default=str, ensure_ascii=False) for item in items]
    return "\n".join(lines)


def to_csv(rows: Iterable[Mapping[str, Any]]) -> str:
    """Serialise an iterable of mappings to RFC-4180 CSV.

    Column order follows the first row; later rows may omit keys
    (rendered as empty cells).
    """
    rows = list(rows)
    if not rows:
        return ""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_safe(row.get(k)) for k in fieldnames})
    return buffer.getvalue()


def _csv_safe(value: Any) -> str:
    """Render a value as a string suitable for a CSV cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def to_markdown(rows: Iterable[Mapping[str, Any]], *, columns: list[str] | None = None) -> str:
    """Render rows as a Markdown table.

    If ``columns`` is None, columns are taken from the first row.
    """
    rows = list(rows)
    if not rows:
        return ""
    if columns is None:
        seen: list[str] = []
        dedup: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in dedup:
                    dedup.add(key)
                    seen.append(key)
        columns = seen
    if not columns:
        return ""

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines: list[str] = []
    for row in rows:
        body_lines.append(
            "| " + " | ".join(_md_cell(row.get(col, "")) for col in columns) + " |"
        )
    return "\n".join([header, sep, *body_lines])


def _md_cell(value: Any) -> str:
    """Escape a value for inclusion in a Markdown table cell."""
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text.strip()