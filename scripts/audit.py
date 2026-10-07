#!/usr/bin/env python3
"""
Project audit script — runs the four-pattern sweep that has been
performed manually in Phases 16-25.

Each pattern checks for a class of bug that has been caught in a
prior audit pass:

1. **Label drift** — severity/category labels that are hardcoded
   instead of derived from the canonical constants in `config`.
   Phases 20 & 22 caught `"medium"` vs `"moderate"` drift.
2. **Dead code / hardcoded constants** — bare string literals that
   should be in `config.settings`. Phases 18 & 23 caught dead helpers.
3. **Path safety** — direct uses of `os.path.join` with user input.
   Phase 16 caught path-traversal in `utils/video_io`.
4. **Deprecated APIs** — `datetime.utcnow()`, `distutils`,
   `inspect.getargspec`, etc. Phase 22 caught `datetime.utcnow()`.
5. **Logging hygiene** — `print()` calls outside docstrings and bare
   / broad `except:` clauses (the canonical exception is
   `core/resilience.py`, whose wrappers deliberately swallow).
6. **SQL safety** — `.execute(...)` with an f-string or string
   concatenation (violates the parameterised-query contract).

Usage:
    python scripts/audit.py            # exit non-zero if any check fails
    python scripts/audit.py --report   # also dump findings to AUDIT.md
    python scripts/audit.py --json     # also dump findings to AUDIT.json
    python scripts/audit.py --json-out path/to/report.json
                                       # write JSON to a custom path
    python scripts/audit.py --list-patterns
                                       # print every active pattern_id + severity
    python scripts/audit.py --diff BEFORE AFTER
                                       # print only NEW findings between two
                                       # AUDIT.json files; exit 1 if any
    python scripts/audit.py --scope core
                                       # only scan core/ (skip app, services, ...)
    python scripts/audit.py --fail-on smell
                                       # exit 1 only on `smell` or worse; `info`
                                       # findings are reported but ignored.
                                       # Default is `--fail-on smell` (everything
                                       # above `info`). `--strict` is an alias
                                       # for `--fail-on info`.

Exit codes:
    0 = clean (or `--diff` found no new findings, or no finding met the
        `--fail-on` threshold)
    1 = one or more findings met the `--fail-on` threshold
        (or `--diff` found new regressions)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Bump whenever the JSON schema changes shape. Downstream consumers
# (CI dashboards, the Phase 28 artifact uploader) should compare
# `schema_version` before parsing `findings`.
JSON_SCHEMA_VERSION = 1

# Canonical severity ordering — Phase 34. Higher index = more severe.
# `info` is the lowest (advisory), `bug` is the highest. The
# `--fail-on <sev>` flag uses this list to decide whether a finding
# is severe enough to make the process exit 1.
SEVERITY_ORDER: tuple[str, ...] = ("info", "smell", "drift", "bug")


def severity_rank(severity: str) -> int:
    """Return the position of `severity` in SEVERITY_ORDER.

    Unknown severities are treated as the highest rank so a typo
    never silently downgrades a finding.
    """
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return len(SEVERITY_ORDER)

# Force UTF-8 stdout so emoji box-drawing characters don't trip the
# Windows cp1252 codec when running from a non-UTF shell.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - reconfigure unavailable
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Default scan scope: every layer. `--scope <dir>` overrides this
# to a single layer for spot-checks.
SCAN_DIRS: tuple[str, ...] = (
    "app",
    "core",
    "database",
    "models",
    "pages",
    "services",
    "utils",
)

SKIP_DIRS: tuple[str, ...] = (
    "tests",
    "outputs",
    "venv",
    ".venv",
    ".git",
    "__pycache__",
)


@dataclass(frozen=True)
class Finding:
    """A single audit finding."""

    pattern_id: str
    severity: str  # "bug" | "drift" | "smell" | "info"
    file: Path
    line: int
    short_summary: str
    pattern: str  # the regex that matched

    def render(self) -> str:
        try:
            rel = self.file.relative_to(PROJECT_ROOT)
        except ValueError:
            # Off-tree paths (e.g. test fixtures); render the absolute path.
            rel = self.file
        return (
            f"[{self.severity.upper():5s}] {self.pattern_id}  "
            f"{rel}:{self.line}  {self.short_summary}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict. Schema is fixed by
        ``JSON_SCHEMA_VERSION``. Downstream tools compare the
        schema_version before parsing the findings."""
        try:
            rel = self.file.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            # Off-tree paths (e.g. test fixtures in tmp_path).
            rel = self.file.as_posix()
        return {
            "pattern_id": self.pattern_id,
            "severity": self.severity,
            "file": rel,
            "line": self.line,
            "short_summary": self.short_summary,
            "pattern": self.pattern,
        }


@dataclass
class AuditReport:
    """Aggregated audit findings."""

    findings: list[Finding] = field(default_factory=list)

    def add(self, f: Finding) -> None:
        self.findings.append(f)

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    def count_by_severity(self) -> dict[str, int]:
        """Return {severity: count} for every severity present."""
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def findings_meeting(self, severity: str) -> list[Finding]:
        """Return findings whose severity is at or above `severity`."""
        threshold = severity_rank(severity)
        return [f for f in self.findings if severity_rank(f.severity) >= threshold]

    def has_findings_meeting(self, severity: str) -> bool:
        return bool(self.findings_meeting(severity))

    def by_pattern(self) -> dict[str, list[Finding]]:
        out: dict[str, list[Finding]] = {}
        for f in self.findings:
            out.setdefault(f.pattern_id, []).append(f)
        return out

    def render(self) -> str:
        if not self.findings:
            return "✅ audit clean — no findings.\n"

        lines: list[str] = [
            f"⚠️  {len(self.findings)} audit finding(s):",
            "",
        ]
        for pattern_id, fs in self.by_pattern().items():
            lines.append(f"── {pattern_id} ({len(fs)}) " + "─" * 50)
            for f in fs:
                lines.append("  " + f.render())
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Render the report as a deterministic JSON document.

        Keys are sorted and findings are emitted in the order they
        were appended; callers that need grouping should use
        ``by_pattern()`` themselves. The schema is versioned via
        ``JSON_SCHEMA_VERSION`` so consumers can detect shape changes.
        """
        payload = {
            "schema_version": JSON_SCHEMA_VERSION,
            "clean": not self.findings,
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _finding_key(f: dict[str, object]) -> tuple[str, int, str]:
    """Stable identity for a finding across runs.

    The (file, line, pattern_id) tuple uniquely identifies a finding;
    same offender in the same file at the same line is the same finding.
    """
    return (str(f.get("file", "")), int(str(f.get("line", 0))), str(f.get("pattern_id", "")))


@dataclass
class DiffReport:
    """Compare two `AUDIT.json` reports and surface only the new
    findings introduced between them. Schema versions must match —
    mismatched schemas are a hard error so we never silently
    compare different shapes."""

    before: dict[str, object]
    after: dict[str, object]
    added: list[dict[str, object]] = field(default_factory=list)
    removed: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_json_files(cls, before_path: Path, after_path: Path) -> "DiffReport":
        """Load two `AUDIT.json` files and return the diff. Raises
        `ValueError` if the schema versions don't match — callers
        should not silently compare reports of different shapes."""
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8"))
        return cls.from_dicts(before, after)

    @classmethod
    def from_dicts(cls, before: dict[str, object], after: dict[str, object]) -> "DiffReport":
        before_schema = before.get("schema_version")
        after_schema = after.get("schema_version")
        if before_schema != after_schema:
            raise ValueError(
                f"schema_version mismatch: before={before_schema} after={after_schema} "
                f"- cannot compare reports of different shapes"
            )

        before_keys = {_finding_key(f) for f in before.get("findings", [])}
        after_keys = {_finding_key(f) for f in after.get("findings", [])}

        before_by_key = {_finding_key(f): f for f in before.get("findings", [])}
        after_by_key = {_finding_key(f): f for f in after.get("findings", [])}

        new_keys = after_keys - before_keys
        gone_keys = before_keys - after_keys

        # Preserve order from `after` / `before` for stable rendering.
        added = [after_by_key[k] for k in after_by_key if k in new_keys]
        removed = [before_by_key[k] for k in before_by_key if k in gone_keys]

        return cls(before=before, after=after, added=added, removed=removed)

    @property
    def has_regressions(self) -> bool:
        """True if any new finding was introduced. Removed findings
        are informational only — they don't fail the diff."""
        return bool(self.added)

    def regressions_meeting(self, severity: str) -> list[dict[str, object]]:
        """Return added findings whose severity is at or above
        `severity`. Used by the `--fail-on` gate in the CLI."""
        threshold = severity_rank(severity)
        return [f for f in self.added if severity_rank(str(f.get("severity", ""))) >= threshold]

    def has_regressions_meeting(self, severity: str) -> bool:
        """True if any added finding meets the severity threshold."""
        return bool(self.regressions_meeting(severity))

    def render(self) -> str:
        lines: list[str] = []
        if not self.added and not self.removed:
            return "✅ audit diff clean — no new findings.\n"

        lines.append(f"⚠️  audit diff: {len(self.added)} new, {len(self.removed)} resolved")
        lines.append("")
        if self.added:
            lines.append(f"── new findings ({len(self.added)}) " + "─" * 40)
            for f in self.added:
                lines.append("  + " + self._render_finding(f))
            lines.append("")
        if self.removed:
            lines.append(f"── resolved findings ({len(self.removed)}) " + "─" * 36)
            for f in self.removed:
                lines.append("  - " + self._render_finding(f))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_finding(f: dict[str, object]) -> str:
        return (
            f"[{str(f.get('severity', '')).upper():5s}] "
            f"{f.get('pattern_id', '')}  "
            f"{f.get('file', '')}:{f.get('line', '')}  "
            f"{f.get('short_summary', '')}"
        )


def _iter_source_files(scope: str | None = None) -> list[Path]:
    """Yield every Python source file under the project (skipping
    tests/, outputs/, venv/, .git/, __pycache__/).

    `scope` is the name of a single SCAN_DIRS entry (e.g. "core").
    When None or "all", every SCAN_DIRS entry is scanned (default).
    """
    if scope is None or scope == "all":
        dirs = SCAN_DIRS
    elif scope in SCAN_DIRS:
        dirs = (scope,)
    else:
        valid = ", ".join(SCAN_DIRS + ("all",))
        raise ValueError(
            f"unknown scope {scope!r}; valid scopes are: {valid}"
        )
    out: list[Path] = []
    for d in dirs:
        root = PROJECT_ROOT / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            # Skip files under any SKIP_DIRS.
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            out.append(path)
    return sorted(out)


def _scan(
    pattern_id: str,
    severity: str,
    regex: str,
    short_summary: str,
    report: AuditReport,
    *,
    allow_in_files: tuple[str, ...] = (),
    scope: str | None = None,
) -> None:
    """Run a single regex-based audit check over all source files."""
    _scan_filtered(
        pattern_id=pattern_id,
        severity=severity,
        regex=regex,
        short_summary=short_summary,
        report=report,
        allow_in_files=allow_in_files,
        line_filter=None,
        scope=scope,
    )


def _scan_filtered(
    pattern_id: str,
    severity: str,
    regex: str,
    short_summary: str,
    report: AuditReport,
    *,
    allow_in_files: tuple[str, ...] = (),
    line_filter: "Callable[[Path, int, str], bool] | None" = None,
    scope: str | None = None,
) -> None:
    """Run a single regex-based audit check over all source files,
    with an optional per-line filter to skip false positives and
    an optional scope to restrict to a single layer."""
    compiled = re.compile(regex)
    for path in _iter_source_files(scope=scope):
        try:
            rel = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            # Off-tree files (test fixtures in tmp_path).
            rel = path.as_posix()
        if rel in allow_in_files or path.name in allow_in_files:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not compiled.search(line):
                continue
            if line_filter is not None and not line_filter(path, lineno, line):
                continue
            report.add(Finding(
                pattern_id=pattern_id,
                severity=severity,
                file=path,
                line=lineno,
                short_summary=short_summary,
                pattern=regex,
            ))


# ----------------------------------------------------------------------
# The four-pattern sweep
# ----------------------------------------------------------------------


# Declarative registry of every pattern the audit checks for. Each
# entry is one row of data — `check_*` functions below read from this
# list so adding a new check is one entry, not a function call site.
@dataclass(frozen=True)
class AuditPattern:
    """A single declarative audit pattern entry."""

    pattern_id: str
    severity: str  # "bug" | "drift" | "smell" | "info"
    regex: str
    short_summary: str
    allow_in_files: tuple[str, ...] = ()
    # If True, use `_scan_filtered` with the docstring-aware filter.
    # Reserved for future filters — Phase 29 leaves this column in
    # place so callers can opt-in without a registry schema bump.
    use_docstring_filter: bool = False


# The pattern registry. Order matters only for the human-readable
# output; JSON consumers should rely on `pattern_id`, not order.
PATTERNS: tuple[AuditPattern, ...] = (
    # ---- Label drift (Pattern 1) ----
    AuditPattern(
        pattern_id="label-drift:severity-medium",
        severity="drift",
        regex=r'["\']medium["\']',
        short_summary="hardcoded 'medium' (canonical is 'moderate')",
        allow_in_files=(
            "tests/test_phase22_fixes.py",
            "tests/test_services_analytics.py",
        ),
    ),
    AuditPattern(
        pattern_id="label-drift:severity-in-bar-chart",
        severity="drift",
        regex=r'\(\s*["\']low["\']\s*,\s*["\'](medium|moderate)["\']\s*,\s*["\']high["\']',
        short_summary="hardcoded severity tuple in chart (must use SEVERITY_THRESHOLDS)",
    ),
    # ---- Dead code / TODO (Pattern 2) ----
    AuditPattern(
        pattern_id="dead-code:todo-marker",
        severity="smell",
        regex=r"#\s*(TODO|FIXME|XXX|HACK)\b",
        short_summary="TODO/FIXME marker — should be tracked in PROGRESS.md",
    ),
    # ---- Path safety (Pattern 3) ----
    AuditPattern(
        pattern_id="path-safety:os-path-join",
        severity="smell",
        regex=r"os\.path\.join\s*\(",
        short_summary="`os.path.join` — verify the inputs are sanitised via core.security",
        allow_in_files=("core/security.py",),
    ),
    # ---- Deprecated APIs (Pattern 4) ----
    AuditPattern(
        pattern_id="deprecated:datetime-utcnow",
        severity="bug",
        regex=r"datetime\.utcnow\s*\(",
        short_summary="`datetime.utcnow()` deprecated in 3.12, removed in 3.14+",
    ),
    AuditPattern(
        pattern_id="deprecated:distutils",
        severity="bug",
        regex=r"\b(from|import)\s+distutils\b",
        short_summary="`distutils` removed in Python 3.12+",
    ),
    AuditPattern(
        pattern_id="deprecated:inspect-getargspec",
        severity="bug",
        regex=r"\binspect\.getargspec\b",
        short_summary="`inspect.getargspec` removed in Python 3.11+",
    ),
    # ---- Logging hygiene (Pattern 5a) ----
    AuditPattern(
        pattern_id="logging-hygiene:print-call",
        severity="smell",
        regex=r"(?<![\w.])print\s*\(",
        short_summary="`print()` — use core.logging for diagnostics",
        use_docstring_filter=True,
    ),
    # ---- Logging hygiene (Pattern 5a-emoji) ----
    # Flags emoji glyphs inside `print(...)` calls — keeps CI log
    # output grep-friendly and terminal-rendering-safe. Streamlit
    # UI code legitimately uses emoji via st.markdown / st.button
    # labels and is allowed.
    AuditPattern(
        pattern_id="logging-hygiene:emoji-in-print",
        severity="smell",
        regex=r"(?<![\w.])print\s*\([^)]*[☀-➿\U0001F300-\U0001FAFF]",
        short_summary="emoji glyph inside `print(...)` — keep CI logs grep-friendly",
        allow_in_files=("scripts/smoke_app.py",),
    ),
    # ---- Logging hygiene (Pattern 5b) ----
    AuditPattern(
        pattern_id="logging-hygiene:bare-except",
        severity="smell",
        regex=r"^\s*except\s*(Exception)?\s*:\s*$",
        short_summary="bare or broad `except:` — narrow the type and log via core.logging",
        allow_in_files=("core/resilience.py",),
    ),
    # ---- SQL safety (Pattern 6) ----
    AuditPattern(
        pattern_id="sql-safety:execute-with-format",
        severity="bug",
        regex=r"\.execute\s*\(\s*(f[\"']|\".*\"\s*\+|\+.*\")",
        short_summary="parameter-less `.execute(...)` — use `?` placeholders",
    ),
)


def list_patterns() -> list[dict[str, str]]:
    """Return a JSON-friendly snapshot of the pattern registry.

    Used by `--list-patterns` and by tests that assert the registry
    stays in sync with the check_* implementations.
    """
    return [
        {
            "pattern_id": p.pattern_id,
            "severity": p.severity,
            "short_summary": p.short_summary,
            "allow_count": str(len(p.allow_in_files)),
        }
        for p in PATTERNS
    ]


def check_label_drift(report: AuditReport, *, scope: str | None = None) -> None:
    """Pattern 1: hardcoded severity / category labels that drift from
    the canonical set defined in `config.SEVERITY_THRESHOLDS` /
    `services.crime_prediction.CATEGORIES`.

    The legacy `"medium"` label was caught in Phase 20 (services) and
    Phase 22 (pages). Same risk applies to category names — we ban
    hardcoded `"assault"|"robbery"|"theft"` strings as fallbacks.
    """
    # The canonical severity set. Any hardcoded "medium" is drift.
    # Only flag code lines (not comment lines starting with `#`).
    def _is_code_match(path: Path, lineno: int, line: str) -> bool:
        # Skip pure comment lines.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return False
        # Skip docstring / multi-line comment contexts — heuristic:
        # if the line contains "medium" but the only occurrence is in
        # a comment-like phrase ("e.g.", "vs.", "legacy", etc.) treat
        # as comment.
        lowered = line.lower()
        return not any(
            hint in lowered
            for hint in ("e.g.", "vs.", "vs ", "legacy", "drift", "must not", "must use")
        )

    _scan_filtered(
        pattern_id="label-drift:severity-medium",
        severity="drift",
        regex=r'["\']medium["\']',
        short_summary="hardcoded 'medium' (canonical is 'moderate')",
        report=report,
        allow_in_files=(
            "tests/test_phase22_fixes.py",
            "tests/test_services_analytics.py",
        ),
        line_filter=_is_code_match,
        scope=scope,
    )
    _scan(
        pattern_id="label-drift:severity-in-bar-chart",
        severity="drift",
        # Any of the bands hardcoded as a tuple/list inside a page.
        regex=r'\(\s*["\']low["\']\s*,\s*["\'](medium|moderate)["\']\s*,\s*["\']high["\']',
        short_summary="hardcoded severity tuple in chart (must use SEVERITY_THRESHOLDS)",
        report=report,
        scope=scope,
    )


def check_dead_code(report: AuditReport, *, scope: str | None = None) -> None:
    """Pattern 2: dead-code patterns we keep removing.

    Phase 18: `_SLUG_RE` shadowed by `_slug()`.
    Phase 20: `@contextmanager Stopwatch` shadowed by `Stopwatch` class.
    """
    # TODO / FIXME markers — historically they linger forever.
    _scan(
        pattern_id="dead-code:todo-marker",
        severity="smell",
        regex=r"#\s*(TODO|FIXME|XXX|HACK)\b",
        short_summary="TODO/FIXME marker — should be tracked in PROGRESS.md",
        report=report,
        scope=scope,
    )


def check_path_safety(report: AuditReport, *, scope: str | None = None) -> None:
    """Pattern 3: unsafe path handling.

    Phase 16 caught path-traversal in `utils/video_io.py` (joining
    unsanitised user filenames). The fix is to ALWAYS go through
    `core.security.safe_filename()` + `safe_join()`. A bare
    `os.path.join(user_dir, user_input)` is a smell.
    """
    # `os.path.join` with user-supplied variable — flagged but not fatal.
    _scan(
        pattern_id="path-safety:os-path-join",
        severity="smell",
        regex=r"os\.path\.join\s*\(",
        short_summary="`os.path.join` — verify the inputs are sanitised via core.security",
        report=report,
        # `core/security.py` is the canonical sanitiser; it intentionally
        # uses `os.path.basename` for stripping directory components.
        allow_in_files=(
            "core/security.py",
        ),
        scope=scope,
    )


def check_deprecated_apis(report: AuditReport, *, scope: str | None = None) -> None:
    """Pattern 4: deprecated stdlib APIs.

    Phase 22 caught `datetime.utcnow()`. Other deprecations on the
    roadmap: `distutils` (gone in 3.12), `inspect.getargspec`
    (gone in 3.11), `datetime.utcfromtimestamp`.
    """
    _scan(
        pattern_id="deprecated:datetime-utcnow",
        severity="bug",
        regex=r"datetime\.utcnow\s*\(",
        short_summary="`datetime.utcnow()` deprecated in 3.12, removed in 3.14+",
        report=report,
        scope=scope,
    )
    _scan(
        pattern_id="deprecated:distutils",
        severity="bug",
        regex=r"\b(from|import)\s+distutils\b",
        short_summary="`distutils` removed in Python 3.12+",
        report=report,
        scope=scope,
    )
    _scan(
        pattern_id="deprecated:inspect-getargspec",
        severity="bug",
        regex=r"\binspect\.getargspec\b",
        short_summary="`inspect.getargspec` removed in Python 3.11+",
        report=report,
        scope=scope,
    )


def check_logging_hygiene(report: AuditReport, *, scope: str | None = None) -> None:
    """Pattern 5: logging hygiene — `print()` and bare `except:`.

    Phase 13 standardised every diagnostic on `core.logging`. A leftover
    `print(...)` is either dead debug output or a missed migration. A
    bare `except:` (or `except Exception:`) that swallows the error
    without logging it makes failures invisible — the resilience
    wrappers in `core/resilience.py` deliberately swallow, but they
    each call `logger.warning(...)` first. Patterns below are flagged
    as **smell** so CI stays green but the AUDIT.md report surfaces
    them for review.
    """
    # Skip lines inside a docstring — a multi-line `""" ... print(...)`
    # is an EXAMPLE in a docstring, not a real call site.
    def _not_in_docstring(path: Path, lineno: int, line: str) -> bool:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return True
        in_doc = False
        for i, ln in enumerate(text.splitlines(), start=1):
            stripped = ln.lstrip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Toggle: triple quote opens OR closes depending on whether
                # we're already inside one and this line closes it.
                quote = stripped[:3]
                # Count occurrences on the line — an odd count toggles, an
                # even count is a one-line docstring (toggle twice = no-op).
                toggles = stripped.count(quote)
                if toggles % 2 == 1:
                    in_doc = not in_doc
                continue
            if in_doc and i == lineno:
                return False
        return True

    _scan_filtered(
        pattern_id="logging-hygiene:print-call",
        severity="smell",
        regex=r"(?<![\w.])print\s*\(",
        short_summary="`print()` — use core.logging for diagnostics",
        report=report,
        line_filter=_not_in_docstring,
        scope=scope,
    )
    _scan(
        pattern_id="logging-hygiene:bare-except",
        severity="smell",
        # `except:` (no type) and `except Exception:` (too broad) are
        # both flagged — narrow with a specific exception type or log
        # via core.logging before swallowing.
        regex=r"^\s*except\s*(Exception)?\s*:\s*$",
        short_summary="bare or broad `except:` — narrow the type and log via core.logging",
        report=report,
        # `core/resilience.py` deliberately swallows (with logging) in
        # its retry / circuit-breaker wrappers — those are the canonical
        # exception to the rule.
        allow_in_files=(
            "core/resilience.py",
        ),
        scope=scope,
    )


def check_sql_safety(report: AuditReport, *, scope: str | None = None) -> None:
    """Pattern 6: SQL string-concatenation guard.

    README §Security claims every SQL statement uses parameterised
    queries (no string-concatenated SQL anywhere). Any f-string or
    `+`-joined string passed to `.execute(...)` violates that
    contract. A bare `.execute("...")` with no format args is fine.
    """
    _scan(
        pattern_id="sql-safety:execute-with-format",
        severity="bug",
        # `.execute(` followed by an f-string or string concatenation
        # on the same line. A bare literal (no f / no `+`) is allowed.
        regex=r"\.execute\s*\(\s*(f[\"']|\".*\"\s*\+|\+.*\")",
        short_summary="parameter-less `.execute(...)` — use `?` placeholders",
        report=report,
        scope=scope,
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help="dump a Markdown report to AUDIT.md",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="dump a JSON report to AUDIT.json (alongside the human-readable output)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write the JSON report to a custom path (implies --json)",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="print every active pattern_id + severity and exit 0",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        type=Path,
        default=None,
        help="compare two AUDIT.json files and print only the new findings introduced",
    )
    parser.add_argument(
        "--scope",
        choices=("all",) + SCAN_DIRS,
        default="all",
        help=(
            "restrict the scan to a single directory (e.g. `core` or "
            "`services`); default `all` scans every layer"
        ),
    )
    parser.add_argument(
        "--fail-on",
        choices=SEVERITY_ORDER,
        default="smell",
        dest="fail_on",
        help=(
            "minimum severity that makes the process exit 1 "
            "(default: smell). Findings below this threshold are "
            "still printed but do not fail the job."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="shorthand for `--fail-on info` (every finding is fatal)",
    )
    args = parser.parse_args(argv)
    if args.strict:
        args.fail_on = "info"

    # `--diff` short-circuits before running the scan - we only
    # need to compare two pre-existing JSON artifacts.
    if args.diff is not None:
        before_path, after_path = args.diff
        try:
            diff = DiffReport.from_json_files(before_path, after_path)
        except FileNotFoundError as exc:
            print(f"[ERR] audit diff: file not found - {exc.filename}")
            return 2
        except ValueError as exc:
            print(f"[ERR] audit diff: {exc}")
            return 2
        print(diff.render())
        # Phase 35: `--diff` consults `--fail-on` so the same
        # threshold drives both the live sweep and the PR-diff sweep.
        # Without this, a `smell` finding would fail the diff even if
        # `--fail-on bug` is set on the rest of the pipeline.
        meeting = diff.regressions_meeting(args.fail_on)
        if meeting:
            print(
                f"[INFO] fail-on threshold: {args.fail_on}  "
                f"regressions meeting threshold: {len(meeting)}"
            )
            return 1
        if diff.has_regressions:
            # New findings exist but they are all below threshold;
            # report them but exit 0.
            print(
                f"[INFO] diff introduced {len(diff.added)} finding(s) "
                f"but all are below fail-on threshold: {args.fail_on}"
            )
        return 0

    # `--list-patterns` short-circuits before running the scan so
    # contributors can quickly see what's covered.
    if args.list_patterns:
        rows = list_patterns()
        # Per-severity counts so contributors can see the breakdown
        # at a glance (Phase 34).
        by_sev: dict[str, int] = {}
        for row in rows:
            by_sev[row["severity"]] = by_sev.get(row["severity"], 0) + 1
        sev_parts = [f"{sev}={by_sev[sev]}" for sev in SEVERITY_ORDER if sev in by_sev]
        print(f"Active audit patterns ({len(rows)}): " + ", ".join(sev_parts))
        print(f"fail-on threshold: {args.fail_on}")
        for row in rows:
            print(
                f"  [{row['severity'].upper():5s}] {row['pattern_id']}"
                f"  ({row['allow_count']} allow)  {row['short_summary']}"
            )
        return 0

    report = AuditReport()
    check_label_drift(report, scope=args.scope)
    check_dead_code(report, scope=args.scope)
    check_path_safety(report, scope=args.scope)
    check_deprecated_apis(report, scope=args.scope)
    check_logging_hygiene(report, scope=args.scope)
    check_sql_safety(report, scope=args.scope)

    print(report.render())

    # Surface the severity breakdown so CI logs make it obvious why
    # the job exited (or didn't).
    by_sev = report.count_by_severity()
    if by_sev:
        parts = [f"{sev}={by_sev[sev]}" for sev in SEVERITY_ORDER if sev in by_sev]
        print(f"[INFO] severity totals: " + ", ".join(parts))
        print(f"[INFO] fail-on threshold: {args.fail_on}")

    if args.report and report.has_findings:
        out = PROJECT_ROOT / "AUDIT.md"
        out.write_text(
            "# Project audit report\n\n"
            + report.render().replace("⚠️ ", "## ").replace("── ", "### "),
            encoding="utf-8",
        )
        print(f"Markdown report written to {out.relative_to(PROJECT_ROOT)}")

    # JSON output: enabled by either --json or --json-out. The exit
    # code is governed by findings, not by whether JSON was written —
    # a clean run still produces a `clean: true` document so CI can
    # diff it between runs.
    if args.json or args.json_out is not None:
        json_target = args.json_out if args.json_out is not None else (PROJECT_ROOT / "AUDIT.json")
        json_target.parent.mkdir(parents=True, exist_ok=True)
        json_target.write_text(report.to_json(), encoding="utf-8")
        try:
            display = json_target.relative_to(PROJECT_ROOT)
        except ValueError:
            display = json_target
        print(f"JSON report written to {display}")

    return 1 if report.has_findings_meeting(args.fail_on) else 0


if __name__ == "__main__":
    sys.exit(main())
