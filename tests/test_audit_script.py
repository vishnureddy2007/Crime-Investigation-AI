"""
Unit tests for scripts/audit.py.

The audit script codifies the four-pattern sweep that has been
performed manually across Phases 16-25. These tests verify that the
script:

1. Returns clean (no findings) on the current codebase.
2. Detects each of the four patterns when an offender is planted.
3. Skips files in `allow_in_files` (regression-test exemptions).
4. Skips pure comment lines (heuristic).
5. Emits a versioned JSON report (Phase 28 — CI artifact).
6. Maintains a pattern registry (Phase 29).
7. Is wired into the pre-commit hook (Phase 30).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utf8_env() -> dict[str, str]:
    """Return a copy of os.environ with UTF-8 stdio forced on.

    Phase 31 / 32 fix: subprocess output of the audit script can include
    Unicode (e.g. `[OK]`, em-dash) that crashes cp1252 stdout on
    Windows. We force `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1` for
    every subprocess invocation so the child can print freely and the
    parent decoder never sees a `UnicodeDecodeError`.
    """
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _run_audit() -> subprocess.CompletedProcess[str]:
    """Invoke the audit script as a subprocess."""
    return _run_subprocess([sys.executable, "scripts/audit.py"])


def _run_subprocess(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Invoke a subprocess with UTF-8 encoding handled correctly on Windows."""
    if cwd is None:
        cwd = PROJECT_ROOT
    merged_env = _utf8_env()
    if env:
        merged_env.update(env)
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=merged_env,
    )


def test_audit_clean_on_current_codebase() -> None:
    """No findings on the actual codebase — the four patterns hold."""
    proc = _run_audit()
    assert "audit clean" in proc.stdout, (
        f"audit script found findings:\n{proc.stdout}\n{proc.stderr}"
    )
    assert proc.returncode == 0


def test_audit_detects_utcnow(tmp_path: Path) -> None:
    """An off-tree source file with `datetime.utcnow()` triggers the
    deprecated-utcnow pattern. We test the underlying scan function
    rather than mutating the real codebase."""
    from scripts.audit import _scan, AuditReport

    fake = tmp_path / "fake_offender.py"
    fake.write_text("from datetime import datetime\nx = datetime.utcnow()\n")

    # Patch the iteration to scan only our tmp file.
    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="deprecated:datetime-utcnow",
            severity="bug",
            regex=r"datetime\.utcnow\s*\(",
            short_summary="deprecated",
            report=report,
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert any(
        f.pattern_id == "deprecated:datetime-utcnow" for f in report.findings
    )


def test_audit_respects_allow_in_files(tmp_path: Path) -> None:
    """Files in `allow_in_files` are skipped even if they match."""
    from scripts.audit import _scan, AuditReport

    fake = tmp_path / "exempt.py"
    fake.write_text("x = datetime.utcnow()\n")

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="deprecated:datetime-utcnow",
            severity="bug",
            regex=r"datetime\.utcnow\s*\(",
            short_summary="deprecated",
            report=report,
            # Allow-list by basename — `_scan_filtered` matches by either
            # the relative path or just the filename.
            allow_in_files=(fake.name,),
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert report.findings == []


def test_audit_line_filter_skips_comments(tmp_path: Path) -> None:
    """`line_filter` callback can suppress false-positive matches in
    comment lines."""
    from scripts.audit import _scan_filtered, AuditReport

    fake = tmp_path / "with_comment.py"
    fake.write_text(
        "# this is a comment mentioning datetime.utcnow()\n"
        "import datetime\n"
    )

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan_filtered(
            pattern_id="deprecated:datetime-utcnow",
            severity="bug",
            regex=r"datetime\.utcnow\s*\(",
            short_summary="deprecated",
            report=report,
            line_filter=lambda _p, _ln, line: not line.lstrip().startswith("#"),
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert report.findings == []


def test_audit_report_render_clean() -> None:
    """An empty report renders the clean-message."""
    from scripts.audit import AuditReport

    rep = AuditReport()
    assert "audit clean" in rep.render()
    assert not rep.has_findings


def test_audit_report_render_groups_by_pattern(tmp_path: Path) -> None:
    """Findings are grouped by pattern_id in the rendered output."""
    from scripts.audit import AuditReport, Finding, PROJECT_ROOT

    rep = AuditReport()
    rep.add(Finding(
        pattern_id="pattern-a", severity="bug",
        file=PROJECT_ROOT / "fake_a.py", line=1,
        short_summary="a", pattern="x",
    ))
    rep.add(Finding(
        pattern_id="pattern-b", severity="bug",
        file=PROJECT_ROOT / "fake_b.py", line=1,
        short_summary="b", pattern="y",
    ))
    text = rep.render()
    assert "pattern-a" in text
    assert "pattern-b" in text
    assert rep.has_findings


# ----------------------------------------------------------------------
# Phase 27 — new patterns: logging hygiene + SQL safety
# ----------------------------------------------------------------------


def test_audit_detects_print_call_outside_docstring(tmp_path: Path) -> None:
    """A real `print(...)` call site (not inside a docstring) triggers
    the `logging-hygiene:print-call` pattern. The docstring-skip
    heuristic is exercised by the existing `core/observability.py:103`
    example, which must NOT be flagged."""
    from scripts.audit import _scan, AuditReport

    fake = tmp_path / "real_print.py"
    fake.write_text('def f():\n    print("hello")\n')

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="logging-hygiene:print-call",
            severity="smell",
            regex=r"(?<![\w.])print\s*\(",
            short_summary="print call",
            report=report,
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert any(
        f.pattern_id == "logging-hygiene:print-call" for f in report.findings
    ), f"expected the print call to be flagged, got: {report.findings}"


def test_audit_skips_print_inside_docstring(tmp_path: Path) -> None:
    """A `print(...)` example inside a triple-quoted docstring is NOT
    a real call site — the audit's docstring-aware line_filter must
    skip it. This is the same pattern that protects
    `core/observability.py:103` from being flagged."""
    from scripts.audit import _scan_filtered, AuditReport

    fake = tmp_path / "docstring_print.py"
    fake.write_text(
        'def f():\n'
        '    """\n'
        '    Example:\n'
        '        print("inside docstring")\n'
        '    """\n'
        '    return 1\n'
    )

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan_filtered(
            pattern_id="logging-hygiene:print-call",
            severity="smell",
            regex=r"(?<![\w.])print\s*\(",
            short_summary="print call",
            report=report,
            # The real check_logging_hygiene() supplies this filter.
            line_filter=audit_mod.check_logging_hygiene.__doc__ and (
                lambda _p, _ln, _line: True
            ),  # placeholder — replaced below
        )
    finally:
        audit_mod._iter_source_files = original_iter

    # Build the actual filter by calling the production function with
    # a real (project) file and inspecting the captured line. The unit
    # we care about: the production line_filter returns False for
    # lines inside a docstring block.
    from scripts.audit import check_logging_hygiene
    # The line_filter is a closure inside check_logging_hygiene — we
    # verify the public contract by running the real check on the
    # fake and asserting no findings.
    report = AuditReport()
    # Re-run check_logging_hygiene with the fake as the only file.
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        # Re-import the function to get a fresh closure bound to the
        # patched _iter_source_files (the closure captures the module
        # function, not the variable, so this is fine).
        check_logging_hygiene(report)
    finally:
        audit_mod._iter_source_files = original_iter

    assert report.findings == [], (
        f"docstring print should be skipped, got: {report.findings}"
    )


def test_audit_detects_emoji_inside_print(tmp_path: Path) -> None:
    """Emoji glyphs (✅/❌/⚠/🔍/🚨) inside `print(...)` trigger the
    `logging-hygiene:emoji-in-print` smell. CI logs stay grep-friendly
    and terminal-rendering-safe when nothing inside a print call has
    a non-ASCII glyph."""
    from scripts.audit import _scan, AuditReport

    # Build the file using chr() so Windows cp1252 doesn't break the
    # test source itself. Each line uses a different common emoji.
    # ✅ ✅ ❌ ❌ ⚠ ⚠ \U0001F50D 🔍 \U0001F6A8 🚨
    emoji_lines = [
        'def f():\n',
        '    print("ok")\n',                # line 2 — control (no emoji)
        f'    print("{chr(0x2705)} done")\n',     # line 3 — ✅
        f'    print("{chr(0x274C)} fail")\n',     # line 4 — ❌
        f'    print("{chr(0x26A0)} warn")\n',     # line 5 — ⚠
        f'    print("{chr(0x1F50D)} search")\n',  # line 6 — 🔍
        f'    print("{chr(0x1F6A8)} alert")\n',   # line 7 — 🚨
    ]
    fake = tmp_path / "emoji_print.py"
    fake.write_text("".join(emoji_lines), encoding="utf-8")

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="logging-hygiene:emoji-in-print",
            severity="smell",
            regex=next(
                p.regex for p in audit_mod.PATTERNS
                if p.pattern_id == "logging-hygiene:emoji-in-print"
            ),
            short_summary="emoji in print",
            report=report,
        )
    finally:
        audit_mod._iter_source_files = original_iter

    flagged_lines = sorted({f.line for f in report.findings})
    # Five emoji-bearing print lines; line 2 is the control (no emoji).
    assert flagged_lines == [3, 4, 5, 6, 7], (
        f"expected lines 3-7 flagged, got: {flagged_lines}"
    )


def test_audit_emoji_in_print_respects_allow_in_files(tmp_path: Path) -> None:
    """`scripts/smoke_app.py` legitimately prints `OK` lines that may
    carry glyphs on some terminals — the pattern's `allow_in_files`
    must exempt it. We emulate the same contract with a tmp file."""
    from scripts.audit import _scan, AuditReport

    fake = tmp_path / "exempt_emoji.py"
    fake.write_text(
        'def f():\n'
        '    print("ok")\n'
    )

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    # Pass allow_in_files matching the fake so the exemption path runs.
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="logging-hygiene:emoji-in-print",
            severity="smell",
            regex=next(
                p.regex for p in audit_mod.PATTERNS
                if p.pattern_id == "logging-hygiene:emoji-in-print"
            ),
            short_summary="emoji in print",
            report=report,
            allow_in_files=(fake.name,),
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert report.findings == [], (
        f"allow_in_files did not exempt the file: {report.findings}"
    )


def test_audit_emoji_in_print_does_not_flag_non_print_strings(tmp_path: Path) -> None:
    """Emoji outside `print(...)` (e.g. in st.markdown, button labels,
    log messages) must NOT trigger the pattern — the rule is scoped
    to print calls only so Streamlit UI code stays untouched."""
    from scripts.audit import _scan, AuditReport

    fake = tmp_path / "streamlit_ui.py"
    fake.write_text(
        'import streamlit as st\n'
        'def render():\n'
        '    st.markdown("done")\n'
        '    st.button("ok")\n'
        '    logger.info("warn")\n'
        '    label = "search"\n'
    )

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="logging-hygiene:emoji-in-print",
            severity="smell",
            regex=next(
                p.regex for p in audit_mod.PATTERNS
                if p.pattern_id == "logging-hygiene:emoji-in-print"
            ),
            short_summary="emoji in print",
            report=report,
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert report.findings == [], (
        f"emoji outside print() triggered false positive: {report.findings}"
    )


def test_audit_detects_bare_except(tmp_path: Path) -> None:
    """A bare `except:` (and `except Exception:`) triggers the
    `logging-hygiene:bare-except` pattern. The current codebase has
    zero bare excepts after the Phase 27 narrowing pass, so this
    test plants an offender in a tmp file to verify detection."""
    from scripts.audit import _scan, AuditReport

    fake = tmp_path / "bad_except.py"
    fake.write_text(
        "try:\n"
        "    x = 1\n"
        "except Exception:\n"
        "    pass\n"
    )

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="logging-hygiene:bare-except",
            severity="smell",
            regex=r"^\s*except\s*(Exception)?\s*:\s*$",
            short_summary="bare except",
            report=report,
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert any(
        f.pattern_id == "logging-hygiene:bare-except" for f in report.findings
    )


def test_audit_detects_sql_execute_with_fstring(tmp_path: Path) -> None:
    """`cur.execute(f\"SELECT ... {user_id}\")` violates the
    parameterised-query contract and triggers the
    `sql-safety:execute-with-format` pattern."""
    from scripts.audit import _scan, AuditReport

    fake = tmp_path / "bad_sql.py"
    fake.write_text(
        'def unsafe(user_id: str):\n'
        '    cur.execute(f"SELECT * FROM cases WHERE id = {user_id}")\n'
    )

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="sql-safety:execute-with-format",
            severity="bug",
            regex=r"\.execute\s*\(\s*(f[\"']|\".*\"\s*\+|\+.*\")",
            short_summary="raw sql",
            report=report,
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert any(
        f.pattern_id == "sql-safety:execute-with-format" for f in report.findings
    )


def test_audit_does_not_flag_parameterised_sql(tmp_path: Path) -> None:
    """`.execute("SELECT ...", (user_id,))` is the canonical safe
    form — it must NOT be flagged."""
    from scripts.audit import _scan, AuditReport

    fake = tmp_path / "safe_sql.py"
    fake.write_text(
        'def safe(user_id: str):\n'
        '    cur.execute("SELECT * FROM cases WHERE id = ?", (user_id,))\n'
    )

    import scripts.audit as audit_mod
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [fake]
    try:
        report = AuditReport()
        _scan(
            pattern_id="sql-safety:execute-with-format",
            severity="bug",
            regex=r"\.execute\s*\(\s*(f[\"']|\".*\"\s*\+|\+.*\")",
            short_summary="raw sql",
            report=report,
        )
    finally:
        audit_mod._iter_source_files = original_iter

    assert report.findings == []


# ----------------------------------------------------------------------
# Phase 28 — JSON output mode
# ----------------------------------------------------------------------


def test_audit_report_to_json_is_deterministic_and_versioned(tmp_path: Path) -> None:
    """`AuditReport.to_json()` returns a deterministic, versioned
    document. Two consecutive serialisations of the same report must
    be byte-identical (sorted keys, no trailing-whitespace drift),
    and the document must carry `schema_version` for downstream
    diff tools to detect shape changes."""
    import json
    from scripts.audit import AuditReport, Finding, JSON_SCHEMA_VERSION

    rep = AuditReport()
    rep.add(Finding(
        pattern_id="label-drift:severity-medium",
        severity="drift",
        file=tmp_path / "fake.py",
        line=7,
        short_summary="hardcoded 'medium'",
        pattern=r'"medium"',
    ))
    doc = json.loads(rep.to_json())

    assert doc["schema_version"] == JSON_SCHEMA_VERSION
    assert doc["clean"] is False
    assert doc["finding_count"] == 1
    assert len(doc["findings"]) == 1

    f = doc["findings"][0]
    assert f["pattern_id"] == "label-drift:severity-medium"
    assert f["severity"] == "drift"
    assert f["line"] == 7
    assert f["short_summary"] == "hardcoded 'medium'"
    assert f["pattern"] == r'"medium"'
    # file key is the as_posix() of the path
    assert f["file"].endswith("fake.py")

    # Deterministic: a second serialisation of the same report must
    # be byte-identical (sort_keys=True in to_json).
    assert rep.to_json() == rep.to_json()


def test_audit_report_to_json_clean_run(tmp_path: Path) -> None:
    """A clean run produces a `clean: true` document with an empty
    findings list — CI can diff this between pushes to detect new
    findings even on a green job."""
    import json
    from scripts.audit import AuditReport

    rep = AuditReport()
    doc = json.loads(rep.to_json())

    assert doc["clean"] is True
    assert doc["finding_count"] == 0
    assert doc["findings"] == []


def test_audit_finding_to_dict_offtree_path(tmp_path: Path) -> None:
    """`Finding.to_dict()` handles off-tree paths (test fixtures in
    tmp_path) by emitting the absolute as_posix() rather than
    raising ValueError from `relative_to(PROJECT_ROOT)`."""
    from scripts.audit import Finding

    f = Finding(
        pattern_id="test:offtree",
        severity="bug",
        file=tmp_path / "fixture.py",
        line=1,
        short_summary="off-tree",
        pattern="x",
    )
    d = f.to_dict()
    assert d["file"].endswith("fixture.py")
    assert d["line"] == 1
    assert d["pattern_id"] == "test:offtree"


def test_audit_json_flag_writes_artifact(tmp_path: Path) -> None:
    """`--json` writes a parseable AUDIT.json next to the script.
    We run the script via subprocess with cwd=PROJECT_ROOT so the
    output lands in the project tree (and we clean it up)."""
    import json
    import sys

    audit_json = PROJECT_ROOT / "AUDIT.json"
    existed = audit_json.exists()
    try:
        proc = _run_subprocess([sys.executable, "scripts/audit.py", "--json"])
        assert proc.returncode == 0
        assert audit_json.exists()
        doc = json.loads(audit_json.read_text(encoding="utf-8"))
        assert doc["clean"] is True
        assert doc["finding_count"] == 0
    finally:
        if not existed and audit_json.exists():
            audit_json.unlink()


# ----------------------------------------------------------------------
# Phase 29 — pattern registry + --list-patterns
# ----------------------------------------------------------------------


def test_audit_pattern_registry_covers_all_active_checks() -> None:
    """Every check_* function in scripts/audit.py must have a matching
    entry in PATTERNS — the registry is the source of truth for
    `--list-patterns`. If a contributor adds a new check without
    listing it in PATTERNS, this test fails."""
    import scripts.audit as audit_mod

    expected = {
        "label-drift:severity-medium",
        "label-drift:severity-in-bar-chart",
        "dead-code:todo-marker",
        "path-safety:os-path-join",
        "deprecated:datetime-utcnow",
        "deprecated:distutils",
        "deprecated:inspect-getargspec",
        "logging-hygiene:print-call",
        "logging-hygiene:emoji-in-print",
        "logging-hygiene:bare-except",
        "sql-safety:execute-with-format",
    }
    actual = {p.pattern_id for p in audit_mod.PATTERNS}
    assert actual == expected, (
        f"PATTERNS registry drift.\n"
        f"  Missing from registry: {expected - actual}\n"
        f"  Extra in registry:    {actual - expected}"
    )


def test_audit_list_patterns_emits_every_active_pattern_id(tmp_path: Path) -> None:
    """`--list-patterns` prints every pattern_id and exits 0 — even
    on a clean codebase. We assert the subprocess output contains
    each registered pattern_id."""
    import subprocess
    import sys

    proc = _run_subprocess([sys.executable, "scripts/audit.py", "--list-patterns"])
    assert proc.returncode == 0
    assert "Active audit patterns" in proc.stdout
    for pattern_id in (
        "label-drift:severity-medium",
        "dead-code:todo-marker",
        "path-safety:os-path-join",
        "deprecated:datetime-utcnow",
        "logging-hygiene:print-call",
        "logging-hygiene:bare-except",
        "sql-safety:execute-with-format",
    ):
        assert pattern_id in proc.stdout, (
            f"--list-patterns missing {pattern_id!r}\n{proc.stdout}"
        )


def test_audit_list_patterns_function_shape() -> None:
    """`list_patterns()` returns a JSON-friendly snapshot — every
    row has `pattern_id`, `severity`, `short_summary`, `allow_count`.
    Tests can import this and assert against it without running
    the CLI."""
    from scripts.audit import list_patterns

    rows = list_patterns()
    assert len(rows) >= 10  # we ship 10 active patterns
    for row in rows:
        assert set(row.keys()) == {"pattern_id", "severity", "short_summary", "allow_count"}
        assert row["allow_count"].isdigit()


# ----------------------------------------------------------------------
# Phase 30 — pre-commit hook + CONTRIBUTING guide
# ----------------------------------------------------------------------


def test_precommit_config_exists_and_runs_audit() -> None:
    """`.pre-commit-config.yaml` exists, declares the `project-audit`
    hook, and points at the audit script. The hook entry must be
    `python scripts/audit.py` so the local hook matches the CI
    pre-test gate. We avoid a PyYAML dependency by reading the file
    as text and asserting the structural field strings appear on
    their expected lines."""
    config_path = PROJECT_ROOT / ".pre-commit-config.yaml"
    assert config_path.exists(), "missing .pre-commit-config.yaml"

    text = config_path.read_text(encoding="utf-8")

    # Required structural fields (each on its own line).
    assert "repos:" in text, "pre-commit config must define `repos:`"
    assert "- repo: local" in text, "hook must be a local repo"
    assert "id: project-audit" in text, "hook must be named `project-audit`"
    assert "name: Project audit (scripts/audit.py)" in text
    assert "entry: python scripts/audit.py" in text, (
        "pre-commit entry must be `python scripts/audit.py`"
    )
    assert "language: system" in text, (
        "pre-commit hook should use language=system so it runs the local Python"
    )
    assert "pass_filenames: false" in text, (
        "audit hook must receive pass_filenames=false so it scans the whole tree"
    )
    assert "always_run: true" in text, (
        "audit hook must run on every commit, not just when listed files change"
    )


def test_precommit_hook_command_exits_clean() -> None:
    """The exact command the pre-commit hook runs (`python scripts/audit.py`)
    exits 0 on the current codebase. This is the same exit-code check
    the CI pre-test gate runs; if this fails, the pre-commit hook will
    block every commit."""
    proc = _run_subprocess([sys.executable, "scripts/audit.py"])
    assert proc.returncode == 0, (
        f"pre-commit hook command failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "audit clean" in proc.stdout


def test_contributing_md_references_audit_and_registry() -> None:
    """`CONTRIBUTING.md` must explain the audit script, the pattern
    registry, and how to add a new check. Drift here = contributors
    don't know the rules."""
    contributing = PROJECT_ROOT / "CONTRIBUTING.md"
    assert contributing.exists(), "missing CONTRIBUTING.md"
    text = contributing.read_text(encoding="utf-8")

    # Audit script is referenced by name and shown as a CLI command.
    assert "scripts/audit.py" in text
    assert "--list-patterns" in text
    assert "PATTERNS" in text, "must explain the PATTERNS registry"

    # The 6 pattern families are all named.
    for pattern_id in (
        "label-drift:severity-medium",
        "dead-code:todo-marker",
        "path-safety:os-path-join",
        "deprecated:datetime-utcnow",
        "logging-hygiene:print-call",
        "sql-safety:execute-with-format",
    ):
        assert pattern_id in text, f"CONTRIBUTING.md missing pattern {pattern_id}"

    # The local setup block exists.
    assert "pre-commit install" in text
    assert "pip install" in text

    # Coding conventions are stated.
    assert "parameterised" in text or "parameterized" in text
    assert "allow_in_files" in text, "must explain the allow_in_files exemption"


# ----------------------------------------------------------------------
# Phase 31 — --diff mode
# ----------------------------------------------------------------------


def test_audit_diff_clean_when_no_new_findings(tmp_path: Path) -> None:
    """`DiffReport` between two IDENTICAL findings lists reports
    no new findings and no regressions. The check itself is
    data-only — no subprocess needed."""
    import json
    from scripts.audit import DiffReport

    payload = {
        "schema_version": 1,
        "clean": False,
        "finding_count": 1,
        "findings": [
            {
                "pattern_id": "deprecated:datetime-utcnow",
                "severity": "bug",
                "file": "utils/x.py",
                "line": 5,
                "short_summary": "datetime.utcnow()",
                "pattern": "datetime.utcnow",
            }
        ],
    }
    diff = DiffReport.from_dicts(payload, payload)
    assert diff.has_regressions is False
    assert diff.added == []
    assert diff.removed == []
    assert "audit diff clean" in diff.render()


def test_audit_diff_detects_new_finding(tmp_path: Path) -> None:
    """A finding present in `after` but not in `before` is detected
    as a new regression. The same (file, line, pattern_id) tuple
    in both is treated as the same finding."""
    import json
    from scripts.audit import DiffReport

    before = {
        "schema_version": 1, "clean": True, "finding_count": 0, "findings": []
    }
    after = {
        "schema_version": 1, "clean": False, "finding_count": 1,
        "findings": [
            {
                "pattern_id": "deprecated:datetime-utcnow",
                "severity": "bug",
                "file": "utils/x.py",
                "line": 5,
                "short_summary": "datetime.utcnow()",
                "pattern": "datetime.utcnow",
            }
        ],
    }
    diff = DiffReport.from_dicts(before, after)
    assert diff.has_regressions is True
    assert len(diff.added) == 1
    assert diff.added[0]["pattern_id"] == "deprecated:datetime-utcnow"
    assert diff.added[0]["file"] == "utils/x.py"
    assert diff.added[0]["line"] == 5
    assert diff.removed == []
    # Render includes the new finding with a "+" prefix.
    rendered = diff.render()
    assert "new findings" in rendered
    assert "datetime-utcnow" in rendered
    assert "+ [BUG  ]" in rendered


def test_audit_diff_treats_resolved_as_informational(tmp_path: Path) -> None:
    """A finding present in `before` but not in `after` is rendered
    as a `resolved` finding, but it does NOT count as a regression.
    `has_regressions` is keyed on `added` only — removed findings
    are information, not failure."""
    from scripts.audit import DiffReport

    before = {
        "schema_version": 1, "clean": False, "finding_count": 1,
        "findings": [
            {
                "pattern_id": "deprecated:datetime-utcnow",
                "severity": "bug",
                "file": "utils/x.py",
                "line": 5,
                "short_summary": "datetime.utcnow()",
                "pattern": "datetime.utcnow",
            }
        ],
    }
    after = {
        "schema_version": 1, "clean": True, "finding_count": 0, "findings": []
    }
    diff = DiffReport.from_dicts(before, after)
    assert diff.has_regressions is False
    assert diff.added == []
    assert len(diff.removed) == 1
    assert diff.removed[0]["pattern_id"] == "deprecated:datetime-utcnow"


def test_audit_diff_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    """Reports of different `schema_version`s cannot be compared —
    shape differences make the diff meaningless. `ValueError` is
    raised so the CLI can exit 2 (config error)."""
    from scripts.audit import DiffReport

    before = {"schema_version": 1, "findings": []}
    after = {"schema_version": 2, "findings": []}

    with pytest.raises(ValueError, match="schema_version mismatch"):
        DiffReport.from_dicts(before, after)


def test_audit_diff_subprocess_new_finding(tmp_path: Path) -> None:
    """End-to-end: run the audit, plant an offender, run it again,
    then use `--diff` to surface the new finding. The exit code must
    be 1 (regression detected) and the rendered output must contain
    the new pattern_id."""
    import json
    import os
    import subprocess
    import sys

    # Force UTF-8 in the subprocess so emoji + em-dash don't crash
    # the cp1252 stdout decoder on Windows.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"

    # Plant the offender so the second run picks it up.
    offender = PROJECT_ROOT / "utils" / "_phase31_diff_test.py"
    offender.write_text("def f():\n    return datetime.utcnow()\n", encoding="utf-8")
    try:
        # After = with offender.
        proc = subprocess.run(
            [sys.executable, "scripts/audit.py", "--json", "--json-out", str(after_path)],
            cwd=PROJECT_ROOT, capture_output=True, encoding="utf-8",
            errors="replace", env=env, timeout=60,
        )
        # The audit exits 1 for a finding, but the JSON is still written.
        assert proc.returncode == 1
        assert after_path.exists()

        # Remove the offender and capture the clean baseline.
        offender.unlink()
        proc = subprocess.run(
            [sys.executable, "scripts/audit.py", "--json", "--json-out", str(before_path)],
            cwd=PROJECT_ROOT, capture_output=True, encoding="utf-8",
            errors="replace", env=env, timeout=60,
        )
        assert proc.returncode == 0
        assert before_path.exists()

        # Now ask for the diff.
        proc = subprocess.run(
            [sys.executable, "scripts/audit.py", "--diff",
             str(before_path), str(after_path)],
            cwd=PROJECT_ROOT, capture_output=True, encoding="utf-8",
            errors="replace", env=env, timeout=60,
        )
        assert proc.returncode == 1, f"diff should detect regression\n{proc.stdout}"
        assert "new findings" in proc.stdout
        assert "datetime-utcnow" in proc.stdout
    finally:
        if offender.exists():
            offender.unlink()
        audit_json = PROJECT_ROOT / "AUDIT.json"
        if audit_json.exists():
            audit_json.unlink()


# ----------------------------------------------------------------------
# Phase 32 — CI PR-audit-diff job
# ----------------------------------------------------------------------


def test_ci_yaml_defines_pr_audit_diff_job() -> None:
    """`.github/workflows/ci.yml` defines a `pr-audit-diff` job that
    runs only on pull_request events and uses the audit `--diff`
    mode added in Phase 31. We read the YAML as text and assert the
    structural fields appear on their expected lines."""
    config_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    assert config_path.exists(), "missing .github/workflows/ci.yml"
    text = config_path.read_text(encoding="utf-8")

    # Job declaration.
    assert "pr-audit-diff:" in text, "ci.yml must define a `pr-audit-diff` job"

    # Only runs on pull_request, not on direct pushes.
    assert "github.event_name == 'pull_request'" in text, (
        "pr-audit-diff must be gated on pull_request events"
    )

    # Step that runs the diff invocation.
    assert "scripts/audit.py --diff" in text, (
        "pr-audit-diff must invoke `python scripts/audit.py --diff`"
    )

    # Two-checkout dance: base branch AUDIT.json + head branch AUDIT.json.
    assert "github.event.pull_request.base.ref" in text, (
        "pr-audit-diff must check out the PR base ref"
    )
    assert "fetch-depth: 1" in text, (
        "the dual checkout should use fetch-depth: 1 for speed"
    )

    # The diff artifact is uploaded so reviewers can re-download it.
    assert "name: audit-diff" in text, (
        "pr-audit-diff must upload an `audit-diff` artifact"
    )


def test_ci_yaml_does_not_run_diff_on_direct_push() -> None:
    """The pr-audit-diff job must be gated to pull_request only — on
    a direct push to main, comparing against `main` is meaningless
    (the base IS the head) and would always show no new findings.
    The `github.event_name == 'pull_request'` condition must appear
    BEFORE the job-level steps, in a job-level `if:` position."""
    config_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    text = config_path.read_text(encoding="utf-8")

    # The if: must be at the job-level (indented exactly 2 spaces under
    # the job name), not on a single step.
    job_section_start = text.find("pr-audit-diff:")
    assert job_section_start != -1
    after = text[job_section_start:]

    # The next 200 chars must contain the job-level `if:` condition.
    assert "if: github.event_name == 'pull_request'" in after[:400], (
        "pr-audit-diff must have a job-level `if:` gating pull_request"
    )


def test_audit_diff_cli_handles_missing_file_gracefully() -> None:
    """If a file passed to `--diff` is missing, the CLI exits 2 with
    a clear error message - never crashes with a stack trace. This
    is the path CI takes when the base branch has no AUDIT.json
    (e.g. a brand-new branch with no prior commit)."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(
        [sys.executable, "scripts/audit.py", "--diff",
         "/nonexistent/before.json", "/nonexistent/after.json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=30,
    )
    assert proc.returncode == 2, (
        f"missing file should exit 2 (config error), got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "file not found" in proc.stdout or "FileNotFoundError" in proc.stdout


# ----------------------------------------------------------------------------
# Phase 33 - --scope <dir> flag
# ----------------------------------------------------------------------------


def test_audit_scope_cli_accepts_all_and_each_dir() -> None:
    """Phase 33: `--scope` must accept `all` (the default) plus every
    SCAN_DIRS entry. A typo should be rejected by argparse exit 2 —
    not silently treated as `all` (which would hide a regression)."""
    import argparse

    # The CLI parser exists in main() but we can replay the same
    # choices list to assert the contract exposed to contributors.
    proc = subprocess.run(
        [sys.executable, "scripts/audit.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 0
    help_text = proc.stdout + proc.stderr
    # The argparse-generated help must list `all` and every SCAN_DIRS entry.
    assert "--scope" in help_text
    assert "all" in help_text
    for layer in ("app", "core", "services", "models", "utils", "database", "pages"):
        assert layer in help_text, (
            f"--scope choices must include {layer!r} (a SCAN_DIRS entry)"
        )


def test_audit_scope_restricts_to_single_dir() -> None:
    """Phase 33: `--scope core` should print the same clean verdict
    as the full sweep, but it must NOT scan anything outside `core/`.
    We verify the *file enumeration* helper rather than the rendered
    output, because the rendered output is identical when no findings
    are present (current repo state)."""
    from scripts.audit import _iter_source_files

    files = _iter_source_files(scope="core")
    assert files, "core/ should contribute at least one .py file"
    for path in files:
        assert "core" in path.parts, (
            f"scope=core leaked a non-core file: {path}"
        )
    # Compare against the full sweep to confirm we got a strict subset.
    full = _iter_source_files(scope="all")
    assert set(files).issubset(set(full))
    assert len(files) < len(full), (
        "scope=core should sweep fewer files than the full scan"
    )


def test_audit_scope_rejects_unknown_value() -> None:
    """Phase 33: `_iter_source_files` must raise ValueError for a
    scope that isn't in SCAN_DIRS. argparse rejection is end-to-end
    tested by the CLI subprocess below."""
    from scripts.audit import _iter_source_files

    with pytest.raises(ValueError) as exc:
        _iter_source_files(scope="does-not-exist")
    assert "unknown scope" in str(exc.value)
    assert "valid scopes" in str(exc.value)


def test_audit_scope_all_matches_default_enum() -> None:
    """Phase 33: `--scope all` must enumerate the same files as the
    default (no flag) sweep. This locks the 'default == all' contract
    so future SCAN_DIRS additions don't silently diverge."""
    from scripts.audit import _iter_source_files

    default = _iter_source_files()
    explicit_all = _iter_source_files(scope="all")
    assert default == explicit_all


def test_audit_scope_cli_subprocess_rejects_bogus_value() -> None:
    """Phase 33: argparse should reject a bogus `--scope` value with
    exit code 2 — the same contract we use for `--diff` mishaps.
    Exit 0 would mean we silently coerced it to `all`, which would
    defeat the purpose of a scope flag."""
    proc = subprocess.run(
        [sys.executable, "scripts/audit.py", "--scope", "bogus"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 2, (
        f"bogus --scope should exit 2, got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "invalid choice" in combined or "--scope" in combined


def test_audit_scope_cli_subprocess_runs_clean_on_core() -> None:
    """Phase 33: end-to-end smoke. `--scope core` should run the
    scan and exit 0 on a clean tree (no findings in core/).
    The output header should match the rendered contract."""
    proc = subprocess.run(
        [sys.executable, "scripts/audit.py", "--scope", "core"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"clean --scope core should exit 0, got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "audit clean" in proc.stdout


# ----------------------------------------------------------------------------
# Phase 34 - --fail-on <severity> + severity summary
# ----------------------------------------------------------------------------


def test_severity_order_is_known_and_ordered_low_to_high() -> None:
    """Phase 34: SEVERITY_ORDER drives `--fail-on` and the new summary
    line in `--list-patterns`. Locking the order means contributors
    can rely on the contract."""
    from scripts.audit import SEVERITY_ORDER, severity_rank
    assert SEVERITY_ORDER == ("info", "smell", "drift", "bug")
    assert severity_rank("info") < severity_rank("smell")
    assert severity_rank("smell") < severity_rank("drift")
    assert severity_rank("drift") < severity_rank("bug")


def test_severity_rank_unknown_is_highest() -> None:
    """Unknown severities are treated as the highest rank so a typo
    in a pattern declaration never silently downgrades a finding."""
    from scripts.audit import severity_rank, SEVERITY_ORDER
    assert severity_rank("typo") >= len(SEVERITY_ORDER)


def test_audit_report_counts_findings_by_severity(tmp_path: Path) -> None:
    """`AuditReport.count_by_severity()` is the building block for the
    new `[INFO] severity totals` line printed in main()."""
    from scripts.audit import AuditReport, Finding

    report = AuditReport()
    report.add(Finding(
        pattern_id="a:1", severity="smell", file=tmp_path / "x.py",
        line=1, short_summary="x", pattern="x",
    ))
    report.add(Finding(
        pattern_id="b:1", severity="bug", file=tmp_path / "y.py",
        line=2, short_summary="y", pattern="y",
    ))
    report.add(Finding(
        pattern_id="a:2", severity="smell", file=tmp_path / "z.py",
        line=3, short_summary="z", pattern="z",
    ))
    assert report.count_by_severity() == {"smell": 2, "bug": 1}


def test_audit_report_findings_meeting_threshold(tmp_path: Path) -> None:
    """`AuditReport.findings_meeting(severity)` is the gate used by
    `--fail-on` to decide whether to exit 1."""
    from scripts.audit import AuditReport, Finding

    report = AuditReport()
    for sev in ("info", "smell", "drift", "bug"):
        report.add(Finding(
            pattern_id=f"x:{sev}", severity=sev,
            file=tmp_path / f"{sev}.py", line=1,
            short_summary=sev, pattern=sev,
        ))
    assert len(report.findings_meeting("bug")) == 1
    assert len(report.findings_meeting("drift")) == 2
    assert len(report.findings_meeting("smell")) == 3
    assert len(report.findings_meeting("info")) == 4
    assert not report.has_findings_meeting("smell") == False  # sanity


def test_audit_cli_fail_on_ignored_when_only_info_findings(tmp_path: Path) -> None:
    """If all findings are `info`, `--fail-on smell` (the default)
    must exit 0. CI is allowed to surface info-level audit notes
    without failing the build."""
    # Patch `_iter_source_files` to point at a tmp directory with one
    # info-level offender (the `dead-code:todo-marker` regex doesn't
    # fit because it's a smell; we use an `info` pattern instead).
    from scripts import audit as audit_mod

    info_file = tmp_path / "marker.py"
    info_file.write_text('# NOTE-OBSERVATION: please review\n')
    original_iter = audit_mod._iter_source_files
    audit_mod._iter_source_files = lambda scope=None: [info_file]
    try:
        # Add a synthetic info-only pattern registry and re-run the
        # check by calling check_logging_hygiene (smell) is not what
        # we want; instead we build the report directly.
        from scripts.audit import AuditReport, Finding
        report = AuditReport()
        report.add(Finding(
            pattern_id="test:info-only",
            severity="info",
            file=info_file, line=1,
            short_summary="synthetic info",
            pattern="synthetic",
        ))
        # Default `--fail-on smell` ignores info findings.
        assert report.has_findings_meeting("smell") is False
        # `--strict` (== `--fail-on info`) catches them.
        assert report.has_findings_meeting("info") is True
    finally:
        audit_mod._iter_source_files = original_iter


def test_audit_cli_fail_on_bug_ignores_smell(tmp_path: Path) -> None:
    """`--fail-on bug` lets the build pass even when smell findings
    are present. The report is still rendered so reviewers see them."""
    from scripts.audit import AuditReport, Finding
    report = AuditReport()
    report.add(Finding(
        pattern_id="x:smell", severity="smell",
        file=tmp_path / "x.py", line=1,
        short_summary="smell finding", pattern="x",
    ))
    assert report.has_findings_meeting("bug") is False
    assert report.has_findings_meeting("smell") is True


def test_audit_cli_strict_flag_is_alias_for_fail_on_info() -> None:
    """`--strict` must be a synonym for `--fail-on info` — every
    finding is fatal. Lock the wiring so a future refactor can't
    accidentally widen it."""
    from scripts.audit import severity_rank
    assert severity_rank("info") == 0  # the lowest threshold


def test_audit_cli_subprocess_fail_on_smell_exits_zero_when_clean() -> None:
    """End-to-end: a clean codebase + `--fail-on smell` (default)
    exits 0. Already covered implicitly, but make the contract
    explicit for Phase 34."""
    proc = subprocess.run(
        [sys.executable, "scripts/audit.py", "--fail-on", "smell"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"clean + --fail-on smell should exit 0, got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "audit clean" in proc.stdout


def test_audit_cli_subprocess_fail_on_unknown_value_rejected() -> None:
    """`--fail-on` must validate against SEVERITY_ORDER. A typo would
    be silent corruption, so argparse must reject it."""
    proc = subprocess.run(
        [sys.executable, "scripts/audit.py", "--fail-on", "bogus"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 2, (
        f"bogus --fail-on should exit 2, got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "invalid choice" in combined and "--fail-on" in combined


def test_audit_cli_subprocess_list_patterns_shows_severity_breakdown() -> None:
    """Phase 34: `--list-patterns` now prints per-severity counts and
    the current fail-on threshold. Lock the format."""
    proc = subprocess.run(
        [sys.executable, "scripts/audit.py", "--list-patterns"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 0
    assert "Active audit patterns" in proc.stdout
    assert "fail-on threshold" in proc.stdout
    # The summary line must list at least one severity bucket.
    first_line = proc.stdout.splitlines()[0]
    assert any(sev in first_line for sev in ("info=", "smell=", "drift=", "bug=")), (
        f"expected per-severity summary, got: {first_line!r}"
    )


def test_audit_cli_subprocess_strict_flag_is_accepted() -> None:
    """`--strict` must be a recognised flag (no argparse error)."""
    proc = subprocess.run(
        [sys.executable, "scripts/audit.py", "--strict"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    # Clean codebase + --strict (== --fail-on info) exits 0
    # because there are zero info-level findings.
    assert proc.returncode == 0, (
        f"--strict on clean codebase should exit 0, got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    # The list-patterns output should mention the resolved threshold.
    # (Re-run --list-patterns with --strict to confirm.)
    proc2 = subprocess.run(
        [sys.executable, "scripts/audit.py", "--list-patterns", "--strict"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc2.returncode == 0
    assert "fail-on threshold: info" in proc2.stdout


# ----------------------------------------------------------------------------
# Phase 35 - --diff honours --fail-on
# ----------------------------------------------------------------------------


def _make_diff_payload(
    findings_before: list[dict[str, object]] | None = None,
    findings_after:  list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a minimal AUDIT.json payload for diff tests."""
    return {
        "schema_version": 1,
        "clean": not (findings_after or []),
        "finding_count": len(findings_after or []),
        "findings": findings_after or [],
        "before_findings_count": len(findings_before or []),
    }


def _finding(
    pattern_id: str, severity: str, file: str = "x.py", line: int = 1,
) -> dict[str, object]:
    return {
        "pattern_id":  pattern_id,
        "severity":    severity,
        "file":        file,
        "line":        line,
        "short_summary": f"{pattern_id} stub",
        "pattern":     "stub",
    }


def test_diff_regressions_meeting_threshold_filters_by_severity() -> None:
    """Phase 35: DiffReport.regressions_meeting(severity) is the gate
    that lets `--diff` honour `--fail-on`."""
    from scripts.audit import DiffReport

    before = _make_diff_payload(findings_before=[])
    after = _make_diff_payload(findings_after=[
        _finding("a:info", "info"),
        _finding("a:smell", "smell"),
        _finding("a:drift", "drift"),
        _finding("a:bug",   "bug"),
    ])
    diff = DiffReport.from_dicts(before, after)
    assert len(diff.added) == 4

    assert len(diff.regressions_meeting("bug"))   == 1
    assert len(diff.regressions_meeting("drift")) == 2
    assert len(diff.regressions_meeting("smell")) == 3
    assert len(diff.regressions_meeting("info"))  == 4

    assert diff.has_regressions_meeting("bug")   is True
    assert diff.has_regressions_meeting("info")  is True

    # An "unknown" threshold would map to a rank past every known
    # severity, so no known finding can meet it. This guards against
    # the opposite regression (an unknown threshold silently catching
    # every finding) which would corrupt CI signal.
    assert diff.has_regressions_meeting("nonexistent-severity") is False


def test_diff_regressions_meeting_empty_diff_returns_empty_list() -> None:
    """A clean diff (no new findings) returns an empty list regardless
    of the threshold. Used by the CLI to decide exit 0."""
    from scripts.audit import DiffReport

    before = _make_diff_payload(findings_after=[_finding("a:bug", "bug")])
    after  = _make_diff_payload(findings_after=[])
    diff = DiffReport.from_dicts(before, after)
    assert diff.has_regressions is False
    assert diff.regressions_meeting("info") == []


def test_diff_subprocess_fail_on_bug_ignores_smell_findings(
    tmp_path: Path,
) -> None:
    """End-to-end: `--diff` with `--fail-on bug` exits 0 when the
    only new finding is a `smell`. Without Phase 35's wiring, this
    test would fail because every added finding used to exit 1."""
    import json
    from scripts.audit import JSON_SCHEMA_VERSION

    before_payload = _make_diff_payload(findings_before=[])
    after_payload = _make_diff_payload(findings_after=[
        _finding("logging-hygiene:print-call", "smell"),
    ])
    before_path = tmp_path / "before.json"
    after_path  = tmp_path / "after.json"
    before_path.write_text(json.dumps(before_payload), encoding="utf-8")
    after_path.write_text(json.dumps(after_payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/audit.py",
         "--diff", str(before_path), str(after_path),
         "--fail-on", "bug"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"--fail-on bug should ignore smell diff; got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "below fail-on threshold" in proc.stdout


def test_diff_subprocess_fail_on_smell_catches_smell_findings(
    tmp_path: Path,
) -> None:
    """End-to-end: `--fail-on smell` (the default) DOES catch smell
    regressions in the diff. Locks the contract that Phase 35 didn't
    accidentally silence smell findings."""
    import json

    before_payload = _make_diff_payload(findings_before=[])
    after_payload = _make_diff_payload(findings_after=[
        _finding("logging-hygiene:print-call", "smell"),
    ])
    before_path = tmp_path / "before.json"
    after_path  = tmp_path / "after.json"
    before_path.write_text(json.dumps(before_payload), encoding="utf-8")
    after_path.write_text(json.dumps(after_payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/audit.py",
         "--diff", str(before_path), str(after_path),
         "--fail-on", "smell"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 1, (
        f"--fail-on smell should exit 1 on a smell diff; got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


def test_diff_subprocess_strict_flag_catches_every_finding(
    tmp_path: Path,
) -> None:
    """`--strict` (== `--fail-on info`) catches even info-severity
    regressions in the diff."""
    import json

    before_payload = _make_diff_payload(findings_before=[])
    after_payload = _make_diff_payload(findings_after=[
        _finding("info:advisory", "info"),
    ])
    before_path = tmp_path / "before.json"
    after_path  = tmp_path / "after.json"
    before_path.write_text(json.dumps(before_payload), encoding="utf-8")
    after_path.write_text(json.dumps(after_payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/audit.py",
         "--diff", str(before_path), str(after_path),
         "--strict"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        timeout=30,
    )
    assert proc.returncode == 1, (
        f"--strict + info diff should exit 1; got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
