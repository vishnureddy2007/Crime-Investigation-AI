# Contributing to Crime-Investigation-AI

Thanks for contributing! This document covers the developer workflow,
the audit-script enforcement layer, and how to extend the project
without breaking the CI guarantees.

---

## Local setup

```bash
git clone https://github.com/<your-username>/Crime-Investigation-AI.git
cd Crime-Investigation-AI

# Windows
python -m venv .venv && .venv\Scripts\activate

# macOS / Linux
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

Optional but recommended — install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

After `pre-commit install`, every `git commit` automatically runs
`python scripts/audit.py` locally and blocks the commit if any of
the 10 audit patterns regress. See [Audit script](#audit-script)
below for the pattern list and how to add a new one.

---

## Running the test suite

```bash
pytest                       # fast suite
pytest --cov=...             # with line coverage

# The audit gate (same as CI)
python scripts/audit.py

# The audit JSON artifact (Phase 28)
python scripts/audit.py --json

# List every active audit pattern (Phase 29)
python scripts/audit.py --list-patterns

# Spot-check a single layer (Phase 33) — `--scope all` is the default
python scripts/audit.py --scope core
python scripts/audit.py --scope services

# Tune which severity is fatal in CI (Phase 34). Default is `smell`.
python scripts/audit.py --fail-on bug         # only `bug` (or worse) fails
python scripts/audit.py --strict              # shorthand for --fail-on info

# Compare two AUDIT.json artifacts and print only NEW findings (Phase 31)
python scripts/audit.py --diff before.json after.json

# Combine --diff with --fail-on so PR-diff only fails on regressions at or above the threshold (Phase 35)
python scripts/audit.py --diff before.json after.json --fail-on bug
python scripts/audit.py --diff before.json after.json --strict
```

Today: **551 tests across 39 files, ~98% line coverage on
`app/ core/ services/ models/ utils/ database/ config/`**.

---

## Audit script

`scripts/audit.py` codifies a six-family pattern sweep that was
performed by hand across Phases 16–26. The script is dependency-free
(stdlib only) and is wired into both:

1. **Local pre-commit hook** — `.pre-commit-config.yaml` runs it on
   every `git commit`. Failures block the commit.
2. **CI pre-test gate** — `.github/workflows/ci.yml` runs it before
   pytest on every push and PR. Failures fail the job.

If both layers see the same script, a regression never reaches CI.

### Active patterns

Run `python scripts/audit.py --list-patterns` for the live list.
Today it covers:

| Pattern | Severity | What it catches |
|---|---|---|
| `label-drift:severity-medium` | drift | hardcoded `"medium"` (canonical is `"moderate"`) |
| `label-drift:severity-in-bar-chart` | drift | hardcoded severity tuple in a chart |
| `dead-code:todo-marker` | smell | `# TODO / FIXME / XXX / HACK` markers |
| `path-safety:os-path-join` | smell | bare `os.path.join(...)` (verify inputs are sanitised) |
| `deprecated:datetime-utcnow` | bug | `datetime.utcnow()` (removed in 3.14+) |
| `deprecated:distutils` | bug | `from distutils` / `import distutils` (removed in 3.12) |
| `deprecated:inspect-getargspec` | bug | `inspect.getargspec` (removed in 3.11) |
| `logging-hygiene:print-call` | smell | `print(...)` outside docstrings |
| `logging-hygiene:emoji-in-print` | smell | emoji glyph inside `print(...)` (keeps CI logs grep-friendly) |
| `logging-hygiene:bare-except` | smell | bare `except:` / `except Exception:` |
| `sql-safety:execute-with-format` | bug | `.execute(f"...")` or string-concat |

### Adding a new pattern

Adding a check is **one row of data** in the `PATTERNS` registry at
the top of `scripts/audit.py`:

```python
@dataclass(frozen=True)
class AuditPattern:
    pattern_id: str
    severity: str  # "bug" | "drift" | "smell" | "info"
    regex: str
    short_summary: str
    allow_in_files: tuple[str, ...] = ()
    use_docstring_filter: bool = False

PATTERNS: tuple[AuditPattern, ...] = (
    # ... existing patterns ...

    # My new check
    AuditPattern(
        pattern_id="my-pattern:short-id",
        severity="smell",
        regex=r"...",
        short_summary="short human-readable explanation",
        allow_in_files=("tests/test_my_thing.py",),  # optional
    ),
)
```

Then:

1. **Add a regression test** in `tests/test_audit_script.py` that
   plants an offender in `tmp_path` and asserts the new
   `pattern_id` appears in `report.findings`.
2. **Run `python scripts/audit.py`** — the current codebase should
   still be clean.
3. **Update the table above** in this file with the new pattern.

The `test_audit_pattern_registry_covers_all_active_checks` test
asserts the registry contains the expected set of pattern_ids, so
forgetting to add a row will fail CI.

### Exempting a file

If a pattern is correct but a specific file is a documented
exception (e.g. `core/resilience.py` deliberately swallows in its
retry / circuit-breaker wrappers), add the relative POSIX path to
the pattern's `allow_in_files`:

```python
AuditPattern(
    pattern_id="logging-hygiene:bare-except",
    ...
    allow_in_files=("core/resilience.py",),
),
```

The path can match either the full relative path
(`"core/resilience.py"`) or just the basename (`"resilience.py"`).

### Custom line filters

If a pattern produces too many false positives in comments or
docstrings, pass a `line_filter` to `_scan_filtered` (see
`check_label_drift` for an example using a comment-skip heuristic,
or `check_logging_hygiene` for a docstring-skip heuristic).

---

## Coding conventions

- **No paid dependencies.** Every package in `requirements.txt`
  must be free and open source.
- **No cloud calls.** No `openai`, `anthropic`, `google-generativeai`,
  etc. — the audit script enforces this as part of the
  pattern sweep.
- **All SQL is parameterised.** `.execute("SELECT ... WHERE id = ?", (id,))`,
  never `.execute(f"SELECT ... WHERE id = {id}")`.
- **Bare excepts are smells, not bugs.** Use `(ImportError, OSError)`
  etc., and `core.logging.logger.warning(...)` if you're swallowing.
- **Free-text inputs go through `core.security.validate_text_input`.**
  File uploads go through `core.security.safe_filename` +
  `safe_join`.

---

## Submitting a change

1. Branch off `main`: `git checkout -b my-feature`
2. Make your change.
3. Run the audit + tests locally:
   ```bash
   python scripts/audit.py
   pytest
   ```
4. Commit. If `pre-commit install` was run, the audit hook runs
   automatically. Otherwise run it manually.
5. Push and open a PR. CI will run the audit + pytest + the
   docker-build smoke job.

---

## See also

- [README.md](README.md) — project overview, features, installation.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layer responsibilities + mermaid.
- [docs/API.md](docs/API.md) — every public symbol.
- [docs/PROGRESS.md](docs/PROGRESS.md) — milestone-by-milestone changelog.
- [docs/DEVELOPER.md](docs/DEVELOPER.md) — extension cookbook.
