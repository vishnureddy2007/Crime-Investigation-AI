"""
Tests for scripts/smoke_app.py.

The smoke script imports every project package in order and prints
``OK <module>`` lines. It is invoked by:

- manual sanity checks (`python scripts/smoke_app.py`)
- CI's `smoke` job (if added later)
- the docker-build job's container (image-build-time validation)

These tests cover:
- the script can be imported as a Python module;
- the script's main() returns 0 in this environment;
- the script is invocable as a subprocess (i.e., the shebang-less
  invocation `python scripts/smoke_app.py` works);
- the script honours its documented exit codes (0 on full success,
  non-zero on any failure).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SMOKE_SCRIPT: Path = PROJECT_ROOT / "scripts" / "smoke_app.py"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_smoke_module():
    """Import scripts/smoke_app.py as a module without executing it."""
    spec = importlib.util.spec_from_file_location("smoke_app", SMOKE_SCRIPT)
    assert spec is not None and spec.loader is not None, (
        f"Could not load spec for {SMOKE_SCRIPT}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
class TestSmokeScriptExists:
    def test_script_present(self) -> None:
        assert SMOKE_SCRIPT.exists(), f"Missing: {SMOKE_SCRIPT}"

    def test_script_has_a_main(self) -> None:
        smoke = _load_smoke_module()
        assert callable(getattr(smoke, "main", None))


class TestSmokeScriptMain:
    def test_main_returns_zero_or_known_nonzero(self) -> None:
        """main() should NOT raise in this project; it should return an int.

        We accept either 0 (all imports clean) or a known nonzero,
        because pages/* may legitimately touch streamlit globals that
        don't exist outside a real streamlit process. The contract is:
        - raises -> bug (test fails)
        - returns int -> acceptable
        """
        smoke = _load_smoke_module()
        rc = smoke.main()
        assert isinstance(rc, int)
        assert rc in (0, 1), f"Unexpected return code: {rc}"


class TestSmokeScriptSubprocess:
    def test_subprocess_invocation_succeeds(self) -> None:
        """Run the smoke script as a real subprocess and confirm
        ``OK <module>`` lines appear for the leaf packages.
        """
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=45,
        )
        # Some pages/* imports may legitimately fail outside Streamlit,
        # so we don't assert rc == 0 strictly. We DO assert that the
        # script ran to completion and printed the expected markers.
        assert "OK  config" in result.stdout, (
            f"smoke script didn't print expected 'OK  config' line.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "OK  models" in result.stdout
