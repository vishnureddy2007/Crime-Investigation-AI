"""
Tests for the Streamlit auto-save hooks (utils/db_hooks.py).

`utils/db_hooks.py` reads `st.session_state`. To test without launching
Streamlit, we monkeypatch `utils.db_hooks.st` with a SimpleNamespace
backed by a plain dict (a per-test `fake_streamlit` fixture), then
drive the hooks against a real per-test SQLite DB.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import utils.db_hooks as hooks
from database.repository import (
    list_analyses_for_case,
    list_reports_for_case,
    list_storyboards_for_case,
    list_summaries_for_case,
)
from models.schemas import (
    EvidenceAnalysis,
    InvestigationSummary,
    ReportData,
    Storyboard,
    StoryboardScene,
)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture
def fake_streamlit(monkeypatch: pytest.MonkeyPatch, db_path: Path):
    """Replace `utils.db_hooks.st` with a dict-backed fake.

    Also initializes the SQLite schema in the per-test `db_path` so
    the hooks can actually save rows. Returns `(state_dict,
    warnings_list)` so tests can both seed session_state and inspect
    any warnings the hooks raised.
    """
    # Create the schema up-front so the auto-save hooks can write rows.
    from database.db import init_db
    init_db(db_path)

    state: dict = {}
    warnings: list[str] = []
    monkeypatch.setattr(
        hooks, "st",
        SimpleNamespace(
            session_state=state,
            warning=lambda msg, *a, **k: warnings.append(str(msg)),
        ),
    )
    return state, warnings


def _make_analysis(name: str = "x.jpg") -> EvidenceAnalysis:
    return EvidenceAnalysis(
        source_name=name, source_type="image",
        counts_by_label={"person": 1},
        total_objects=1, unique_labels=["person"],
        average_confidence=0.9,
        person_count=1, verified_weapon_count=0,
        candidate_weapon_count=0, weapon_count=0,
        vehicle_count=0, bag_count=0,
        severity_score=10, severity_level="low",
        suggested_category="suspicious_activity",
        key_observations=["one person"],
        has_threat=False, frame_count=1,
        timestamp=datetime.datetime.now(),
    )


def _make_summary(name: str = "x.jpg") -> InvestigationSummary:
    return InvestigationSummary(
        source_name=name, source_type="image",
        template_text="t", ai_text="t", used_ai=False,
        model_name="stub", generation_time_sec=0.0,
        evidence_snapshot={"x": 1},
        timestamp=datetime.datetime.now(),
    )


def _make_report(name: str = "x.jpg") -> ReportData:
    return ReportData(
        title="R", report_id="r1",
        generated_at=datetime.datetime.now(),
        source_name=name, source_type="image",
        model_name="stub", summary_text="t",
        severity_score=10, severity_level="low",
        suggested_category="suspicious_activity", has_threat=False,
        counts_by_label={}, total_objects=0,
        person_count=0, weapon_count=0, vehicle_count=0, bag_count=0,
        key_observations=[], average_confidence=0.0,
        remarks="", app_version="0.10.0",
    )


def _make_storyboard(name: str = "x.jpg") -> Storyboard:
    return Storyboard(
        source_name=name, source_type="image",
        scenes=[StoryboardScene(
            index=0, title="t", caption="c",
            image=None, duration_sec=1.0, based_on_real_frame=False,
        )],
        timestamp=datetime.datetime.now(),
    )


# ----------------------------------------------------------------------
# Per-hook write tests
# ----------------------------------------------------------------------
def test_auto_save_last_analysis_writes_to_db(
    fake_streamlit, db_path: Path,
) -> None:
    state, _ = fake_streamlit
    state["last_analysis"] = _make_analysis()
    new_id = hooks.auto_save_last_analysis(db_path=db_path)
    assert isinstance(new_id, int)
    case_id = hooks._resolve_case_id(  # type: ignore[attr-defined]
        db_path, "x.jpg", "image",
    )
    rows = list_analyses_for_case(db_path, case_id)
    assert len(rows) == 1
    assert rows[0]["analysis_id"] == new_id


def test_auto_save_last_summary_writes_to_db(
    fake_streamlit, db_path: Path,
) -> None:
    state, _ = fake_streamlit
    state["last_summary"] = _make_summary()
    hooks.auto_save_last_summary(db_path=db_path)
    case_id = hooks._resolve_case_id(  # type: ignore[attr-defined]
        db_path, "x.jpg", "image",
    )
    assert len(list_summaries_for_case(db_path, case_id)) == 1


def test_auto_save_last_report_writes_to_db(
    fake_streamlit, db_path: Path,
) -> None:
    state, _ = fake_streamlit
    state["last_report_data"] = _make_report()
    hooks.auto_save_last_report(db_path=db_path)
    case_id = hooks._resolve_case_id(  # type: ignore[attr-defined]
        db_path, "x.jpg", "image",
    )
    assert len(list_reports_for_case(db_path, case_id)) == 1


def test_auto_save_last_storyboard_writes_to_db(
    fake_streamlit, db_path: Path,
) -> None:
    state, _ = fake_streamlit
    state["last_storyboard"] = _make_storyboard()
    hooks.auto_save_last_storyboard(db_path=db_path)
    case_id = hooks._resolve_case_id(  # type: ignore[attr-defined]
        db_path, "x.jpg", "image",
    )
    assert len(list_storyboards_for_case(db_path, case_id)) == 1


# ----------------------------------------------------------------------
# Behavior
# ----------------------------------------------------------------------
def test_auto_save_is_idempotent(fake_streamlit, db_path: Path) -> None:
    state, _ = fake_streamlit
    state["last_analysis"] = _make_analysis()
    a1 = hooks.auto_save_last_analysis(db_path=db_path)
    a2 = hooks.auto_save_last_analysis(db_path=db_path)
    assert a1 == a2
    # Only one row should exist
    case_id = hooks._resolve_case_id(  # type: ignore[attr-defined]
        db_path, "x.jpg", "image",
    )
    assert len(list_analyses_for_case(db_path, case_id)) == 1


def test_auto_save_returns_none_when_no_session_value(
    fake_streamlit, db_path: Path,
) -> None:
    assert hooks.auto_save_last_analysis(db_path=db_path) is None
    assert hooks.auto_save_last_summary(db_path=db_path) is None
    assert hooks.auto_save_last_report(db_path=db_path) is None
    assert hooks.auto_save_last_storyboard(db_path=db_path) is None


def test_auto_save_swallows_exceptions(
    fake_streamlit, db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ = fake_streamlit
    state["last_summary"] = _make_summary()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(hooks, "save_summary", _boom)
    # Must not raise; must return None.
    result = hooks.auto_save_last_summary(db_path=db_path)
    assert result is None
    # Must have produced a warning.
    assert any("auto-save failed" in w.lower() for w in fake_streamlit[1])


# ----------------------------------------------------------------------
# Phase 19 — defensive coverage of _warn + every hook's error path
# ----------------------------------------------------------------------


def test_warn_swallows_streamlit_exception(monkeypatch) -> None:
    """If `st.warning` raises, the hook must still complete silently."""
    import types

    def _raise(*_a, **_kw):
        raise RuntimeError("streamlit died")

    fake_st = types.SimpleNamespace(
        session_state={},
        warning=_raise,
    )
    monkeypatch.setattr(hooks, "st", fake_st)
    # _warn must not raise.
    hooks._warn("anything")  # type: ignore[attr-defined]


def test_auto_save_analysis_handles_save_exception(
    fake_streamlit, db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save_analysis exception must be swallowed with a warning."""
    state, warnings = fake_streamlit
    state["last_analysis"] = _make_analysis()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("sqlite broken")

    monkeypatch.setattr(hooks, "save_analysis", _boom)
    result = hooks.auto_save_last_analysis(db_path=db_path)
    assert result is None
    assert any("auto-save failed" in w.lower() for w in warnings)


def test_auto_save_report_handles_save_exception(
    fake_streamlit, db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, warnings = fake_streamlit
    state["last_report_data"] = _make_report()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("report broken")

    monkeypatch.setattr(hooks, "save_report", _boom)
    result = hooks.auto_save_last_report(db_path=db_path)
    assert result is None
    assert any("auto-save failed" in w.lower() for w in warnings)


def test_auto_save_storyboard_handles_save_exception(
    fake_streamlit, db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, warnings = fake_streamlit
    state["last_storyboard"] = _make_storyboard()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("storyboard broken")

    monkeypatch.setattr(hooks, "save_storyboard", _boom)
    result = hooks.auto_save_last_storyboard(db_path=db_path)
    assert result is None
    assert any("auto-save failed" in w.lower() for w in warnings)


def test_resolve_path_uses_default_when_none(monkeypatch) -> None:
    """`_resolve_path(None)` falls back to DATABASE_PATH."""
    from pathlib import Path
    sentinel = Path("from-config.sqlite")
    monkeypatch.setattr("utils.db_hooks.DATABASE_PATH", sentinel, raising=False)
    assert hooks._resolve_path(None) == sentinel  # type: ignore[attr-defined]


def test_resolve_path_passes_through_when_provided() -> None:
    from pathlib import Path
    explicit = Path("explicit.sqlite")
    assert hooks._resolve_path(explicit) == explicit  # type: ignore[attr-defined]


def test_resolve_case_id_reuses_existing(fake_streamlit, db_path: Path) -> None:
    """An existing case for this source is reused, not duplicated."""
    state, _ = fake_streamlit
    state["last_analysis"] = _make_analysis("shared.png")
    hooks.auto_save_last_analysis(db_path=db_path)
    case_id_1 = hooks._resolve_case_id(db_path, "shared.png", "image")  # type: ignore[attr-defined]
    case_id_2 = hooks._resolve_case_id(db_path, "shared.png", "image")  # type: ignore[attr-defined]
    assert case_id_1 == case_id_2
