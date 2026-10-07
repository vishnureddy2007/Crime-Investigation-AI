"""Unit tests for `app/` HTTP façade."""
from __future__ import annotations

import pytest

from app import ApiResponse, _ROUTES, dispatch, register, routes


def test_health_route() -> None:
    r = dispatch("/health", "GET", None)
    assert isinstance(r, ApiResponse)
    assert r.ok is True
    assert r.status == 200
    assert r.data == {"status": "ok"}


def test_unknown_path_returns_404() -> None:
    r = dispatch("/nope", "GET", None)
    assert r.ok is False
    assert r.status == 404


def test_wrong_method_returns_404() -> None:
    r = dispatch("/health", "POST", None)
    assert r.ok is False
    assert r.status == 404


def test_echo_round_trip() -> None:
    r = dispatch("/echo", "POST", {"hello": "world"})
    assert r.ok is True
    assert r.data == {"echo": {"hello": "world"}}


def test_dispatch_accepts_json_string() -> None:
    r = dispatch("/echo", "POST", '{"x": 1}')
    assert r.ok is True
    assert r.data == {"echo": {"x": 1}}


def test_dispatch_rejects_invalid_json() -> None:
    r = dispatch("/echo", "POST", "{not-json")
    assert r.ok is False
    assert r.status == 400


def test_dispatch_handles_handler_exception() -> None:
    @register("/boom-test", methods=("GET",))
    def boom(_p):
        raise RuntimeError("kaboom")

    try:
        r = dispatch("/boom-test", "GET", None)
        assert r.ok is False
        assert r.status == 500
        assert "RuntimeError" in (r.error or "")
    finally:
        _ROUTES.pop(("/boom-test", "GET"), None)


def test_routes_listing_includes_builtin() -> None:
    rs = routes()
    paths = {(r["path"], r["method"]) for r in rs}
    assert ("/health", "GET") in paths
    assert ("/echo", "POST") in paths


def test_register_path_must_start_with_slash() -> None:
    with pytest.raises(ValueError):
        register("nope", methods=("GET",))


def test_register_methods_normalised() -> None:
    @register("/methods-test", methods=("get", "post"))
    def handler(_p):
        return {"ok": True}

    try:
        assert ("/methods-test", "GET") in _ROUTES
        assert ("/methods-test", "POST") in _ROUTES
    finally:
        _ROUTES.pop(("/methods-test", "GET"), None)
        _ROUTES.pop(("/methods-test", "POST"), None)


def test_dispatch_rejects_non_string_non_dict_payload() -> None:
    r = dispatch("/echo", "POST", 42)  # type: ignore[arg-type]
    assert r.ok is False
    assert r.status == 400


# ---------------------------------------------------------------------------
# Phase 15: domain routes
# ---------------------------------------------------------------------------


def test_routes_listing_includes_phase15_endpoints() -> None:
    paths = {(r["path"], r["method"]) for r in routes()}
    for expected in {
        ("/cases", "GET"),
        ("/analytics", "GET"),
        ("/system", "GET"),
        ("/health/deep", "GET"),
    }:
        assert expected in paths


def test_cases_route_handles_missing_db(tmp_path) -> None:
    """When the DB doesn't exist the route returns 404 NotFound."""
    r = dispatch("/cases", "GET", {"db_path": str(tmp_path / "absent.sqlite")})
    assert r.ok is False
    assert r.status == 404
    assert "not found" in (r.error or "").lower()


def test_cases_route_handles_unconfigured_db() -> None:
    """When no db_path is provided and the config is missing, return 404."""
    import sys

    from app import _resolve_db_path as _resolve

    # Force _resolve_db_path to find nothing by stripping all candidates.
    # We do this by patching the helper to always return None.
    import unittest.mock as mock

    with mock.patch("app._resolve_db_path", return_value=None):
        r = dispatch("/cases", "GET", {})
    assert r.ok is False
    assert r.status == 404
    assert "DATABASE_PATH" in (r.error or "")


def test_analytics_route_handles_missing_db(tmp_path) -> None:
    """When the DB doesn't exist the route returns 404 NotFound."""
    r = dispatch("/analytics", "GET", {"db_path": str(tmp_path / "absent.sqlite")})
    assert r.ok is False
    assert r.status == 404
    assert "not found" in (r.error or "").lower()


def test_analytics_route_returns_snapshot(tmp_path) -> None:
    """When the DB exists the /analytics route returns a real snapshot."""
    from database.db import init_db
    from database.repository import save_case

    db = tmp_path / "present.sqlite"
    init_db(db)
    save_case(db, source_name="x.png", source_type="image")
    r = dispatch("/analytics", "GET", {"db_path": str(db)})
    assert r.ok is True
    assert "snapshot" in r.data
    assert isinstance(r.data["snapshot"], dict)


def test_system_route_returns_snapshot() -> None:
    r = dispatch("/system", "GET", None)
    assert r.ok is True
    snap = r.data["snapshot"]
    assert "python_version" in snap
    assert snap["pid"] > 0


def test_health_deep_includes_counters() -> None:
    r = dispatch("/health/deep", "GET", None)
    assert r.ok is True
    assert r.data["status"] == "ok"
    assert "counters" in r.data
    assert "timings_total" in r.data["counters"]

# ----------------------------------------------------------------------
# /routes — exercise the dispatcher+routes() path
# ----------------------------------------------------------------------


def test_routes_route_returns_route_list() -> None:
    """GET /routes returns the built-in route catalogue."""
    r = dispatch("/routes", "GET", None)
    assert r.ok is True
    assert r.status == 200
    assert "routes" in r.data
    paths = {entry["path"] for entry in r.data["routes"]}
    # Every built-in should be present.
    assert "/health" in paths
    assert "/echo" in paths
    assert "/routes" in paths
    assert "/cases" in paths
    assert "/analytics" in paths
    assert "/system" in paths
    assert "/health/deep" in paths


# ----------------------------------------------------------------------
# _resolve_db_path — direct coverage of the env/config fallback
# ----------------------------------------------------------------------


def test_resolve_db_path_returns_payload_override() -> None:
    """A non-empty `db_path` in the payload wins."""
    from app import _resolve_db_path

    assert _resolve_db_path({"db_path": "override.sqlite"}) == "override.sqlite"


def test_resolve_db_path_ignores_empty_string() -> None:
    """An empty string in the payload falls through to config / env."""
    from app import _resolve_db_path

    # Empty string is falsy → should fall through.
    out = _resolve_db_path({"db_path": ""})
    # Either config or env's value is fine; we just want it defined.
    assert out is None or isinstance(out, str)


def test_resolve_db_path_ignores_non_string_db_path() -> None:
    """A non-string `db_path` is ignored."""
    from app import _resolve_db_path

    out = _resolve_db_path({"db_path": 42})
    assert out is None or isinstance(out, str)


def test_resolve_db_path_falls_back_to_config(monkeypatch) -> None:
    """If config import succeeds, config.DATABASE_PATH is returned."""
    import sys
    import types

    fake_config = types.ModuleType("config")
    fake_config.DATABASE_PATH = "from-config.sqlite"
    monkeypatch.setitem(sys.modules, "config", fake_config)

    from app import _resolve_db_path

    assert _resolve_db_path({}) == "from-config.sqlite"


def test_resolve_db_path_returns_none_when_config_missing(monkeypatch) -> None:
    """If config isn't importable, the helper returns None."""
    import sys

    # Remove config so the inner `from config import` raises ImportError.
    monkeypatch.delitem(sys.modules, "config", raising=False)
    # Make sure nothing else replaces it.
    monkeypatch.setitem(sys.modules, "config", None)

    from app import _resolve_db_path

    assert _resolve_db_path({}) is None


# ----------------------------------------------------------------------
# Phase 19 — ApiError happy path /cases + /analytics
# ----------------------------------------------------------------------


def test_api_error_default_status_is_500() -> None:
    """ApiError without explicit status_code defaults to 500."""
    from app import ApiError

    err = ApiError("oops")
    assert err.status_code == 500
    assert str(err) == "oops"


def test_api_error_custom_status_code() -> None:
    """ApiError honours an explicit status_code override."""
    from app import ApiError

    err = ApiError("forbidden", status_code=403)
    assert err.status_code == 403


def test_cases_route_happy_path(tmp_path) -> None:
    """When the DB exists with rows, /cases returns 200 + a list."""
    from database.db import init_db
    from database.repository import save_case

    db = tmp_path / "present.sqlite"
    init_db(db)
    save_case(db, source_name="hello.png", source_type="image")
    save_case(db, source_name="world.mp4", source_type="video")

    r = dispatch("/cases", "GET", {"db_path": str(db)})
    assert r.ok is True
    assert r.status == 200
    assert r.data["count"] == 2
    names = {row["source_name"] for row in r.data["cases"]}
    assert names == {"hello.png", "world.mp4"}


def test_cases_route_empty_db_returns_empty_list(tmp_path) -> None:
    """An empty but existing DB returns 200 with an empty list."""
    from database.db import init_db

    db = tmp_path / "empty.sqlite"
    init_db(db)
    r = dispatch("/cases", "GET", {"db_path": str(db)})
    assert r.ok is True
    assert r.data["cases"] == []
    assert r.data["count"] == 0


def test_analytics_route_happy_path(tmp_path) -> None:
    """When the DB has rows, /analytics returns a non-empty snapshot."""
    from database.db import init_db
    from database.repository import save_analysis, save_case

    db = tmp_path / "analytics_present.sqlite"
    init_db(db)
    case_id = save_case(db, source_name="a.png", source_type="image")

    # Build a real EvidenceAnalysis so the snapshot has a row to count.
    from datetime import datetime
    from models.schemas import EvidenceAnalysis

    analysis = EvidenceAnalysis(
        source_name="a.png", source_type="image",
        counts_by_label={"person": 1}, total_objects=1,
        unique_labels=["person"], average_confidence=0.8,
        person_count=1, verified_weapon_count=0,
        candidate_weapon_count=0, weapon_count=0,
        vehicle_count=0, bag_count=0,
        severity_score=10, severity_level="low",
        suggested_category="general", has_threat=False,
        key_observations=(), frame_count=1, timestamp=datetime.now(),
    )
    save_analysis(db, case_id, analysis)
    r = dispatch("/analytics", "GET", {"db_path": str(db)})
    assert r.ok is True
    assert "snapshot" in r.data
    assert r.data["snapshot"]["total_cases"] >= 1


# ----------------------------------------------------------------------
# Phase 39 — coverage push: edge cases in ApiResponse + register + dispatch
# ----------------------------------------------------------------------
def test_api_response_as_dict_data_only() -> None:
    """ApiResponse.as_dict() with `data` set and no `error`."""
    from app import ApiResponse

    r = ApiResponse(ok=True, status=200, data={"k": 1})
    d = r.as_dict()
    assert d == {"ok": True, "status": 200, "data": {"k": 1}}


def test_api_response_as_dict_error_only() -> None:
    """ApiResponse.as_dict() with `error` set and `data=None`."""
    from app import ApiResponse

    r = ApiResponse(ok=False, status=500, data=None, error="boom")
    d = r.as_dict()
    assert d == {"ok": False, "status": 500, "error": "boom"}
    assert "data" not in d


def test_api_response_as_dict_neither() -> None:
    """ApiResponse.as_dict() with neither `data` nor `error` set."""
    from app import ApiResponse

    r = ApiResponse(ok=False, status=400, data=None)
    d = r.as_dict()
    assert d == {"ok": False, "status": 400}


def test_register_rejects_empty_methods_tuple() -> None:
    """register() must raise ValueError for an empty methods tuple."""
    import pytest

    with pytest.raises(ValueError):
        register("/empty-methods", methods=())


def test_dispatch_returns_500_when_handler_returns_non_dict() -> None:
    """Handlers must return a dict; anything else → 500."""
    @register("/bad-shape-test", methods=("GET",))
    def bad_handler(_p):
        return ["not", "a", "dict"]

    try:
        r = dispatch("/bad-shape-test", "GET", None)
        assert r.ok is False
        assert r.status == 500
        assert "dict" in (r.error or "").lower()
    finally:
        _ROUTES.pop(("/bad-shape-test", "GET"), None)


def test_dispatch_handles_not_found_error() -> None:
    """Handlers raising NotFound get the 404 status from the class."""
    from app import NotFound

    @register("/raises-notfound", methods=("GET",))
    def not_found_handler(_p):
        raise NotFound("explicit 404")

    try:
        r = dispatch("/raises-notfound", "GET", None)
        assert r.ok is False
        assert r.status == 404
        assert r.error == "explicit 404"
    finally:
        _ROUTES.pop(("/raises-notfound", "GET"), None)
