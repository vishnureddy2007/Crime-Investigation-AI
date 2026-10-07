"""HTTP API façade (no Flask dep).

This package provides a tiny, dependency-free HTTP-style dispatcher
for programmatic access to the project's core services. It is
**not** wired into the Streamlit app — it's here so future external
clients (CLI tools, notebooks, integration tests) can call the same
business logic without spinning up a UI.

Usage::

    from app.api import register, dispatch

    @register("/health", methods=("GET",))
    def health(_payload):
        return {"status": "ok"}

    dispatch("/health", "GET", {})  # -> {"status": "ok"}

Handlers receive a dict payload and return a JSON-serialisable dict.
Errors raised inside a handler are captured into a structured
response with `ok=False` instead of bubbling up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ApiError(Exception):
    """Raise from a handler to control the HTTP status returned.

    The dispatcher maps :class:`ApiError` to ``ApiResponse(ok=False,
    status=status_code, error=message)``. ``status_code`` defaults
    to ``500``; subclasses can override.
    """

    status_code: int = 500

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class NotFound(ApiError):
    """The requested resource does not exist (HTTP 404)."""

    status_code = 404


# ----------------------------------------------------------------------
# Route registry
# ----------------------------------------------------------------------
_ROUTES: dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register(
    path: str,
    methods: tuple[str, ...] = ("POST",),
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Decorator: register `func` as the handler for `path` + `methods`.

    Paths must start with `/`. Methods are uppercased.
    """
    if not path.startswith("/"):
        raise ValueError(f"path must start with '/': {path!r}")
    normalised = path.strip()
    methods_u = tuple(m.upper() for m in methods)
    if not methods_u:
        raise ValueError("methods must be a non-empty tuple")

    def decorator(
        func: Callable[..., dict[str, Any]],
    ) -> Callable[..., dict[str, Any]]:
        for m in methods_u:
            _ROUTES[(normalised, m)] = func
        return func

    return decorator


def routes() -> list[dict[str, str]]:
    """Return a sorted list of registered routes for introspection."""
    out = []
    for (path, method), _ in _ROUTES.items():
        out.append({"path": path, "method": method})
    out.sort(key=lambda r: (r["path"], r["method"]))
    return out


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ApiResponse:
    """A structured envelope for every dispatch result."""

    ok: bool
    status: int
    data: dict[str, Any] | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"ok": self.ok, "status": self.status}
        if self.data is not None:
            d["data"] = self.data
        if self.error is not None:
            d["error"] = self.error
        return d


def dispatch(
    path: str,
    method: str,
    payload: dict[str, Any] | str | None = None,
) -> ApiResponse:
    """Route `method + path` to its handler and return an `ApiResponse`.

    `payload` may be a dict, a JSON string, or None. If JSON, it is
    parsed; invalid JSON is returned as a 400.

    The function never raises. Handlers that raise are caught and
    returned as a 500 with the exception class name.
    """
    if payload is None:
        body: dict[str, Any] = {}
    elif isinstance(payload, dict):
        body = payload
    elif isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            body = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError as exc:
            return ApiResponse(ok=False, status=400, data=None, error=f"invalid JSON: {exc}")
    else:
        return ApiResponse(ok=False, status=400, data=None, error="payload must be dict or JSON")

    key = (path.strip(), method.upper())
    handler = _ROUTES.get(key)
    if handler is None:
        return ApiResponse(ok=False, status=404, data=None, error=f"no route for {method} {path}")

    try:
        result = handler(body)
        if not isinstance(result, dict):
            return ApiResponse(
                ok=False, status=500, data=None,
                error="handler must return a dict",
            )
        return ApiResponse(ok=True, status=200, data=result)
    except ApiError as api_exc:
        return ApiResponse(
            ok=False,
            status=api_exc.status_code,
            data=None,
            error=str(api_exc) or api_exc.__class__.__name__,
        )
    except Exception as exc:
        return ApiResponse(
            ok=False, status=500, data=None,
            error=f"{exc.__class__.__name__}: {exc}",
        )


# ----------------------------------------------------------------------
# Built-in routes (always available, no domain imports required)
# ----------------------------------------------------------------------
@register("/health", methods=("GET",))
def _health(_payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok"}


@register("/routes", methods=("GET",))
def _routes(_payload: dict[str, Any]) -> dict[str, Any]:
    return {"routes": routes()}


@register("/echo", methods=("POST",))
def _echo(payload: dict[str, Any]) -> dict[str, Any]:
    return {"echo": payload}


# ----------------------------------------------------------------------
# Domain routes (lazy imports so unit tests can mock the DB)
# ----------------------------------------------------------------------
def _resolve_db_path(payload: dict[str, Any]) -> str | None:
    """Resolve the database path from the payload, env, or default."""
    candidate = payload.get("db_path")
    if isinstance(candidate, str) and candidate:
        return candidate
    try:
        from config import DATABASE_PATH  # local import keeps tests light

        return str(DATABASE_PATH)
    except (ImportError, AttributeError):
        return None


@register("/cases", methods=("GET",))
def _cases(payload: dict[str, Any]) -> dict[str, Any]:
    """List persisted cases (read-only).

    Returns 200 with an empty list when the database exists but is
    empty, and 404 (via :class:`NotFound`) when the database file is
    missing. Callers can therefore distinguish "no data" from "no
    database".
    """
    db = _resolve_db_path(payload)
    if not db:
        raise NotFound("DATABASE_PATH not configured")
    db_path = Path(db)
    if not db_path.exists():
        raise NotFound(f"database not found: {db_path}")
    from database import repository as repo
    from database.db import init_db

    init_db(db_path)
    rows = repo.list_cases(db_path)
    return {"cases": rows, "count": len(rows)}


@register("/analytics", methods=("GET",))
def _analytics(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregated KPIs across every persisted case.

    Returns 200 with an empty snapshot when the database exists but
    has no rows, and 404 when the database file is missing.
    """
    db = _resolve_db_path(payload)
    if not db:
        raise NotFound("DATABASE_PATH not configured")
    db_path = Path(db)
    if not db_path.exists():
        raise NotFound(f"database not found: {db_path}")
    from services.analytics import compute_snapshot

    snap = compute_snapshot(db_path)
    return {"snapshot": snap.as_dict()}


@register("/system", methods=("GET",))
def _system(_payload: dict[str, Any]) -> dict[str, Any]:
    """Process + Python snapshot for diagnostics."""
    from core.performance import capture_snapshot
    from core.observability import request_log

    snap = capture_snapshot()
    return {"snapshot": snap.as_dict(), "log": request_log.snapshot()}


@register("/health/deep", methods=("GET",))
def _health_deep(_payload: dict[str, Any]) -> dict[str, Any]:
    """Deep healthcheck — includes platform + observability counters."""
    from core.performance import capture_snapshot

    snap = capture_snapshot()
    from core.observability import request_log
    log_snap = request_log.snapshot()
    return {
        "status": "ok",
        "platform": snap.platform,
        "python": snap.python_version,
        "pid": snap.pid,
        "counters": log_snap["counts"],
    }


# Re-export the exception classes so callers can catch them via the
# package surface instead of reaching into ``app.app``.
__all__ = [
    "ApiError",
    "ApiResponse",
    "NotFound",
    "dispatch",
    "register",
    "routes",
]