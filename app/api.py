"""HTTP façade for the project.

This package is a thin dependency-free dispatcher — see `app/__init__.py`
for the public API (`register`, `dispatch`, `ApiResponse`, `routes`).
The additional module `app.api` re-exports the dispatcher for an
import path that mirrors a future REST layer.
"""
from app import ApiResponse, NotFound, dispatch, register, routes  # noqa: F401