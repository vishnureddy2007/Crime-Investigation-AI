"""Cross-cutting infrastructure (logging, security, theming, performance, resilience, observability, notifications).

Re-exports the public surface so callers can ``from core import safe_filename``
without reaching into the sub-modules.
"""

from __future__ import annotations

from core.logging import configure_logging, get_logger, reset_for_tests
from core.notifications import (
    Notification,
    Severity,
    format_message,
    notify,
    show_error,
    show_info,
    show_success,
    show_warning,
)
from core.observability import (
    RequestLog,
    RequestTimer,
    error_report,
    request_log,
    reset as reset_observability,
    timing_report,
)
from core.performance import (
    Stopwatch,
    SystemSnapshot,
    capture_snapshot,
    mark_startup,
    read_startup,
    uptime_seconds,
)
from core.resilience import (
    CircuitBreaker,
    CircuitOpen,
    RateLimiter,
    retry,
    retry_decorator,
)
from core.security import (
    PathTraversalError,
    UnsafeFilenameError,
    is_allowed_extension,
    safe_filename,
    safe_join,
    validate_email,
    validate_text_input,
)
from core.theming import (
    InvalidThemeError,
    available_themes,
    get_theme_css,
    render_theme_markdown,
)

__all__ = [
    "CircuitBreaker",
    "CircuitOpen",
    "InvalidThemeError",
    "Notification",
    "PathTraversalError",
    "RateLimiter",
    "RequestLog",
    "RequestTimer",
    "Severity",
    "Stopwatch",
    "SystemSnapshot",
    "UnsafeFilenameError",
    "available_themes",
    "capture_snapshot",
    "configure_logging",
    "error_report",
    "format_message",
    "get_logger",
    "get_theme_css",
    "is_allowed_extension",
    "mark_startup",
    "notify",
    "read_startup",
    "render_theme_markdown",
    "request_log",
    "reset_for_tests",
    "reset_observability",
    "retry",
    "retry_decorator",
    "safe_filename",
    "safe_join",
    "show_error",
    "show_info",
    "show_success",
    "show_warning",
    "timing_report",
    "uptime_seconds",
    "validate_email",
    "validate_text_input",
]