"""Centralized logging setup.

A single `configure_logging()` call wires:

- a rotating file handler at `logs/app.log` (10 MB × 5 backups)
- a stream handler at WARNING level (so Streamlit's stderr isn't drowned)
- an idempotent guard: calling `configure_logging()` twice is safe

Usage::

    from core.logging import configure_logging, get_logger
    configure_logging("INFO")
    log = get_logger(__name__)
    log.info("hello")

The module deliberately avoids touching `print` so Streamlit's own
log spinners and balloons still work.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Final

_CONFIGURED: bool = False

_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
)
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

_DEFAULT_LEVEL: Final[str] = os.environ.get("LOG_LEVEL", "INFO")


def _resolve_log_file(override: str | None) -> Path:
    """Resolve the log file path from override → config → env → default.

    Order matters: explicit override wins, then the canonical
    :data:`config.LOG_FILE` (so the project has a single source of
    truth), then the ``LOG_FILE`` environment variable, then
    ``logs/app.log`` relative to the current working directory.
    """
    if override:
        return Path(override)
    try:
        from config import LOG_FILE  # type: ignore[import-not-found]
        return Path(LOG_FILE)
    except (ImportError, AttributeError):
        pass
    env_value = os.environ.get("LOG_FILE")
    if env_value:
        return Path(env_value)
    return Path("logs") / "app.log"


def configure_logging(
    level: str | None = None,
    log_file: str | None = None,
) -> Path:
    """Configure root + project loggers. Returns the log file path.

    Idempotent — safe to call from multiple pages or from a script.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return _resolve_log_file(log_file)

    level_name = (level or _DEFAULT_LEVEL).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    file_path = _resolve_log_file(log_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Clear any handlers Streamlit or another lib may have attached.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(max(numeric_level, logging.WARNING))
    root.addHandler(stream_handler)

    # Quiet noisy third-party loggers.
    for noisy in ("ultralytics", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Logging configured: level=%s file=%s", level_name, file_path
    )
    return file_path


def get_logger(name: str) -> logging.Logger:
    """Return a logger; configure logging lazily if not yet done."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


def reset_for_tests() -> None:
    """Drop the configured flag so the next call re-installs handlers.

    Used by tests that want a clean root logger between cases.
    """
    global _CONFIGURED
    _CONFIGURED = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)