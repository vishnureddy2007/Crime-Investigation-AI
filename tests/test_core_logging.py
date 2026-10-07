"""Unit tests for `core/logging.py`."""
from __future__ import annotations

import logging

import pytest

from core.logging import configure_logging, get_logger, reset_for_tests


@pytest.fixture(autouse=True)
def _reset_logger() -> None:
    reset_for_tests()
    yield
    reset_for_tests()


def test_configure_is_idempotent(tmp_path) -> None:
    log_file = tmp_path / "test.log"
    configure_logging("INFO", str(log_file))
    handler_count_first = len(logging.getLogger().handlers)

    configure_logging("INFO", str(log_file))
    handler_count_second = len(logging.getLogger().handlers)

    assert handler_count_first == handler_count_second


def test_configure_writes_to_file(tmp_path) -> None:
    log_file = tmp_path / "test.log"
    configure_logging("INFO", str(log_file))

    log = get_logger("test.module")
    log.warning("hello from a test")

    contents = log_file.read_text(encoding="utf-8")
    assert "hello from a test" in contents
    assert "test.module" in contents


def test_configure_unknown_level_falls_back_to_info(tmp_path) -> None:
    log_file = tmp_path / "test.log"
    # "BOGUS" is invalid — getattr returns the fallback INFO.
    configure_logging("BOGUS", str(log_file))
    assert logging.getLogger().level == logging.INFO


def test_get_logger_returns_named_logger() -> None:
    log = get_logger("my.named.logger")
    assert isinstance(log, logging.Logger)
    assert log.name == "my.named.logger"