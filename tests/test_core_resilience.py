"""Tests for ``core.resilience`` — retry, rate limiter, circuit breaker."""

from __future__ import annotations

import time

import pytest

from core.resilience import (
    CircuitBreaker,
    CircuitOpen,
    RateLimiter,
    retry,
    retry_decorator,
)


# ---- retry ----------------------------------------------------------------


def test_retry_succeeds_first_try() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "ok"

    assert retry(fn, attempts=3) == "ok"
    assert calls["n"] == 1


def test_retry_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert retry(fn, attempts=4, initial_delay=0.001, backoff=1.0) == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_exhaustion() -> None:
    def fn() -> None:
        raise ValueError("always")

    with pytest.raises(ValueError, match="always"):
        retry(fn, attempts=2, initial_delay=0.001, backoff=1.0)


def test_retry_does_not_catch_unexpected_exception() -> None:
    def fn() -> None:
        raise KeyError("different")

    with pytest.raises(KeyError):
        retry(fn, attempts=3, exceptions=(ValueError,), initial_delay=0.001)


def test_retry_decorator() -> None:
    calls = {"n": 0}

    @retry_decorator(attempts=3, initial_delay=0.001)
    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("boom")
        return "ok"

    assert fn() == "ok"
    assert calls["n"] == 2


def test_retry_rejects_bad_attempts() -> None:
    with pytest.raises(ValueError):
        retry(lambda: None, attempts=0)


def test_retry_invokes_on_retry_hook() -> None:
    seen: list[tuple[int, Exception]] = []

    def on_retry(idx: int, exc: Exception) -> None:
        seen.append((idx, exc))

    def fn() -> str:
        if not seen:
            raise RuntimeError("first")
        return "ok"

    assert retry(fn, attempts=2, initial_delay=0.001, on_retry=on_retry) == "ok"
    assert len(seen) == 1
    assert seen[0][0] == 1


# ---- rate limiter ---------------------------------------------------------


def test_rate_limiter_acquire_consumes_token() -> None:
    rl = RateLimiter(capacity=2, refill_rate=1.0)
    assert rl.acquire(blocking=False)
    assert rl.acquire(blocking=False)
    assert rl.acquire(blocking=False) is False


def test_rate_limiter_refills_over_time() -> None:
    rl = RateLimiter(capacity=1, refill_rate=200.0)
    assert rl.acquire(blocking=False)
    time.sleep(0.02)
    assert rl.acquire(blocking=False)


def test_rate_limiter_validates_inputs() -> None:
    with pytest.raises(ValueError):
        RateLimiter(capacity=0, refill_rate=1.0)
    with pytest.raises(ValueError):
        RateLimiter(capacity=1, refill_rate=0.0)
    rl = RateLimiter(capacity=1, refill_rate=1.0)
    with pytest.raises(ValueError):
        rl.acquire(tokens=0.0)


def test_rate_limiter_blocks_until_available() -> None:
    rl = RateLimiter(capacity=1, refill_rate=100.0)
    assert rl.acquire(blocking=False)
    assert rl.acquire(blocking=True, timeout=0.2)


def test_rate_limiter_timeout_returns_false() -> None:
    # Use a very slow refill (1 token per 10 seconds) so 50 ms isn't
    # enough to produce a fresh token.
    rl = RateLimiter(capacity=1, refill_rate=0.1)
    assert rl.acquire(blocking=False)
    assert rl.acquire(blocking=True, timeout=0.05) is False


# ---- circuit breaker ------------------------------------------------------


def test_circuit_breaker_opens_after_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.5)

    def bad() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cb.call(bad)
    with pytest.raises(RuntimeError):
        cb.call(bad)
    assert cb.is_open
    with pytest.raises(CircuitOpen):
        cb.call(bad)


def test_circuit_breaker_resets_on_success() -> None:
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.5)

    def flaky() -> str:
        if not hasattr(flaky, "called"):
            flaky.called = True  # type: ignore[attr-defined]
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(RuntimeError):
        cb.call(flaky)
    assert cb.call(flaky) == "ok"
    assert not cb.is_open


def test_circuit_breaker_closes_after_cooldown() -> None:
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

    def bad() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cb.call(bad)
    assert cb.is_open
    time.sleep(0.1)
    assert not cb.is_open


def test_circuit_breaker_validates_inputs() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)