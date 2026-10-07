"""
Resilience helpers — retry, rate-limit, circuit-breaker-lite.

Pure stdlib. Designed for **light** use:
- `retry` retries a callable with exponential backoff.
- `RateLimiter` is a token-bucket gate for `acquire()`.
- `CircuitBreaker` opens after N consecutive failures and short-circuits
  subsequent calls for a cooldown window.

These are not full battle-tested primitives; they're the minimum useful
shapes for the demo so we can wrap flakier integrations (FLAN-T5 cold
load, file I/O, etc.) without sprinkling try/except everywhere.
"""

from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def retry(
    func: Callable[..., T],
    *args: Any,
    attempts: int = 3,
    initial_delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
    **kwargs: Any,
) -> T:
    """Call ``func(*args, **kwargs)`` with exponential-backoff retry.

    - ``attempts`` — total attempts including the first try (>=1).
    - ``initial_delay`` — seconds to wait before retry #2.
    - ``backoff`` — multiplier for each subsequent wait.
    - ``exceptions`` — tuple of exceptions that trigger a retry.
    - ``on_retry`` — optional hook called with ``(attempt_index, exception)``
      immediately before sleeping.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    delay = initial_delay
    last_exc: BaseException | None = None
    for index in range(attempts):
        try:
            return func(*args, **kwargs)
        except exceptions as exc:  # noqa: PERF203 - explicit handling
            last_exc = exc
            if index == attempts - 1:
                break
            if on_retry is not None:
                on_retry(index + 1, exc)
            time.sleep(delay)
            delay *= backoff
    assert last_exc is not None  # for type-checkers
    raise last_exc


def retry_decorator(
    attempts: int = 3,
    initial_delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of :func:`retry`."""

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry(
                fn,
                *args,
                attempts=attempts,
                initial_delay=initial_delay,
                backoff=backoff,
                exceptions=exceptions,
                **kwargs,
            )

        return wrapper

    return decorate


# ---------------------------------------------------------------------------
# Rate limiter (token bucket)
# ---------------------------------------------------------------------------


@dataclass
class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    ``capacity`` tokens are added at ``refill_rate`` tokens per second
    up to ``capacity``. :meth:`acquire` consumes one token, blocking
    if necessary.
    """

    capacity: int
    refill_rate: float
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")
        self._tokens = float(self.capacity)
        self._last = time.monotonic()

    def acquire(self, tokens: float = 1.0, *, blocking: bool = True, timeout: float | None = None) -> bool:
        """Consume one (or ``tokens``) tokens.

        Returns ``True`` on success, ``False`` if the limiter is in
        non-blocking mode (or the timeout elapses) and the bucket
        was empty.
        """
        if tokens <= 0:
            raise ValueError("tokens must be > 0")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                if not blocking:
                    return False
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                deficit = tokens - self._tokens
                needed_wait = max(deficit / self.refill_rate, 0.001)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait = min(needed_wait, remaining)
                else:
                    wait = needed_wait
                # Release the lock while we sleep so other threads can refill.
                self._lock.release()
                try:
                    time.sleep(max(wait, 0.001))
                finally:
                    self._lock.acquire()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(float(self.capacity), self._tokens + elapsed * self.refill_rate)
            self._last = now

    def available(self) -> float:
        """Return current available tokens (approximate)."""
        with self._lock:
            self._refill()
            return self._tokens


# ---------------------------------------------------------------------------
# Circuit breaker (simple consecutive-failure variant)
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreaker:
    """Consecutive-failure breaker.

    After ``failure_threshold`` consecutive failures the breaker opens
    for ``cooldown_seconds``, during which :meth:`call` raises
    :class:`CircuitOpen` immediately. A single success resets the
    failure count.
    """

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")

    @property
    def is_open(self) -> bool:
        """True if the breaker is currently rejecting calls."""
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                # Cooldown elapsed — close the breaker, reset counter.
                self._opened_at = None
                self._failures = 0
                return False
            return True

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Invoke ``func`` under the breaker policy."""
        if self.is_open:
            raise CircuitOpen("breaker is open")
        try:
            result = func(*args, **kwargs)
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._opened_at = time.monotonic()
            raise
        else:
            with self._lock:
                self._failures = 0
            return result


class CircuitOpen(RuntimeError):
    """Raised when a call is short-circuited by an open breaker."""