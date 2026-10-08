"""Sliding-window rate limiter. Clock and sleep are injected so tests never really wait."""
import threading
import time
from collections import deque
from typing import Callable


class RateLimiter:
    def __init__(
        self,
        max_per_minute: int,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        window_seconds: float = 60.0,
    ):
        if max_per_minute < 1:
            raise ValueError("max_per_minute must be at least 1")
        self._max = max_per_minute
        self._now = now
        self._sleep = sleep
        self._window = window_seconds
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until one more request is allowed, then record it. Callers queue behind each other (by design)."""
        with self._lock:
            self._wait_locked()

    def _wait_locked(self) -> None:
        while True:
            t = self._now()
            while self._stamps and t - self._stamps[0] >= self._window:
                self._stamps.popleft()
            if len(self._stamps) < self._max:
                self._stamps.append(t)
                return
            self._sleep(self._window - (t - self._stamps[0]))
