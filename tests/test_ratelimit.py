import pytest

from safety_signal.ratelimit import RateLimiter
from tests.fakes import FakeClock


def test_never_exceeds_cap_in_any_window():
    clock = FakeClock()
    limiter = RateLimiter(3, now=clock.now, sleep=clock.sleep)
    stamps = []
    for _ in range(10):
        limiter.wait()
        stamps.append(clock.now())
    for i in range(len(stamps) - 3):
        assert stamps[i + 3] - stamps[i] >= 60.0


def test_no_waiting_under_the_cap():
    clock = FakeClock()
    limiter = RateLimiter(5, now=clock.now, sleep=clock.sleep)
    for _ in range(5):
        limiter.wait()
    assert clock.sleeps == []


def test_old_requests_age_out():
    clock = FakeClock()
    limiter = RateLimiter(1, now=clock.now, sleep=clock.sleep)
    limiter.wait()
    clock.t += 61
    limiter.wait()
    assert clock.sleeps == []


def test_invalid_cap_rejected():
    with pytest.raises(ValueError):
        RateLimiter(0)
