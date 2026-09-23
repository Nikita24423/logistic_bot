import time
from unittest.mock import patch

from omnot.rate_limiter import RateLimiter


def test_allow_when_no_limits():
    limiter = RateLimiter()
    assert limiter.allow() is True


def test_rpm_limit():
    limiter = RateLimiter(rpm=2)
    limiter.record()
    limiter.record()
    assert limiter.allow() is False


def test_mark_exhausted():
    limiter = RateLimiter()
    limiter.mark_exhausted(cooldown=100)
    assert limiter.allow() is False


def test_status():
    limiter = RateLimiter(rpm=10)
    limiter.record(tokens=50)
    limiter.record(tokens=100)
    status = limiter.status()
    assert status["requests_this_minute"] == 2
    assert status["tokens_this_minute"] == 150
    assert status["exhausted"] is False
