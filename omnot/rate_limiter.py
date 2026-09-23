from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    rpm: int = 0
    rpd: int = 0
    tpm: int = 0

    _minute_requests: deque[float] = field(default_factory=deque)
    _day_requests: deque[float] = field(default_factory=deque)
    _minute_tokens: deque[tuple[float, int]] = field(default_factory=deque)
    _exhausted_until: float = 0.0

    def allow(self) -> bool:
        now = time.monotonic()

        if now < self._exhausted_until:
            return False

        self._cleanup(now)

        if self.rpm and len(self._minute_requests) >= self.rpm:
            return False
        if self.rpd and len(self._day_requests) >= self.rpd:
            return False
        if self.tpm:
            token_sum = sum(t for _, t in self._minute_tokens)
            if token_sum >= self.tpm:
                return False

        return True

    def record(self, tokens: int = 0) -> None:
        now = time.monotonic()
        self._minute_requests.append(now)
        self._day_requests.append(now)
        if tokens > 0:
            self._minute_tokens.append((now, tokens))

    def mark_exhausted(self, cooldown: float = 60.0) -> None:
        self._exhausted_until = time.monotonic() + cooldown

    def status(self) -> dict:
        now = time.monotonic()
        self._cleanup(now)
        return {
            "requests_this_minute": len(self._minute_requests),
            "requests_today": len(self._day_requests),
            "tokens_this_minute": sum(t for _, t in self._minute_tokens),
            "exhausted": now < self._exhausted_until,
        }

    def _cleanup(self, now: float) -> None:
        minute_ago = now - 60
        while self._minute_requests and self._minute_requests[0] < minute_ago:
            self._minute_requests.popleft()

        day_ago = now - 86400
        while self._day_requests and self._day_requests[0] < day_ago:
            self._day_requests.popleft()

        while self._minute_tokens and self._minute_tokens[0][0] < minute_ago:
            self._minute_tokens.popleft()
