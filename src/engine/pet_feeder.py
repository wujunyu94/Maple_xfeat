"""Monotonic F6-session timer for automatic pet feeding."""

from __future__ import annotations


class PetFeedTimer:
    """The first feed is due one full interval after F6 starts."""

    def __init__(self) -> None:
        self.session_id = None
        self.interval_sec = 0.0
        self.next_due_at = 0.0

    def reset(self) -> None:
        self.session_id = None
        self.interval_sec = 0.0
        self.next_due_at = 0.0

    def due(self, *, session_id: int, started_at: float,
            interval_sec: float, now: float) -> bool:
        interval_sec = max(1.0, float(interval_sec))
        if self.session_id != session_id:
            self.session_id = session_id
            self.interval_sec = interval_sec
            self.next_due_at = float(started_at) + interval_sec
        elif self.interval_sec != interval_sec:
            # A live settings change starts a new full wait; it must not feed
            # immediately because the previous shorter interval has elapsed.
            self.interval_sec = interval_sec
            self.next_due_at = now + interval_sec
        if now < self.next_due_at:
            return False
        self.next_due_at = now + interval_sec
        return True
