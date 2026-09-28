"""Bound a fruitless single-target attack streak without trusting track IDs.

The detector does not currently measure a monster's HP. This is a bounded
visual-persistence fallback, not a claim that damage failed: after a long,
stationary same-species/same-location streak it first forces a fresh turn,
then briefly lets patrol continue if the visual target still persists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


Point = Tuple[float, float]


@dataclass
class _Focus:
    mob_id: str
    player: Point
    target: Point
    first_at: float
    last_at: float
    presses: int = 0
    reoriented: bool = False


class AttackStallGuard:
    def __init__(self, *, reorient_after_sec: float = 3.0,
                 suppress_after_sec: float = 6.0,
                 suppression_sec: float = 3.0) -> None:
        self.reorient_after_sec = reorient_after_sec
        self.suppress_after_sec = suppress_after_sec
        self.suppression_sec = suppression_sec
        self.focus: Optional[_Focus] = None
        self.blocked: List[Tuple[_Focus, float]] = []

    @staticmethod
    def _same(focus: _Focus, mob_id: str, player: Point, target: Point) -> bool:
        return (focus.mob_id == mob_id
                and abs(focus.player[0] - player[0]) <= 35
                and abs(focus.player[1] - player[1]) <= 25
                and abs(focus.target[0] - target[0]) <= 55
                and abs(focus.target[1] - target[1]) <= 40)

    def reset(self) -> None:
        self.focus = None
        self.blocked.clear()

    def is_suppressed(self, mob_id: str, player: Point, target: Point,
                      now: float) -> bool:
        self.blocked = [(f, until) for f, until in self.blocked if until > now]
        return any(self._same(f, mob_id, player, target) for f, _until in self.blocked)

    def before_attack(self, mob_id: str, player: Point, target: Point,
                      now: float) -> Optional[str]:
        """Return ``reorient`` or ``suppress``; otherwise allow this press."""
        if self.is_suppressed(mob_id, player, target, now):
            return "suppress"
        focus = self.focus
        if focus is None or now - focus.last_at > 1.0 or not self._same(
            focus, mob_id, player, target
        ):
            focus = _Focus(mob_id, player, target, now, now)
            self.focus = focus
        focus.last_at = now
        focus.presses += 1
        elapsed = now - focus.first_at
        if focus.reoriented and elapsed >= self.suppress_after_sec and focus.presses >= 16:
            self.blocked.append((focus, now + self.suppression_sec))
            self.focus = None
            return "suppress"
        if not focus.reoriented and elapsed >= self.reorient_after_sec and focus.presses >= 8:
            focus.reoriented = True
            return "reorient"
        return None
