# executor/reentry_guard.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ReentryDecision:
    allowed: bool
    reason: str = ""


@dataclass
class ReentryGuard:
    """
    Prevent order spam per symbol.
    Uses integer trade_count (monotonic) for deterministic checks.
    """
    cooldown_trades: int = 10

    def __post_init__(self):
        self._last_trade_count: Dict[str, int] = {}

    def can_enter(self, symbol: str, trade_count: int) -> ReentryDecision:
        symbol = str(symbol)
        trade_count = int(trade_count)

        last = self._last_trade_count.get(symbol)
        if last is None:
            return ReentryDecision(True, "ok")

        if trade_count - last < self.cooldown_trades:
            return ReentryDecision(False, "cooldown_active")

        return ReentryDecision(True, "ok")

    def mark_entered(self, symbol: str, trade_count: int) -> None:
        self._last_trade_count[str(symbol)] = int(trade_count)

    def get_state(self) -> dict:
        return {
        "cooldown_trades": self.cooldown_trades,
        "_last_trade_count": dict(self._last_trade_count),
    }

    def set_state(self, state: dict) -> None:
        s = state or {}
        self._last_trade_count = dict(s.get("_last_trade_count", {}))
