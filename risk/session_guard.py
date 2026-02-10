# risk/session_guard.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardStatus:
    allowed: bool
    reason: str = "ok"
    pause_remaining: int = 0


@dataclass
class SessionRiskGuard:
    daily_loss_limit: float = 50.0
    max_consecutive_losses: int = 3
    pause_steps: int = 20

    def __post_init__(self):
        self.total_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self._pause_until_step: int = -1

    def can_trade(self, step: int) -> GuardStatus:
        step = int(step)
        if self._pause_until_step >= step:
            return GuardStatus(False, "paused", pause_remaining=(self._pause_until_step - step + 1))
        return GuardStatus(True, "ok", 0)

    def on_outcome(self, step: int, pnl: float) -> GuardStatus:
        step = int(step)
        pnl = float(pnl)

        self.total_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # check triggers
        if self.total_pnl <= -abs(self.daily_loss_limit):
            self._pause_until_step = step + self.pause_steps
            return GuardStatus(False, "daily_loss_limit", pause_remaining=self.pause_steps)

        if self.consecutive_losses >= self.max_consecutive_losses:
            self._pause_until_step = step + self.pause_steps
            return GuardStatus(False, "loss_streak", pause_remaining=self.pause_steps)

        return GuardStatus(True, "ok", 0)

    def reset_session(self) -> None:
        self.total_pnl = 0.0
        self.consecutive_losses = 0
        self._pause_until_step = -1

    def get_state(self) -> dict:
        return {
        "total_pnl": self.total_pnl,
        "consecutive_losses": self.consecutive_losses,
        "_pause_until_step": self._pause_until_step,
        "daily_loss_limit": self.daily_loss_limit,
        "max_consecutive_losses": self.max_consecutive_losses,
        "pause_steps": self.pause_steps,
    }

    def set_state(self, state: dict) -> None:
        s = state or {}
        self.total_pnl = float(s.get("total_pnl", 0.0))
        self.consecutive_losses = int(s.get("consecutive_losses", 0))
        self._pause_until_step = int(s.get("_pause_until_step", -1))
