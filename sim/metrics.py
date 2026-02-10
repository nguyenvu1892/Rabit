# sim/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Metrics:
    steps: int = 0
    decisions: int = 0
    orders: int = 0
    executions: int = 0
    outcomes: int = 0

    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    def on_step(self):
        self.steps += 1

    def on_decision(self):
        self.decisions += 1

    def on_order(self):
        self.orders += 1

    def on_execution(self):
        self.executions += 1

    def on_outcome(self, pnl: float, win: bool):
        self.outcomes += 1
        self.total_pnl += float(pnl)
        if win:
            self.wins += 1
        else:
            self.losses += 1

    def win_rate(self) -> float:
        n = self.wins + self.losses
        return (self.wins / n) if n > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "steps": self.steps,
            "decisions": self.decisions,
            "orders": self.orders,
            "executions": self.executions,
            "outcomes": self.outcomes,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate(),
            "total_pnl": self.total_pnl,
        }
