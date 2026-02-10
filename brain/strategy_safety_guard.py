# brain/strategy_safety_guard.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GuardDecision:
    allowed: bool
    reason: str


@dataclass
class RollbackDecision:
    rollback: bool
    reason: str


class StrategySafetyGuard:
    """
    V1 Safety Guard:
      - cooldown_trades: minimum trades between upgrades
      - min_samples: require samples before allowing upgrade
      - grace_trades: number of trades to evaluate new strategy before considering rollback
      - rollback_drop: rollback if fitness drops by >= rollback_drop vs baseline
    """

    def __init__(
        self,
        cooldown_trades: int = 50,
        min_samples: int = 30,
        grace_trades: int = 30,
        rollback_drop: float = 0.05,
    ):
        self.cooldown_trades = int(cooldown_trades)
        self.min_samples = int(min_samples)
        self.grace_trades = int(grace_trades)
        self.rollback_drop = float(rollback_drop)

        # state
        self.last_upgrade_trade_count = 0
        self.baseline_fitness = None  # fitness of strategy before upgrade
        self.new_strategy_trade_count_at_apply = None
        self.new_strategy_fitness_at_apply = None

    def can_upgrade(self, current_trade_count: int, current_samples: int) -> GuardDecision:
        if current_samples < self.min_samples:
            return GuardDecision(False, "not_enough_samples")

        if (current_trade_count - self.last_upgrade_trade_count) < self.cooldown_trades:
            return GuardDecision(False, "cooldown_active")

        return GuardDecision(True, "allowed")

    def mark_upgraded(
        self,
        current_trade_count: int,
        old_fitness: float,
        new_fitness: float,
    ) -> None:
        self.last_upgrade_trade_count = current_trade_count
        self.baseline_fitness = float(old_fitness)
        self.new_strategy_trade_count_at_apply = int(current_trade_count)
        self.new_strategy_fitness_at_apply = float(new_fitness)

    def should_rollback(self, current_trade_count: int, observed_fitness: float) -> RollbackDecision:
        """
        Evaluate rollback after grace window.
        observed_fitness: fitness computed from recent outcomes while new strategy is active.
        """
        if self.new_strategy_trade_count_at_apply is None or self.baseline_fitness is None:
            return RollbackDecision(False, "no_active_upgrade")

        trades_since_apply = current_trade_count - self.new_strategy_trade_count_at_apply
        if trades_since_apply < self.grace_trades:
            return RollbackDecision(False, "grace_window_active")

        # rollback condition
        drop = float(self.baseline_fitness) - float(observed_fitness)
        if drop >= self.rollback_drop:
            return RollbackDecision(True, "fitness_drop")

        return RollbackDecision(False, "stable")
