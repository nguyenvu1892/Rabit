# brain/strategy_upgrade_scheduler.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from brain.strategy_store import StrategyStore
from brain.strategy_evolver import evolve, EvolutionResult
from brain.strategy_evaluator import StrategyEvaluator
from brain.strategy_safety_guard import StrategySafetyGuard


@dataclass
class UpgradeReport:
    triggered: bool
    upgraded: bool
    reason: str
    old_fitness: float
    new_fitness: float
    old_genome: Dict[str, Any]
    new_genome: Dict[str, Any]


@dataclass
class RollbackReport:
    checked: bool
    rolled_back: bool
    reason: str


class StrategyUpgradeScheduler:
    def __init__(
        self,
        trade_memory,
        decision_engine,
        store: Optional[StrategyStore] = None,
        guard: Optional[StrategySafetyGuard] = None,
        every_n_trades: int = 50,
        min_improve: float = 0.01,
        previous_path: str = "data/previous_strategy.json",
    ):
        self.trade_memory = trade_memory
        self.decision_engine = decision_engine
        self.store = store or StrategyStore()
        self.guard = guard or StrategySafetyGuard()
        self.every_n_trades = int(every_n_trades)
        self.min_improve = float(min_improve)
        self.previous_path = previous_path

        self._last_checked_trades = 0

    def maybe_upgrade(self, rng=None) -> UpgradeReport:
        trade_count, samples = self._get_trade_count_and_samples()

        # Guard gate (cooldown + min samples)
        gd = self.guard.can_upgrade(current_trade_count=trade_count, current_samples=samples)
        if not gd.allowed:
            return UpgradeReport(False, False, gd.reason, 0.0, 0.0, {}, {})

        if self.every_n_trades <= 0:
            return UpgradeReport(False, False, "every_n_trades_disabled", 0.0, 0.0, {}, {})

        if trade_count < self.every_n_trades:
            return UpgradeReport(False, False, "not_enough_trades_yet", 0.0, 0.0, {}, {})

        if (trade_count // self.every_n_trades) == (self._last_checked_trades // self.every_n_trades):
            return UpgradeReport(False, False, "no_new_upgrade_window", 0.0, 0.0, {}, {})

        self._last_checked_trades = trade_count

        evaluator = StrategyEvaluator(self.trade_memory)

        old_genome = self.store.load() or {}
        old_stats = evaluator.evaluate(old_genome)
        old_fitness = self._fitness_from_stats(old_stats)

        res: EvolutionResult = evolve(
            evaluator=evaluator.evaluate,
            generations=5,
            population_size=20,
            elite_k=5,
            mutation_rate=0.2,
            rng=rng,
        )
        new_genome = res.best_genome
        new_fitness = res.best_fitness

        if new_fitness >= (old_fitness + self.min_improve):
            # backup old
            os.makedirs(os.path.dirname(self.previous_path), exist_ok=True)
            self.store.save_as(old_genome, self.previous_path, meta={"fitness": old_fitness})

            # save new
            self.store.save(new_genome, meta={"fitness": new_fitness})

            # mark upgraded for rollback logic
            self.guard.mark_upgraded(trade_count, old_fitness, new_fitness)

            # reload live engine
            if hasattr(self.decision_engine, "reload_strategy"):
                self.decision_engine.reload_strategy()

            return UpgradeReport(True, True, "upgraded", old_fitness, new_fitness, old_genome, new_genome)

        return UpgradeReport(True, False, "not_improved_enough", old_fitness, new_fitness, old_genome, new_genome)

    def maybe_rollback(self, observed_fitness: float) -> RollbackReport:
        trade_count, _samples = self._get_trade_count_and_samples()

        rd = self.guard.should_rollback(current_trade_count=trade_count, observed_fitness=observed_fitness)
        if not rd.rollback:
            return RollbackReport(True, False, rd.reason)

        prev = self.store.load_from(self.previous_path)
        if not prev:
            return RollbackReport(True, False, "no_previous_strategy")

        self.store.save(prev, meta={"rollback": True})

        if hasattr(self.decision_engine, "reload_strategy"):
            self.decision_engine.reload_strategy()

        return RollbackReport(True, True, "rolled_back")

    def _get_trade_count_and_samples(self) -> tuple[int, int]:
        mem = getattr(self.trade_memory, "memory", None)
        if isinstance(mem, list):
            return len(mem), len(mem)
        if isinstance(mem, dict):
            total_samples = 0
            for _k, entry in mem.items():
                total_samples += int(entry.get("samples", 0))
            return total_samples, total_samples
        return 0, 0

    def _fitness_from_stats(self, stats: Dict[str, Any]) -> float:
        avg_pnl = float(stats.get("avg_pnl", 0.0))
        win_rate = float(stats.get("win_rate", 0.0))
        drawdown = float(stats.get("drawdown", 0.0))
        samples = int(stats.get("samples", 0))

        W_PNL, W_WIN, W_DD = 1.0, 0.2, 0.5
        if samples <= 0:
            sample_penalty = 1.0
        elif samples < 10:
            sample_penalty = 0.5
        elif samples < 30:
            sample_penalty = 0.2
        else:
            sample_penalty = 0.0

        return float(avg_pnl * W_PNL + win_rate * W_WIN - drawdown * W_DD - sample_penalty)
