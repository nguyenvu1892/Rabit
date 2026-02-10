# brain/strategy_evaluator.py
from __future__ import annotations

from typing import Dict, Any, Optional


class StrategyEvaluator:
    """
    Evaluates a strategy genome using existing TradeMemory aggregated records.

    TradeMemory expected format (current project):
      trade_memory.memory is a dict keyed by snapshot-key
      each entry has:
        - wins (int)
        - losses (int)
        - total_pnl (float)
        - samples (int)
    """

    def __init__(self, trade_memory):
        self.trade_memory = trade_memory

    def evaluate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return stats dict required by compute_fitness:
          - avg_pnl
          - win_rate
          - drawdown (proxy v1)
          - samples
        Genome is not used deeply in v1. (plug later in v2)
        """
        mem = getattr(self.trade_memory, "memory", None)

        # Handle empty / missing memory safely
        if not mem:
            return {"avg_pnl": 0.0, "win_rate": 0.0, "drawdown": 0.0, "samples": 0}

        # If memory is dict of aggregated entries
        if isinstance(mem, dict):
            return self._eval_from_aggregated_dict(mem)

        # If in future we switch to timeline list
        if isinstance(mem, list):
            return self._eval_from_timeline_list(mem)

        # Unknown type
        return {"avg_pnl": 0.0, "win_rate": 0.0, "drawdown": 0.0, "samples": 0}

    def _eval_from_aggregated_dict(self, mem: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = 0
        total_pnl = 0.0
        total_wins = 0
        total_losses = 0

        for _k, entry in mem.items():
            s = int(entry.get("samples", 0))
            if s <= 0:
                continue
            total_samples += s
            total_pnl += float(entry.get("total_pnl", 0.0))
            total_wins += int(entry.get("wins", 0))
            total_losses += int(entry.get("losses", 0))

        if total_samples <= 0:
            return {"avg_pnl": 0.0, "win_rate": 0.0, "drawdown": 0.0, "samples": 0}

        avg_pnl = total_pnl / total_samples
        win_rate = total_wins / max(1, (total_wins + total_losses))

        # drawdown proxy v1: loss_rate as penalty signal (0..1)
        loss_rate = total_losses / max(1, (total_wins + total_losses))
        drawdown_proxy = float(loss_rate)

        return {
            "avg_pnl": float(avg_pnl),
            "win_rate": float(win_rate),
            "drawdown": float(drawdown_proxy),
            "samples": int(total_samples),
        }

    def _eval_from_timeline_list(self, mem: list) -> Dict[str, Any]:
        # Future-proof: if memory becomes chronological list of {"pnl": ...}
        pnls = []
        wins = 0
        losses = 0

        for rec in mem:
            pnl = float(rec.get("pnl", 0.0))
            pnls.append(pnl)
            if pnl >= 0:
                wins += 1
            else:
                losses += 1

        samples = len(pnls)
        if samples == 0:
            return {"avg_pnl": 0.0, "win_rate": 0.0, "drawdown": 0.0, "samples": 0}

        avg_pnl = sum(pnls) / samples
        win_rate = wins / max(1, (wins + losses))

        # drawdown proxy v1: count consecutive losses max-streak normalized
        max_streak = 0
        streak = 0
        for pnl in pnls:
            if pnl < 0:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
            else:
                streak = 0
        drawdown_proxy = max_streak / max(1, samples)

        return {
            "avg_pnl": float(avg_pnl),
            "win_rate": float(win_rate),
            "drawdown": float(drawdown_proxy),
            "samples": int(samples),
        }
