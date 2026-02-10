# brain/experts/expert_gate.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_base import ExpertDecision
from brain.experts.expert_registry import ExpertRegistry


@dataclass
class ExpertGate:
    """
    Picks best expert decision.

    Supports:
    - epsilon exploration (safe)
    - cooldown for exploration
    - optional weight_store to boost/penalize experts
    """
    registry: ExpertRegistry
    epsilon: float = 0.0
    epsilon_cooldown: int = 0
    rng: Optional[random.Random] = None
    weight_store: Optional[Any] = None  # expects .get(expert, regime=None) but handled safely
    soft_threshold: float = 1.0001

    def __post_init__(self) -> None:
        self._rng = self.rng or random.Random()
        self._cooldown_left = 0

    def tick(self) -> None:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(eps)

    # keep old & new naming compatible
    def _should_explore(self) -> bool:
        if self.epsilon <= 0:
            return False
        if self._cooldown_left > 0:
            return False
        return self._rng.random() < self.epsilon

    def should_explore(self) -> bool:
        # alias (prevents AttributeError from older code paths)
        return self._should_explore()

    def _apply_weight(self, decision: ExpertDecision, regime: Optional[str]) -> float:
        base = float(getattr(decision, "score", 0.0))
        if self.weight_store is None:
            return base

        expert = str(getattr(decision, "expert", "UNKNOWN_EXPERT"))
        try:
            # WeightStore in our project now supports optional regime
            w = float(self.weight_store.get(expert, regime))
        except TypeError:
            # backward compatibility: old WeightStore.get(expert)
            w = float(self.weight_store.get(expert))
        except Exception:
            w = 1.0

        return base * w

    def pick(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> Tuple[ExpertDecision, List[ExpertDecision]]:
        experts = self.registry.all()
        if not experts:
            best = ExpertDecision(False, 0.0, "NO_EXPERT", {"reason": "registry_empty"})
            return best, [best]

        decisions: List[ExpertDecision] = []
        for ex in experts:
            try:
                d = ex.evaluate(trade_features, context)
            except Exception as e:
                d = ExpertDecision(False, 0.0, getattr(ex, "name", "UNKNOWN_EXPERT"), {"error": repr(e)})
            decisions.append(d)

        regime = None
        try:
            regime = str(context.get("regime")) if isinstance(context, dict) else None
        except Exception:
            regime = None

        # choose best among allow=True
        allow_decisions = [d for d in decisions if bool(getattr(d, "allow", False))]
        if allow_decisions:
            best = max(allow_decisions, key=lambda d: self._apply_weight(d, regime))

            # SOFT explore: if best score weak, allow a forced “near miss” occasionally
            best_adj = self._apply_weight(best, regime)
            if best_adj < self.soft_threshold and self._should_explore():
                candidate = max(decisions, key=lambda d: self._apply_weight(d, regime))
                forced = ExpertDecision(
                    True,
                    float(min(self._apply_weight(candidate, regime), 0.55)),
                    str(getattr(candidate, "expert", "UNKNOWN_EXPERT")),
                    {**(getattr(candidate, "meta", {}) or {}), "forced": True, "forced_reason": "soft_exploration"},
                )
                self._cooldown_left = int(self.epsilon_cooldown)
                return forced, decisions

            return best, decisions

        # HARD explore: all deny
        if self._should_explore():
            candidate = max(decisions, key=lambda d: self._apply_weight(d, regime))
            forced = ExpertDecision(
                True,
                float(min(self._apply_weight(candidate, regime), 0.55)),
                str(getattr(candidate, "expert", "UNKNOWN_EXPERT")),
                {**(getattr(candidate, "meta", {}) or {}), "forced": True, "forced_reason": "safe_exploration_kickstart"},
            )
            self._cooldown_left = int(self.epsilon_cooldown)
            return forced, decisions

        # default deny
        best = max(decisions, key=lambda d: self._apply_weight(d, regime))
        return best, decisions
