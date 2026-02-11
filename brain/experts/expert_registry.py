# brain/experts/expert_gate.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_base import ExpertDecision
from brain.experts.expert_registry import ExpertRegistry


class ExpertGate:
    """
    5.0.8.9: Expert–Regime pair intelligence
    - score is adjusted by weight_store.get(expert, regime, default)
    - exploration: can force "near-miss" decision with cooldown
    - IMPORTANT: if no expert returns allow=True, we can still allow best
      when best_adj >= soft_threshold (prevents deny=2000 deadlock)
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
        rng: Optional[random.Random] = None,
        weight_store: Optional[Any] = None,
        soft_threshold: float = 1.0001,
    ) -> None:
        self.registry = registry
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(epsilon_cooldown)
        self._cooldown_left = 0
        self.rng = rng or random.Random()
        self.soft_threshold = float(soft_threshold)
        self.weight_store = weight_store

    def tick(self) -> None:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(eps)

    def set_weight_store(self, ws: Any) -> None:
        self.weight_store = ws

    def _should_explore(self) -> bool:
        if self.epsilon <= 0:
            return False
        if self._cooldown_left > 0:
            return False
        return self.rng.random() < self.epsilon

    def pick(
        self,
        trade_features: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[ExpertDecision, List[ExpertDecision]]:
        # Pull experts robustly
        try:
            experts = list(self.registry.get_all())
        except Exception:
            experts = []

        if not experts:
            return (
                ExpertDecision(
                    expert="NO_EXPERTS",
                    score=0.0,
                    allow=False,
                    meta={"reason": "empty_registry"},
                ),
                [],
            )

        decisions: List[ExpertDecision] = []
        for exp in experts:
            try:
                d = exp.decide(trade_features, context)
                if d is None:
                    continue

                # Ensure expert field populated
                if getattr(d, "expert", None) in (None, "", "UNKNOWN_EXPERT"):
                    try:
                        d.expert = getattr(exp, "name", None) or exp.__class__.__name__
                    except Exception:
                        pass

                decisions.append(d)
            except Exception as e:
                decisions.append(
                    ExpertDecision(
                        expert=getattr(exp, "name", None) or exp.__class__.__name__,
                        allow=False,
                        score=0.0,
                        meta={"error": "decide_exception", "exc": repr(e)},
                    )
                )

        if not decisions:
            dummy = ExpertDecision(
                expert="NO_DECISIONS", allow=False, score=0.0, meta={"reason": "all_none"}
            )
            return dummy, []

        regime = str(context.get("regime", "UNKNOWN"))

        def _get_weight(expert_name: str) -> float:
            if self.weight_store is None:
                return 1.0
            try:
                # expected signature: get(expert, regime, default)
                return float(self.weight_store.get(expert_name, regime, 1.0))
            except Exception:
                return 1.0

        def _adj_score(d: ExpertDecision) -> float:
            base = float(getattr(d, "score", 0.0))
            w = _get_weight(str(getattr(d, "expert", "")))
            return base * w

        allow_decisions = [d for d in decisions if bool(getattr(d, "allow", False))]

        # Best selection
        best = max(allow_decisions, key=_adj_score) if allow_decisions else max(decisions, key=_adj_score)
        best_adj = _adj_score(best)

        # Attach meta debug
        try:
            if getattr(best, "meta", None) is None:
                best.meta = {}
            best.meta["weight"] = _get_weight(str(getattr(best, "expert", "")))
            best.meta["score_adj"] = best_adj
            best.meta["regime"] = regime
        except Exception:
            pass

        # ✅ Critical anti-deadlock:
        # If nothing is allow=True, we still allow the best if it clears threshold.
        if not allow_decisions and best_adj >= self.soft_threshold:
            try:
                best.allow = True
                if getattr(best, "meta", None) is None:
                    best.meta = {}
                best.meta["forced"] = True
                best.meta["forced_reason"] = "no_allow_decisions_but_adj>=threshold"
            except Exception:
                pass
            return best, decisions

        # Exploration: if best is near-miss and we explore, allow a runner-up
        if best_adj < self.soft_threshold and self._should_explore():
            sorted_by_adj = sorted(decisions, key=_adj_score, reverse=True)
            pick2 = None
            for d in sorted_by_adj:
                if d is best:
                    continue
                pick2 = d
                break

            if pick2 is not None:
                try:
                    pick2.allow = True
                    if getattr(pick2, "meta", None) is None:
                        pick2.meta = {}
                    pick2.meta["forced"] = True
                    pick2.meta["forced_reason"] = "epsilon_explore_near_miss"
                    pick2.meta["score_adj"] = _adj_score(pick2)
                except Exception:
                    pass
                self._cooldown_left = int(self.epsilon_cooldown)
                return pick2, decisions

        return best, decisions
