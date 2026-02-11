# brain/experts/expert_gate.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_base import ExpertDecision
from brain.experts.expert_registry import ExpertRegistry


class ExpertGate:
    """
    5.0.8.9: Expert–Regime pair intelligence
      - score is adjusted by weight_store.get(expert, regime)
      - exploration: can force "near-miss" decision with cooldown
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
        rng: Optional[random.Random] = None,
        weight_store: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(epsilon_cooldown)
        self._cooldown_left = 0
        self.rng = rng or random.Random()
        self.soft_threshold = 1.0001
        self.weight_store = weight_store

    def tick(self) -> None:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(eps)

    def set_weight_store(self, ws: Any) -> None:
        self.weight_store = ws

    def should_explore(self) -> bool:
        return self._should_explore()

    def _should_explore(self) -> bool:
        if self.epsilon <= 0:
            return False
        if self._cooldown_left > 0:
            return False
        return self.rng.random() < self.epsilon

    def pick(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> Tuple[ExpertDecision, List[ExpertDecision]]:
        experts = self._iter_experts()

        if not experts:
            return self._fallback_decision_no_expert(trade_features, context)
        decisions: List[ExpertDecision] = []
        for exp in experts:
            try:
                d = exp.decide(trade_features, context)
                if d is None:
                    continue
                if getattr(d, "expert", None) in (None, "", "UNKNOWN_EXPERT"):
                    try:
                        d.expert = getattr(exp, "name", None) or exp.__class__.__name__
                    except Exception:
                        pass
                decisions.append(d)
            except Exception as e:
                # don't silently swallow; keep a trace decision for debugging
                decisions.append(
                    ExpertDecision(
                        expert=getattr(exp, "name", None) or exp.__class__.__name__,
                        allow=False,
                        score=0.0,
                        meta={"error": "decide_exception", "exc": repr(e)},
                    )
                )

        # If all failed/none, return a deterministic deny with meta
        if not decisions:
            dummy = ExpertDecision(expert="NO_DECISIONS", allow=False, score=0.0, meta={"reason": "all_none"})
            return dummy, []

        regime = str(context.get("regime", "UNKNOWN"))

        def _adj_score(d: ExpertDecision) -> float:
            base = float(getattr(d, "score", 0.0))
            w = 1.0
            if self.weight_store is not None:
                try:
                    w = float(self.weight_store.get(str(getattr(d, "expert", "")), regime, 1.0))
                except Exception:
                    w = 1.0
            return base * w

        allow_decisions = [d for d in decisions if bool(getattr(d, "allow", False))]
        best = max(allow_decisions, key=_adj_score) if allow_decisions else max(decisions, key=_adj_score)

        best_adj = _adj_score(best)

        # attach debug meta
        try:
            if getattr(best, "meta", None) is None:
                best.meta = {}
            w_best = 1.0
            if self.weight_store is not None:
                try:
                    w_best = float(self.weight_store.get(str(getattr(best, "expert", "")), regime, 1.0))
                except Exception:
                    w_best = 1.0
            best.meta["weight"] = w_best
            best.meta["score_adj"] = best_adj
        except Exception:
            pass

        # exploration: if best is near-miss and we explore, allow it forcibly
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
                except Exception:
                    pass

                self._cooldown_left = int(self.epsilon_cooldown)
                return pick2, decisions

        def _iter_experts(self):
            if self.registry is None:
                return []
            for name in ("get_all", "all", "list_all"):
                fn = getattr(self.registry, name, None)
                if callable(fn):
                    try:
                        xs = fn()
                        return list(xs) if xs is not None else []
                    except Exception:
                        pass

            for attr in ("experts", "_experts"):
                d = getattr(self.registry, attr, None)
                if isinstance(d, dict):
                    return list(d.values())
                if isinstance(d, list):
                    return d

            return []

        return best, decisions
