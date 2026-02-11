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

    # --- tick/cooldown
    def tick(self) -> None:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(eps)

    def set_weight_store(self, ws: Any) -> None:
        self.weight_store = ws

    # alias to avoid old code calling should_explore()
    def should_explore(self) -> bool:
        return self._should_explore()

    def _should_explore(self) -> bool:
        if self.epsilon <= 0:
            return False
        if self._cooldown_left > 0:
            return False
        return self.rng.random() < self.epsilon

    # ------------------------
    # Core pick
    # ------------------------
    def pick(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> Tuple[ExpertDecision, List[ExpertDecision]]:
        """
        Returns (best_decision, all_decisions).
        best_decision.score is ORIGINAL score from expert.
        We add meta['score_adj'] and meta['weight'] for debugging/training.
        """
        decisions: List[ExpertDecision] = []
        for exp in self.registry.get_all():
            try:
                d = exp.decide(trade_features, context)
                if d is None:
                    continue
                # ensure expert label
                if getattr(d, "expert", None) in (None, "", "UNKNOWN_EXPERT"):
                    try:
                        d.expert = getattr(exp, "name", None) or getattr(exp, "__class__", type("X", (), {})).__name__
                    except Exception:
                        pass
                decisions.append(d)
            except Exception:
                continue

        if not decisions:
            # emergency fallback
            dummy = ExpertDecision(expert="NO_EXPERT", allow=False, score=0.0, meta={})
            return dummy, []

        regime = str(context.get("regime", "UNKNOWN"))

        def _adj_score(d: ExpertDecision) -> float:
            s = float(getattr(d, "score", 0.0))
            w = 1.0
            if self.weight_store is not None:
                try:
                    w = float(self.weight_store.get(str(getattr(d, "expert", "UNKNOWN_EXPERT")), regime, 1.0))
                except Exception:
                    w = 1.0
            return s * w

        # choose best by adjusted score
        best = max(decisions, key=_adj_score)
        best_adj = _adj_score(best)

        # attach debug meta
        try:
            if getattr(best, "meta", None) is None:
                best.meta = {}
            w_best = 1.0
            if self.weight_store is not None:
                try:
                    w_best = float(self.weight_store.get(str(getattr(best, "expert", "UNKNOWN_EXPERT")), regime, 1.0))
                except Exception:
                    w_best = 1.0
            best.meta["weight"] = w_best
            best.meta["score_adj"] = best_adj
        except Exception:
            pass

        # exploration: if best is near-miss and we explore, allow it forcibly
        if best_adj < self.soft_threshold and self._should_explore():
            # pick a near-miss candidate among allow=False (or lowest margin)
            sorted_by_adj = sorted(decisions, key=_adj_score, reverse=True)
            pick2 = None
            for d in sorted_by_adj:
                # near-miss means close to best (or just next)
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

        return best, decisions
