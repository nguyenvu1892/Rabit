from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_base import ExpertDecision
from brain.experts.expert_registry import ExpertRegistry
from brain.weight_store import WeightStore


class ExpertGate:
    def __init__(
        self,
        registry: ExpertRegistry,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
        rng: Optional[random.Random] = None,
        weight_store: Optional[WeightStore] = None,
        soft_threshold: float = 1.0,
    ) -> None:
        self.registry = registry
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(epsilon_cooldown)
        self._cooldown_left = 0
        self._rng = rng or random.Random()
        self.weight_store = weight_store or WeightStore()
        self.soft_threshold = float(soft_threshold)

    def tick(self) -> None:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(eps)

    def _should_explore(self) -> bool:
        if self.epsilon <= 0:
            return False
        if self._cooldown_left > 0:
            return False
        return self._rng.random() < self.epsilon
        
    def should_explore(self) -> bool:
    # backward-compatible alias
        return self._should_explore()

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
                d = ExpertDecision(False, 0.0, ex.name, {"error": repr(e)})
            decisions.append(d)

        regime = str((context or {}).get("regime") or "UNKNOWN")

        # enrich meta with weights + adjusted score
        for d in decisions:
            base_score = float(getattr(d, "score", 0.0))
            expert_name = str(getattr(d, "expert", "UNKNOWN_EXPERT"))
            w = float(self.weight_store.get(expert_name))
            adj = base_score * w

            meta = dict(getattr(d, "meta", {}) or {})
            meta.update({
                "regime": regime,
                "weight": w,
                "base_score": base_score,
                "adj_score": adj,
            })
            d.meta = meta  # keep object shape as-is

        allow_decisions = [d for d in decisions if bool(getattr(d, "allow", False))]

        # NORMAL: choose best allow by adj_score
        best = None
        if allow_decisions:
            best = max(allow_decisions, key=lambda d: float((getattr(d, "meta", {}) or {}).get("adj_score", getattr(d, "score", 0.0))))
            # SOFT explore if best is weak
            best_adj = float((getattr(best, "meta", {}) or {}).get("adj_score", getattr(best, "score", 0.0)))
            if best_adj < self.soft_threshold and self.should_explore():
                candidate = max(decisions, key=lambda d: float((getattr(d, "meta", {}) or {}).get("adj_score", getattr(d, "score", 0.0))))
                best = self._force(candidate, reason="soft_exploration")
        else:
            # HARD explore if all deny
            if self.should_explore():
                candidate = max(decisions, key=lambda d: float((getattr(d, "meta", {}) or {}).get("adj_score", getattr(d, "score", 0.0))))
                best = self._force(candidate, reason="safe_exploration_kickstart")
            else:
                best = max(decisions, key=lambda d: float((getattr(d, "meta", {}) or {}).get("adj_score", getattr(d, "score", 0.0))))

        return best, decisions

    def _force(self, candidate: ExpertDecision, reason: str) -> ExpertDecision:
        meta = dict(getattr(candidate, "meta", {}) or {})
        meta.update({
            "forced": True,
            "forced_reason": reason,
            "epsilon": self.epsilon,
            "cooldown": self.epsilon_cooldown,
        })
        self._cooldown_left = self.epsilon_cooldown

        forced_score = float(meta.get("adj_score", getattr(candidate, "score", 0.0)))
        forced_score = max(0.0, min(forced_score, 0.55))  # safety cap
        meta["forced_score_cap"] = 0.55

        return ExpertDecision(
            allow=True,
            score=forced_score,
            expert=str(getattr(candidate, "expert", "UNKNOWN_EXPERT")),
            meta=meta,
        )
