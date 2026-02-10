from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_base import ExpertDecision
from brain.experts.expert_registry import ExpertRegistry


class ExpertGate:
    """
    5.0.7.x -> 5.0.8.1 bridge:
    - Pick best among allow=True using (score * weight).
    - Safe exploration (force) when:
        * all deny, or
        * allow exists but "weak confidence" (best_adj < soft_threshold)
    - Cooldown avoids spamming forced trades.
    - Backward compatible: constructor still works with (registry, epsilon).
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
        rng: Optional[random.Random] = None,
        weight_store: Any = None,
        soft_threshold: float = 1.0001,
        forced_score_cap: float = 0.55,
    ) -> None:
        self.registry = registry
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(epsilon_cooldown)
        self._cooldown_left = 0
        self._rng = rng or random.Random()
        self.weight_store = weight_store
        self.soft_threshold = float(soft_threshold)
        self.forced_score_cap = float(forced_score_cap)

    def tick(self) -> None:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

    # Backward compat: old code might call set_epsilon
    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(eps)

    # IMPORTANT: expose public name (fix AttributeError)
    def should_explore(self) -> bool:
        return self._should_explore()

    def _should_explore(self) -> bool:
        if self.epsilon <= 0:
            return False
        if self._cooldown_left > 0:
            return False
        return (self._rng.random() < self.epsilon)

    def _get_weight(self, expert: str, regime: str) -> float:
        if self.weight_store is None:
            return 1.0
        try:
            # supports get(expert, regime) or get(expert) fallback
            try:
                return float(self.weight_store.get(expert, regime))
            except TypeError:
                return float(self.weight_store.get(expert))
        except Exception:
            return 1.0

    def _force(self, candidate: ExpertDecision, reason: str, regime: str, weight: float) -> ExpertDecision:
        forced_score = float(getattr(candidate, "score", 0.0))
        forced_score = max(0.0, min(forced_score, self.forced_score_cap))

        meta = dict(getattr(candidate, "meta", {}) or {})
        meta.update(
            {
                "forced": True,
                "forced_reason": reason,
                "forced_score_cap": self.forced_score_cap,
                "epsilon": self.epsilon,
                "cooldown": self.epsilon_cooldown,
                "regime": regime,
                "weight": weight,
                "adj_score": forced_score * weight,
            }
        )

        self._cooldown_left = self.epsilon_cooldown
        return ExpertDecision(
            allow=True,
            score=forced_score,
            expert=str(getattr(candidate, "expert", "UNKNOWN_EXPERT")),
            meta=meta,
        )

    def pick(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> Tuple[ExpertDecision, List[ExpertDecision]]:
        experts = self.registry.all()
        if not experts:
            best = ExpertDecision(False, 0.0, "NO_EXPERT", {"reason": "registry_empty"})
            return best, [best]

        regime = str((context or {}).get("regime", "UNKNOWN"))

        decisions: List[ExpertDecision] = []
        for ex in experts:
            try:
                d = ex.evaluate(trade_features, context)
            except Exception as e:
                d = ExpertDecision(False, 0.0, ex.name, {"error": repr(e)})
            decisions.append(d)

        # compute adjusted scores
        def adj(d: ExpertDecision) -> float:
            expert_name = str(getattr(d, "expert", "UNKNOWN_EXPERT"))
            w = self._get_weight(expert_name, regime)
            return float(getattr(d, "score", 0.0)) * w

        allow_decisions = [d for d in decisions if bool(getattr(d, "allow", False))]

        # 1) Normal: pick best among allow=True
        if allow_decisions:
            best = max(allow_decisions, key=adj)
            best_expert = str(getattr(best, "expert", "UNKNOWN_EXPERT"))
            w = self._get_weight(best_expert, regime)
            best_adj = float(getattr(best, "score", 0.0)) * w

            meta = dict(getattr(best, "meta", {}) or {})
            meta.update({"regime": regime, "weight": w, "adj_score": best_adj, "forced": False})
            best = ExpertDecision(bool(getattr(best, "allow", False)), float(getattr(best, "score", 0.0)), best_expert, meta)

            # 1b) Soft explore: allow exists but weak confidence
            if best_adj < self.soft_threshold and self.should_explore():
                candidate = max(decisions, key=lambda d: float(getattr(d, "score", 0.0)))
                cand_expert = str(getattr(candidate, "expert", "UNKNOWN_EXPERT"))
                cand_w = self._get_weight(cand_expert, regime)
                best = self._force(candidate, reason="soft_exploration", regime=regime, weight=cand_w)

            return best, decisions

        # 2) All deny -> maybe safe explore (forced)
        if self.should_explore():
            candidate = max(decisions, key=lambda d: float(getattr(d, "score", 0.0)))
            cand_expert = str(getattr(candidate, "expert", "UNKNOWN_EXPERT"))
            cand_w = self._get_weight(cand_expert, regime)
            forced = self._force(candidate, reason="safe_exploration_kickstart", regime=regime, weight=cand_w)
            return forced, decisions

        # 3) No explore -> return the "best" decision by adjusted score, still deny
        best = max(decisions, key=adj)
        best_expert = str(getattr(best, "expert", "UNKNOWN_EXPERT"))
        w = self._get_weight(best_expert, regime)
        meta = dict(getattr(best, "meta", {}) or {})
        meta.update({"regime": regime, "weight": w, "adj_score": float(getattr(best, "score", 0.0)) * w, "forced": False})
        best = ExpertDecision(bool(getattr(best, "allow", False)), float(getattr(best, "score", 0.0)), best_expert, meta)
        return best, decisions
