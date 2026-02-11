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

    - epsilon exploration (safe)
    - cooldown for exploration
    - optional weight_store (WeightStore-like)

    IMPORTANT (5.0.8.5):
    - Do NOT multiply raw weight directly (can explode).
    - Use a safe multiplier: w ** power (sqrt by default).
    """
    registry: ExpertRegistry
    epsilon: float = 0.0
    epsilon_cooldown: int = 0
    rng: Optional[random.Random] = None
    weight_store: Optional[Any] = None
    soft_threshold: float = 1.0001  # if best adjusted score < this => allow soft explore occasionally

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
        return self._should_explore()

    def _safe_multiplier(self, expert: str, regime: Optional[str]) -> float:
        """
        Prefer WeightStore.multiplier(...) if available,
        fallback to WeightStore.get(...) then sqrt ourselves.
        """
        if self.weight_store is None:
            return 1.0

        m_exp = 1.0
        m_reg = 1.0

        # Expert
        try:
            if hasattr(self.weight_store, "multiplier"):
                m_exp = float(self.weight_store.multiplier("expert", expert, 1.0))
            else:
                w = float(self.weight_store.get("expert", expert, 1.0))
                m_exp = max(0.2, min(10.0, w)) ** 0.5
        except TypeError:
            # older signature: get(expert)
            try:
                w = float(self.weight_store.get(expert))  # type: ignore[misc]
                m_exp = max(0.2, min(10.0, w)) ** 0.5
            except Exception:
                m_exp = 1.0
        except Exception:
            m_exp = 1.0

        # Regime (weaker impact)
        if regime:
            try:
                if hasattr(self.weight_store, "multiplier"):
                    m_reg = float(self.weight_store.multiplier("regime", str(regime), 1.0, power=0.35))
                else:
                    w = float(self.weight_store.get("regime", str(regime), 1.0))
                    m_reg = max(0.2, min(10.0, w)) ** 0.35
            except Exception:
                m_reg = 1.0

        return float(m_exp * m_reg)

    def _apply_weight(self, decision: ExpertDecision, regime: Optional[str]) -> float:
        base = float(getattr(decision, "score", 0.0))
        expert = str(getattr(decision, "expert", "UNKNOWN_EXPERT"))
        return float(base) * self._safe_multiplier(expert, regime)

    def _force(self, candidate: ExpertDecision, regime: Optional[str], reason: str) -> ExpertDecision:
        forced_score = float(min(self._apply_weight(candidate, regime), 0.55))
        meta = dict(getattr(candidate, "meta", {}) or {})
        meta.update(
            {
                "forced": True,
                "forced_reason": reason,
                "epsilon": float(self.epsilon),
                "cooldown": int(self.epsilon_cooldown),
            }
        )
        self._cooldown_left = int(self.epsilon_cooldown)
        return ExpertDecision(
            True,
            forced_score,
            str(getattr(candidate, "expert", "UNKNOWN_EXPERT")),
            meta,
        )

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

        regime: Optional[str] = None
        try:
            if isinstance(context, dict) and context.get("regime") is not None:
                regime = str(context.get("regime"))
        except Exception:
            regime = None

        allow_decisions = [d for d in decisions if bool(getattr(d, "allow", False))]

        # --- Normal pick ---
        if allow_decisions:
            best = max(allow_decisions, key=lambda d: self._apply_weight(d, regime))
            best_adj = self._apply_weight(best, regime)

            # SOFT explore if best is weak
            if best_adj < self.soft_threshold and self._should_explore():
                candidate = max(decisions, key=lambda d: self._apply_weight(d, regime))
                return self._force(candidate, regime, reason="soft_exploration"), decisions

            return best, decisions

        # --- HARD explore if all deny ---
        if self._should_explore():
            candidate = max(decisions, key=lambda d: self._apply_weight(d, regime))
            return self._force(candidate, regime, reason="safe_exploration_kickstart"), decisions

        best = max(decisions, key=lambda d: self._apply_weight(d, regime))
        return best, decisions
