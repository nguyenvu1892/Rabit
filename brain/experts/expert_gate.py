# brain/experts/expert_gate.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .expert_base import ExpertDecision, coerce_decision

if TYPE_CHECKING:
    from .expert_registry import ExpertRegistry


class ExpertGate:
    """
    Chooses an expert decision from ExpertRegistry.
    Supports exploration (epsilon) and weight adjustment via WeightStore.

    WeightStore compatibility:
      - get(expert_name) -> float
      - OR get(expert_name, regime) -> float
    """

    def __init__(
        self,
        registry: "ExpertRegistry",
        weight_store: Any = None,
        epsilon: float = 0.05,
        epsilon_cooldown: int = 0,
        soft_threshold: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.registry = registry
        self.weight_store = weight_store
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(epsilon_cooldown)
        self.soft_threshold = float(soft_threshold)

        self._rng = random.Random(seed)
        self._cooldown_left = 0

    def set_epsilon(self, epsilon: float, cooldown: Optional[int] = None) -> None:
        self.epsilon = float(epsilon)
        if cooldown is not None:
            self.epsilon_cooldown = int(cooldown)

    def _should_explore(self) -> bool:
        if self.epsilon <= 0:
            return False
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return False
        if self._rng.random() < self.epsilon:
            self._cooldown_left = max(0, int(self.epsilon_cooldown))
            return True
        return False

    def _get_weight(self, expert_name: str, regime: Optional[str]) -> float:
        ws = self.weight_store
        if ws is None:
            return 1.0

        # try pair weight first (expert, regime)
        if regime is not None:
            try:
                w = ws.get(expert_name, regime)
                return float(w)
            except TypeError:
                pass
            except Exception:
                pass

        # fallback single-key weight (expert)
        try:
            w = ws.get(expert_name)
            return float(w)
        except Exception:
            return 1.0

    def pick(
        self,
        trade_features: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[ExpertDecision, List[ExpertDecision]]:
        """
        Returns:
          - best_decision: ExpertDecision
          - all_decisions: List[ExpertDecision]  (raw per expert)
        """
        experts = self.registry.get_all()
        if not experts:
            best = ExpertDecision(
                allow=False,
                score=0.0,
                expert="NO_EXPERT",
                meta={"reason": "registry empty"},
            )
            return best, [best]

        regime = None
        try:
            regime = context.get("regime")
        except Exception:
            regime = None

        all_decisions: List[ExpertDecision] = []
        scored: List[Tuple[float, ExpertDecision]] = []

        for exp in experts:
            name = getattr(exp, "name", exp.__class__.__name__)
            try:
                raw = exp.decide(trade_features, context)
                dec = coerce_decision(raw, fallback_expert=str(name))
            except Exception as e:
                dec = ExpertDecision(
                    allow=False,
                    score=0.0,
                    expert=str(name),
                    meta={"error": repr(e)},
                )

            all_decisions.append(dec)

            w = self._get_weight(dec.expert, regime)
            adj = float(dec.score) * float(w)
            # keep weight inside meta for debugging
            dec.meta = dict(dec.meta or {})
            dec.meta.setdefault("weight", float(w))
            dec.meta.setdefault("adj_score", float(adj))

            scored.append((adj, dec))

        # Exploration: pick random among candidates (prefer allow if any)
        if self._should_explore():
            allow_candidates = [d for d in all_decisions if d.allow]
            chosen = self._rng.choice(allow_candidates or all_decisions)
            return chosen, all_decisions

        # Exploitation: choose best adjusted score (prefer allow if soft_threshold requires)
        scored.sort(key=lambda x: x[0], reverse=True)
        best_adj, best_dec = scored[0]

        # Optional: if best is not allow and you want "soft threshold" to force explore/deny,
        # keep it simple: only allow if it passes threshold.
        if self.soft_threshold > 0 and best_adj < self.soft_threshold:
            # below threshold -> deny
            best_dec = ExpertDecision(
                allow=False,
                score=float(best_dec.score),
                expert=str(best_dec.expert),
                meta={**(best_dec.meta or {}), "reason": "below_soft_threshold"},
            )

        return best_dec, all_decisions
