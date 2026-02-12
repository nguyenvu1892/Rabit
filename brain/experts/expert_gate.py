# brain/experts/expert_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExpertDecision:
    expert: str
    score: float
    meta: Dict[str, Any]


class ExpertGate:
    """
    Picks the best expert based on expert score, then adjusts using WeightStore(expert, regime).
    """

    def __init__(
        self,
        registry: Any,
        weight_store: Optional[Any] = None,
        *,
        soft_threshold: float = 0.0,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
    ) -> None:
        self.registry = registry
        self.weight_store = weight_store

        self.soft_threshold = float(soft_threshold)
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(epsilon_cooldown)

        self._step = 0
        self._last_explore_step = -10**9

    # -------------
    # exploration
    # -------------
    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(epsilon)

    def set_exploration(self, epsilon: float, cooldown: int = 0) -> None:
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(cooldown)

    def _should_explore(self) -> bool:
        if self.epsilon <= 0:
            return False
        if (self._step - self._last_explore_step) < max(0, self.epsilon_cooldown):
            return False
        # deterministic-ish: use modulus instead of random to keep reproducible
        # (you already seed elsewhere; this is just a safe fallback)
        trigger = (self._step % int(max(1, round(1.0 / max(1e-9, self.epsilon))))) == 0
        if trigger:
            self._last_explore_step = self._step
            return True
        return False

    # keep compatibility for older callers
    def should_explore(self) -> bool:
        return self._should_explore()

    # -------------
    # registry iteration (compat)
    # -------------
    def _iter_experts(self) -> List[Any]:
        """
        Support multiple registry APIs:
          - get_all()
          - all()
          - list()
          - values()
          - internal dict-like
        """
        r = self.registry

        if r is None:
            return []

        for name in ("get_all", "all", "list", "values"):
            fn = getattr(r, name, None)
            if callable(fn):
                try:
                    xs = fn()
                    return list(xs) if xs is not None else []
                except Exception:
                    pass

        # dict-like fallback
        try:
            if isinstance(r, dict):
                return list(r.values())
        except Exception:
            pass

        # last resort: try attribute _experts
        try:
            xs = getattr(r, "_experts", None)
            if isinstance(xs, dict):
                return list(xs.values())
            if xs is not None:
                return list(xs)
        except Exception:
            pass

        return []

    # -------------
    # main pick
    # -------------
    def pick(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Optional[ExpertDecision], List[ExpertDecision]]:
        self._step += 1

        regime = context.get("regime", None)
        if regime is None:
            regime = "UNKNOWN"
        regime = str(regime)

        experts = self._iter_experts()
        decisions: List[ExpertDecision] = []

        for exp in experts:
            try:
                name = getattr(exp, "name", None) or getattr(exp, "id", None) or exp.__class__.__name__
                name = str(name)

                # score API: score(trade_features, context) OR evaluate(...)
                score = None
                if hasattr(exp, "score") and callable(getattr(exp, "score")):
                    score = exp.score(trade_features, context)
                elif hasattr(exp, "evaluate") and callable(getattr(exp, "evaluate")):
                    score = exp.evaluate(trade_features, context)
                else:
                    # minimal fallback
                    score = float(getattr(exp, "base_score", 0.0))

                s = float(score) if score is not None else 0.0

                w = 1.0
                if self.weight_store is not None and hasattr(self.weight_store, "get"):
                    try:
                        w = float(self.weight_store.get(name, regime))
                    except Exception:
                        w = 1.0

                adj = s * w
                decisions.append(ExpertDecision(expert=name, score=adj, meta={"raw_score": s, "w": w, "regime": regime}))
            except Exception:
                continue

        if not decisions:
            return None, []

        decisions.sort(key=lambda d: d.score, reverse=True)
        best = decisions[0]

        # exploration: pick a different expert when exploring
        if self._should_explore() and len(decisions) > 1:
            best = decisions[1]

        # soft threshold: allow deny upstream if too weak
        # (DecisionEngine decides allow/deny; here we just return best + list)
        return best, decisions
