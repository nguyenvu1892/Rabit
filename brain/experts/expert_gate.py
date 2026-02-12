from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_base import ExpertDecision, coerce_decision


class ExpertGate:
    """
    Picks the best expert decision, optionally adjusted by weights (expert, regime).
    """

    def __init__(self, registry: Any, weight_store: Optional[Any] = None, enable_fallback_allow: bool = True):
        self.registry = registry
        self.weight_store = weight_store
        self.enable_fallback_allow = enable_fallback_allow

    def _iter_experts(self) -> List[Any]:
        # Compatibility: registry may expose get_all / all / list / experts dict
        if self.registry is None:
            return []
        if hasattr(self.registry, "get_all"):
            try:
                return list(self.registry.get_all())
            except Exception:
                pass
        if hasattr(self.registry, "all"):
            try:
                return list(self.registry.all())
            except Exception:
                pass
        if hasattr(self.registry, "experts"):
            try:
                # dict or list
                exps = self.registry.experts
                return list(exps.values()) if isinstance(exps, dict) else list(exps)
            except Exception:
                pass
        return []

    def pick(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[ExpertDecision], List[ExpertDecision]]:
        context = context or {}
        regime = context.get("regime") or context.get("regime_id") or "unknown"

        decisions: List[ExpertDecision] = []

        experts = self._iter_experts()
        if not experts:
            # no experts -> deny safely
            return None, []

        for exp in experts:
            name = getattr(exp, "name", None) or getattr(exp, "__name__", None) or exp.__class__.__name__
            name = str(name)

            try:
                raw = exp.decide(features, context)
            except Exception as e:
                decisions.append(
                    ExpertDecision(
                        expert=name,
                        score=0.0,
                        allow=False,
                        action="hold",
                        meta={"error": repr(e)},
                    )
                )
                continue

            dec = coerce_decision(raw, fallback_expert=name)
            if dec is None:
                continue

            # weight adjust (expert, regime)
            w = 1.0
            if self.weight_store is not None and hasattr(self.weight_store, "get"):
                try:
                    w = float(self.weight_store.get(name, regime))
                except Exception:
                    w = 1.0

            raw_score = float(dec.score or 0.0)
            dec.score = raw_score * w
            dec.meta = dec.meta or {}
            dec.meta.update({"raw_score": raw_score, "w": w, "regime": regime})
            dec.expert = name  # normalize

            decisions.append(dec)

        if not decisions:
            return None, []

        # If there is at least one allow=True, pick best among allow=True
        allow_pool = [d for d in decisions if getattr(d, "allow", False)]
        if allow_pool:
            best = max(allow_pool, key=lambda d: float(d.score or 0.0))
            return best, decisions

        # else: all denied
        if self.enable_fallback_allow:
            fb = ExpertDecision(
                expert="FALLBACK",
                score=0.0001,          # tiny positive to show it's picked
                allow=True,
                action="hold",
                meta={"reason": "all_experts_denied", "regime": regime},
            )
            decisions.append(fb)
            return fb, decisions

        # strict deny: pick best scoring deny (for debugging)
        best = max(decisions, key=lambda d: float(d.score or 0.0))
        # Nếu không có decision nào allow => thêm fallback HOLD để tránh deny=100%
        if decisions and not any(getattr(d, "allow", False) for d in decisions):
            decisions.append(
                ExpertDecision(
                    expert="FALLBACK",
                    score=0.0,
                    allow=True,
                    action="hold",
                    meta={"reason": "all_experts_denied"},
                )
            )
            best_dec = decisions[-1]

        # Nếu decisions rỗng hoàn toàn => trả fallback luôn
        if not decisions:
            # emergency fallback decision
            fallback = ExpertDecision(
                expert="BASELINE_FALLBACK",
                score=0.0001,
                meta={"reason": "no_expert_decision"}
            )
            return fallback, [fallback]

        return best, decisions
