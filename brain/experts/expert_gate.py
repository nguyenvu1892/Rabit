from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .expert_base import ExpertDecision, coerce_decision

def _decision_to_trace(d):
    # d có thể là ExpertDecision hoặc dict-like
    meta = getattr(d, "meta", None) or {}
    return {
        "expert": getattr(d, "expert", None),
        "allow": bool(getattr(d, "allow", False)),
        "action": getattr(d, "action", "hold"),
        "score": float(getattr(d, "score", 0.0) or 0.0),           # adjusted score (sau weight)
        "raw_score": float(meta.get("raw_score", meta.get("raw", 0.0)) or 0.0),
        "w": float(meta.get("w", 1.0) or 1.0),
        "regime": meta.get("regime"),
        "bucket": meta.get("bucket"),
    }

class ExpertGate:
    """
    Picks the best expert decision, optionally adjusted by weights (expert, regime).
    """
    registry: Any
    weight_store: Any = None
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
    def _get_weight(self, expert_name: str, regime: str) -> float:
        # default weight
        w = 1.0
        try:
            if self.weight_store is not None and hasattr(self.weight_store, "get"):
                w = float(self.weight_store.get(expert_name, regime))
        except Exception:
            w = 1.0
        # safety clamp
        if not (w == w):  # NaN
            w = 1.0
        if w <= 0:
            w = 1.0
        return w

    def pick(
        self,
        features: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ExpertDecision], List[ExpertDecision]]:
        context = context or {}

        regime = str(context.get("regime") or "unknown")

        decisions: List[ExpertDecision] = []

        # --- Iterate experts safely
        try:
            experts = list(self.registry.get_all()) if hasattr(self.registry, "get_all") else []
        except Exception:
            experts = []

        for exp in experts:
            name = getattr(exp, "name", None) or exp.__class__.__name__
            try:
                raw = exp.decide(features, context)
                dec = coerce_decision(raw, fallback_expert=str(name))
            except Exception as e:
                dec = ExpertDecision(
                    expert=str(name),
                    score=0.0,
                    allow=False,
                    action="hold",
                    meta={"error": repr(e), "regime": regime},
                )

            if dec is None:
                continue

            # Weight adjust
            w = self._get_weight(str(name), regime)
            raw_score = float(getattr(dec, "score", 0.0) or 0.0)
            adj = raw_score * w

            # If expert didn't set allow, infer from score (keep backward compatibility)
            allow = bool(getattr(dec, "allow", False))
            if allow is False and raw_score > 0:
                # optional: treat positive score as allow
                allow = True

            action = str(getattr(dec, "action", "hold") or "hold")
            meta = dict(getattr(dec, "meta", {}) or {})
            meta.update(
                {
                    "raw_score": raw_score,
                    "w": w,
                    "regime": regime,
                }
            )

            decisions.append(
                ExpertDecision(
                    expert=str(name),
                    score=float(adj),
                    allow=bool(allow),
                    action=action,
                    meta=meta,
                )
            )

        # --- No decisions at all
        if not decisions:
            fb = ExpertDecision(
                expert="FALLBACK",
                score=0.01,
                allow=True,
                action="hold",
                meta={"reason": "no_expert_decisions", "regime": regime},
            )
            return fb, [fb]

        # --- Sort by score desc
        decisions.sort(key=lambda d: float(getattr(d, "score", 0.0) or 0.0), reverse=True)

        # --- If everyone denied -> inject fallback allow=True and re-pick best
        if not any(bool(getattr(d, "allow", False)) for d in decisions):
            fb = ExpertDecision(
                expert="FALLBACK",
                score=0.01,
                allow=True,
                action="hold",
                meta={"reason": "all_experts_denied", "regime": regime},
            )
            decisions.append(fb)
            decisions.sort(key=lambda d: float(getattr(d, "score", 0.0) or 0.0), reverse=True)

        best_dec = decisions[0] if decisions else None
        return best_dec, decisions
