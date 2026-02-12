# brain/experts/expert_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# NOTE:
# - ExpertDecision SHOULD be defined in expert_base.py (single source of truth)
# - We re-export it here for backward compatibility.
from brain.experts.expert_base import ExpertBase, ExpertDecision, coerce_decision


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


class ExpertGate:
    """
    ExpertGate selects a decision from experts registry.

    Contract:
      pick(features, context) -> (best_decision: Optional[ExpertDecision], all_decisions: List[ExpertDecision])

    Behavior:
      - Collect decisions from experts
      - Adjust score by weight_store(expert, regime) if available
      - Choose best by score (desc)
      - If all experts deny, inject FALLBACK allow=True (hold) to avoid deny=100%
    """

    def __init__(
        self,
        registry: Any,
        weight_store: Optional[Any] = None,
        regime_detector: Optional[Any] = None,
        debug: bool = False,
    ) -> None:
        self.registry = registry
        self.weight_store = weight_store
        self.regime_detector = regime_detector
        self.debug = bool(debug)

    def _get_regime(self, features: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        # regime can come from context first
        if context and context.get("regime") is not None:
            return str(context["regime"])

        # try regime_detector
        try:
            if self.regime_detector is not None and hasattr(self.regime_detector, "detect"):
                rr = self.regime_detector.detect(features, context or {})
                # rr may be object; stringify for stable keys
                if rr is None:
                    return "unknown"
                # if it has .regime use it
                if hasattr(rr, "regime"):
                    return str(getattr(rr, "regime"))
                return str(rr)
        except Exception:
            pass

        return "unknown"

    def pick(
        self,
        features: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ExpertDecision], List[ExpertDecision]]:
        context = context or {}
        regime = self._get_regime(features, context)

        experts: List[Any] = []
        try:
            if self.registry is not None and hasattr(self.registry, "get_all"):
                experts = list(self.registry.get_all())
        except Exception:
            experts = []

        decisions: List[ExpertDecision] = []

        for exp in experts:
            try:
                # exp can be ExpertBase or any compatible object
                name = getattr(exp, "name", None) or getattr(exp, "expert_name", None) or exp.__class__.__name__
                name = str(name)

                raw = exp.decide(features, context)  # can return ExpertDecision / dict / tuple / None
                dec = coerce_decision(raw, fallback_expert=name)
                if dec is None:
                    continue

                # ensure expert name is consistent
                dec.expert = str(getattr(dec, "expert", None) or name)

                # base score
                s = _safe_float(getattr(dec, "score", 0.0), 0.0)

                # weight by store
                w = 1.0
                if self.weight_store is not None and hasattr(self.weight_store, "get"):
                    try:
                        w = _safe_float(self.weight_store.get(dec.expert, regime), 1.0)
                    except Exception:
                        w = 1.0

                adj = s * w
                dec.score = adj

                # meta enrich
                meta = dict(getattr(dec, "meta", None) or {})
                meta.setdefault("raw_score", s)
                meta.setdefault("w", w)
                meta.setdefault("regime", regime)
                dec.meta = meta

                decisions.append(dec)

            except Exception:
                continue

        if not decisions:
            # no expert produced a decision -> do NOT force deny loop; return None for engine to decide
            return None, []

        # sort by score desc
        decisions.sort(key=lambda d: _safe_float(getattr(d, "score", 0.0), 0.0), reverse=True)
        best = decisions[0]

        # If every expert denies, inject a safe fallback allow=True
        if not any(bool(getattr(d, "allow", False)) for d in decisions):
            try:
                fb = ExpertDecision(
                    expert="FALLBACK",
                    score=0.01,          # tiny positive score
                    allow=True,
                    action="hold",
                    meta={"reason": "all_experts_denied", "regime": regime},
                )
                decisions.append(fb)
                best = fb
            except Exception:
                # if dataclass fails for any reason, keep best as-is
                pass

        if self.debug:
            try:
                print("DEBUG REGISTRY EXPERTS:", [getattr(e, "name", e.__class__.__name__) for e in experts])
                print("DEBUG BEST:", best)
            except Exception:
                pass

        return best, decisions
