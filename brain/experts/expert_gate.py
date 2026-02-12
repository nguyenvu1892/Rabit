# brain/experts/expert_gate.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@dataclass
class ExpertDecision:
    # Keep compatible fields
    expert: str
    score: float
    allow: bool = False
    action: str = "hold"
    meta: Dict[str, Any] = field(default_factory=dict)


class ExpertGate:
    """
    ExpertGate:
    - Calls each expert.decide(features, context)
    - Normalizes decision to ExpertDecision
    - Applies weight_store multiplier (expert, regime) if available
    - Applies score_threshold gating from context/meta_controller
    """

    def __init__(self, registry: Any, weight_store: Any = None, **kwargs) -> None:
        self.registry = registry
        self.weight_store = weight_store

    def _coerce_decision(self, raw: Any, fallback_expert: str) -> Optional[ExpertDecision]:
        if raw is None:
            return None

        # If already correct type
        if isinstance(raw, ExpertDecision):
            if not raw.expert:
                raw.expert = fallback_expert
            raw.score = _safe_float(raw.score, 0.0)
            raw.allow = bool(getattr(raw, "allow", False))
            raw.action = str(getattr(raw, "action", "hold") or "hold")
            raw.meta = dict(getattr(raw, "meta", {}) or {})
            return raw

        # If dict-like
        if isinstance(raw, dict):
            expert = str(raw.get("expert") or fallback_expert)
            score = _safe_float(raw.get("score"), 0.0)
            allow = bool(raw.get("allow", False))
            action = str(raw.get("action") or "hold")
            meta = dict(raw.get("meta") or {})
            return ExpertDecision(expert=expert, score=score, allow=allow, action=action, meta=meta)

        # If tuple(score, allow, action)
        try:
            if isinstance(raw, tuple) and len(raw) >= 1:
                score = _safe_float(raw[0], 0.0)
                allow = bool(raw[1]) if len(raw) >= 2 else False
                action = str(raw[2]) if len(raw) >= 3 else "hold"
                return ExpertDecision(expert=fallback_expert, score=score, allow=allow, action=action, meta={})
        except Exception:
            pass

        return None

    def pick(self, features: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Optional[ExpertDecision], List[ExpertDecision]]:
        regime = None
        try:
            regime = (context.get("regime") if isinstance(context, dict) else None)
        except Exception:
            regime = None

        # threshold: from context meta_policy if available
        score_threshold = 0.0
        try:
            mp = context.get("meta_policy") or {}
            score_threshold = _safe_float(mp.get("score_threshold"), 0.0)
        except Exception:
            score_threshold = 0.0

        decisions: List[ExpertDecision] = []

        # get experts list with compatibility
        experts = []
        try:
            if hasattr(self.registry, "get_all"):
                experts = list(self.registry.get_all())
            elif hasattr(self.registry, "all"):
                experts = list(self.registry.all())
            elif hasattr(self.registry, "experts"):
                experts = list(self.registry.experts)
        except Exception:
            experts = []

        for exp in experts:
            name = getattr(exp, "name", None) or exp.__class__.__name__
            try:
                raw = exp.decide(features, context)
            except Exception as e:
                # record error decision (deny)
                decisions.append(
                    ExpertDecision(
                        expert=str(name),
                        score=0.0,
                        allow=False,
                        action="hold",
                        meta={"error": repr(e)},
                    )
                )
                continue

            dec = self._coerce_decision(raw, fallback_expert=str(name))
            if dec is None:
                continue

            # apply weight multiplier by expert-regime
            w = 1.0
            if self.weight_store is not None and hasattr(self.weight_store, "get"):
                try:
                    w = _safe_float(self.weight_store.get(dec.expert, str(regime or "unknown")), 1.0)
                except Exception:
                    w = 1.0
            adj_score = _safe_float(dec.score, 0.0) * w

            # store trace
            dec.meta = dict(dec.meta or {})
            dec.meta.setdefault("raw_score", _safe_float(dec.score, 0.0))
            dec.meta.setdefault("w", w)
            dec.meta.setdefault("regime", str(regime or "unknown"))
            dec.score = adj_score

            # gate allow by threshold
            dec.allow = bool(dec.allow and (dec.score >= score_threshold))

            decisions.append(dec)

        # choose best among allow; if none allow, best = max score but allow False
        if not decisions:
            return None, []

        decisions.sort(key=lambda d: _safe_float(d.score, 0.0), reverse=True)

        best_allow = next((d for d in decisions if bool(getattr(d, "allow", False))), None)
        if best_allow is not None:
            return best_allow, decisions

        # no allow => return top but will be deny by engine; keep trace
        return decisions[0], decisions
