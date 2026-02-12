# brain/experts/expert_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .expert_base import ExpertDecision, coerce_decision, safe_float


@dataclass
class GateTrace:
    expert: str
    allow: bool
    action: str
    score: float
    raw_score: float
    w: float
    regime: str
    bucket: Optional[str] = None


def _decision_to_trace(d: ExpertDecision) -> GateTrace:
    meta = dict(d.meta or {})
    return GateTrace(
        expert=str(d.expert),
        allow=bool(d.allow),
        action=str(d.action or "hold"),
        score=safe_float(d.score, 0.0),
        raw_score=safe_float(meta.get("raw_score", meta.get("raw", d.score)), 0.0),
        w=safe_float(meta.get("w", 1.0), 1.0),
        regime=str(meta.get("regime") or "unknown"),
        bucket=meta.get("bucket"),
    )


class ExpertGate:
    """
    Picks the best expert decision, optionally adjusted by weights(expert, regime).

    Key goals:
    - never break older registries (get_all / all / experts dict/list)
    - never return empty in a way that blocks whole engine (fallback allow)
    """

    def __init__(
        self,
        registry: Any,
        weight_store: Optional[Any] = None,
        enable_fallback_allow: bool = True,
        treat_positive_score_as_allow: bool = True,
    ) -> None:
        self.registry = registry
        self.weight_store = weight_store
        self.enable_fallback_allow = bool(enable_fallback_allow)
        self.treat_positive_score_as_allow = bool(treat_positive_score_as_allow)

    def _iter_experts(self) -> List[Any]:
        r = self.registry
        if r is None:
            return []
        # get_all
        if hasattr(r, "get_all"):
            try:
                return list(r.get_all())
            except Exception:
                pass
        # all()
        if hasattr(r, "all"):
            try:
                return list(r.all())
            except Exception:
                pass
        # experts attr (dict or list)
        if hasattr(r, "experts"):
            try:
                exps = r.experts
                return list(exps.values()) if isinstance(exps, dict) else list(exps)
            except Exception:
                pass
        return []

    def _get_weight(self, expert_name: str, regime: str) -> float:
        w = 1.0
        try:
            if self.weight_store is not None and hasattr(self.weight_store, "get"):
                w = float(self.weight_store.get(expert_name, regime))
        except Exception:
            w = 1.0

        if w != w or w <= 0:
            w = 1.0
        return float(w)

    def pick(
        self,
        features: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ExpertDecision], List[ExpertDecision]]:
        context = context or {}
        regime = str(context.get("regime") or "unknown")

        decisions: List[ExpertDecision] = []
        experts = self._iter_experts()

        for exp in experts:
            name = str(getattr(exp, "name", None) or exp.__class__.__name__)
            try:
                raw = exp.decide(features, context)
                dec = coerce_decision(raw, fallback_expert=name)
            except Exception as e:
                dec = ExpertDecision(
                    expert=name,
                    score=0.0,
                    allow=False,
                    action="hold",
                    meta={"error": repr(e), "regime": regime},
                )

            if dec is None:
                continue

            # weight adjust
            w = self._get_weight(name, regime)
            raw_score = safe_float(dec.score, 0.0)
            adj_score = raw_score * w

            allow = bool(dec.allow)
            if (not allow) and self.treat_positive_score_as_allow and raw_score > 0:
                allow = True

            meta = dict(dec.meta or {})
            meta.update({"raw_score": raw_score, "w": w, "regime": regime})

            decisions.append(
                ExpertDecision(
                    expert=name,
                    score=float(adj_score),
                    allow=bool(allow),
                    action=str(dec.action or "hold"),
                    meta=meta,
                )
            )

        # hard fallback: no decisions at all
        if not decisions:
            fb = ExpertDecision(
                expert="FALLBACK",
                score=0.01,
                allow=True,
                action="hold",
                meta={"reason": "no_expert_decisions", "regime": regime},
            )
            return fb, [fb]

        # sort
        decisions.sort(key=lambda d: safe_float(d.score, 0.0), reverse=True)

        # if everyone denied -> inject fallback allow (fail-open)
        if self.enable_fallback_allow and (not any(bool(d.allow) for d in decisions)):
            fb = ExpertDecision(
                expert="FALLBACK",
                score=0.01,
                allow=True,
                action="hold",
                meta={"reason": "all_experts_denied", "regime": regime},
            )
            decisions.append(fb)
            decisions.sort(key=lambda d: safe_float(d.score, 0.0), reverse=True)

        best = decisions[0] if decisions else None
        return best, decisions
