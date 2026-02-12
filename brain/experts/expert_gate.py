# brain/experts/expert_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .expert_base import ExpertDecision, coerce_decision


@dataclass
class GatePickResult:
    best: Optional[ExpertDecision]
    all_decisions: List[ExpertDecision]


class ExpertGate:
    """
    Gate picks the best expert decision.

    - It supports weights from weight_store (optional): weight_store.get(expert_name, regime)
    - It also supports regime information from context or meta_controller (optional)
    """

    def __init__(self, registry, weight_store=None, meta_controller=None) -> None:
        self.registry = registry
        self.weight_store = weight_store
        self.meta_controller = meta_controller

    def _get_regime(self, context: Optional[Dict[str, Any]]) -> str:
        ctx = context or {}
        # common keys (keep compatibility)
        r = (
            ctx.get("regime")
            or ctx.get("market_regime")
            or (ctx.get("risk_cfg") or {}).get("regime")
            or (ctx.get("meta") or {}).get("regime")
        )
        return str(r) if r is not None else "unknown"

    def _get_weight(self, expert_name: str, regime: str) -> float:
        w = 1.0
        if self.weight_store is None:
            return w
        if not hasattr(self.weight_store, "get"):
            return w
        try:
            w = float(self.weight_store.get(expert_name, regime))
        except Exception:
            w = 1.0
        if w <= 0:
            w = 0.0001
        return w

    def pick(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[ExpertDecision], List[ExpertDecision]]:
        regime = self._get_regime(context)

        experts = []
        try:
            experts = list(self.registry.get_all() or [])
        except Exception:
            experts = []

        decisions: List[ExpertDecision] = []

        for exp in experts:
            try:
                name = getattr(exp, "name", None) or getattr(exp, "__class__", type("X", (), {})).__name__
                raw = exp.decide(features, context)
                dec = coerce_decision(raw, fallback_expert=str(name))
                if dec is None:
                    continue

                # ensure expert is set
                if not dec.expert:
                    dec.expert = str(name)

                # apply weight
                w = self._get_weight(dec.expert, regime)
                raw_score = float(dec.score) if dec.score is not None else 0.0
                adj_score = raw_score * w

                # keep meta full
                meta = dict(dec.meta or {})
                meta.setdefault("raw_score", raw_score)
                meta.setdefault("w", w)
                meta.setdefault("regime", regime)

                decisions.append(
                    ExpertDecision(
                        expert=dec.expert,
                        score=adj_score,
                        allow=bool(dec.allow),
                        action=str(dec.action or "hold"),
                        meta=meta,
                    )
                )
            except Exception:
                # swallow expert failure to keep system alive
                continue

        if not decisions:
            return None, []

        # sort by score desc (stable)
        decisions.sort(key=lambda d: float(d.score or 0.0), reverse=True)
        best = decisions[0]

        # If all experts deny, add a safe fallback (compat with older "avoid deny=100%")
        if decisions and not any(bool(getattr(d, "allow", False)) for d in decisions):
            try:
                decisions.append(
                    ExpertDecision(
                        expert="FALLBACK",
                        score=0.0001,         # tiny positive score
                        allow=True,            # allow to let pipeline produce outcomes
                        action="hold",
                        meta={"reason": "all_experts_denied", "regime": regime},
                    )
                )
                decisions.sort(key=lambda d: float(d.score or 0.0), reverse=True)
                best = decisions[0]
            except Exception:
                pass

        return best, decisions
