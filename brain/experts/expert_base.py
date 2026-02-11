# brain/experts/expert_base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class ExpertDecision:
    """
    Output of an expert decision.

    - allow: should enter a trade
    - score: confidence/utility score (higher is better)
    - expert: expert name
    - meta: optional extra info (signals, thresholds, etc.)
    """
    allow: bool
    score: float
    expert: str
    meta: Dict[str, Any] = field(default_factory=dict)


class ExpertBase(Protocol):
    """
    Protocol (no inheritance needed) to avoid circular imports.
    Any expert class that has:
      - name: str
      - decide(trade_features, context) -> ExpertDecision | dict | tuple
    is compatible.
    """
    name: str

    def decide(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> Any:
        ...


def coerce_decision(obj: Any, fallback_expert: str) -> ExpertDecision:
    """
    Normalize different expert outputs into ExpertDecision.
    Supports:
      - ExpertDecision
      - dict {allow, score, meta?, expert?}
      - tuple/list (allow, score) or (allow, score, meta)
    """
    if isinstance(obj, ExpertDecision):
        if not obj.expert:
            obj.expert = fallback_expert
        return obj

    if isinstance(obj, dict):
        allow = bool(obj.get("allow", False))
        score = float(obj.get("score", 0.0))
        expert = str(obj.get("expert") or fallback_expert)
        meta = obj.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {"meta": meta}
        return ExpertDecision(allow=allow, score=score, expert=expert, meta=meta)

    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        allow = bool(obj[0])
        score = float(obj[1])
        meta: Optional[Dict[str, Any]] = None
        if len(obj) >= 3 and isinstance(obj[2], dict):
            meta = obj[2]
        return ExpertDecision(
            allow=allow,
            score=score,
            expert=fallback_expert,
            meta=meta or {},
        )

    # fallback
    return ExpertDecision(allow=False, score=0.0, expert=fallback_expert, meta={"raw": str(obj)})
