# brain/experts/expert_base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExpertDecision:
    """
    Normalized decision object returned by experts / gate.

    - allow: whether trade is allowed
    - action: string action ("buy"/"sell"/"hold"/...)
    - score: normalized score (higher is better)
    - expert: expert name
    - meta: extra info
    """
    expert: str = "UNKNOWN"
    score: float = 0.0
    allow: bool = False
    action: str = "hold"
    meta: Dict[str, Any] = field(default_factory=dict)


class ExpertBase:
    """
    Base class for all experts. Experts should implement decide(features, context).
    """
    name: str = "BASE"

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError


def coerce_decision(raw: Any, fallback_expert: str = "UNKNOWN") -> Optional[ExpertDecision]:
    """
    Convert common raw outputs into ExpertDecision.

    Supported:
    - ExpertDecision: passthrough
    - dict: {allow, score, action, meta, expert}
    - tuple/list: (allow, score) or (allow, score, action) or (allow, score, action, meta)
    - bool/float: treated as score (allow if score>0)
    """
    if raw is None:
        return None

    if isinstance(raw, ExpertDecision):
        if not raw.expert:
            raw.expert = fallback_expert
        return raw

    # dict output
    if isinstance(raw, dict):
        expert = str(raw.get("expert") or fallback_expert)
        try:
            score = float(raw.get("score", 0.0))
        except Exception:
            score = 0.0
        allow = bool(raw.get("allow", score > 0))
        action = str(raw.get("action") or "hold")
        meta = raw.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {"meta": meta}
        return ExpertDecision(expert=expert, score=score, allow=allow, action=action, meta=meta)

    # tuple/list output
    if isinstance(raw, (tuple, list)):
        if len(raw) == 0:
            return None
        allow = bool(raw[0]) if len(raw) >= 1 else False
        try:
            score = float(raw[1]) if len(raw) >= 2 else 0.0
        except Exception:
            score = 0.0
        action = str(raw[2]) if len(raw) >= 3 and raw[2] is not None else "hold"
        meta = raw[3] if len(raw) >= 4 else {}
        if not isinstance(meta, dict):
            meta = {"meta": meta}
        return ExpertDecision(expert=fallback_expert, score=score, allow=allow, action=action, meta=meta)

    # numeric => score
    if isinstance(raw, (int, float)):
        score = float(raw)
        return ExpertDecision(expert=fallback_expert, score=score, allow=(score > 0), action="hold", meta={})

    # bool => allow
    if isinstance(raw, bool):
        return ExpertDecision(expert=fallback_expert, score=1.0 if raw else 0.0, allow=raw, action="hold", meta={})

    # unknown
    return None
