# brain/experts/expert_base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExpertDecision:
    """
    Unified decision object used across engine/gate.

    Backward-compat:
    - some older code may return dicts or objects with similar fields
    - coerce_decision() will normalize.
    """
    expert: str
    score: float = 0.0
    allow: bool = False
    action: str = "hold"  # 'buy'/'sell'/'hold'
    meta: Dict[str, Any] = field(default_factory=dict)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def coerce_decision(raw: Any, fallback_expert: str = "UNKNOWN") -> Optional[ExpertDecision]:
    """
    Normalize various decision formats into ExpertDecision.

    Supports:
    - ExpertDecision
    - dict-like: {expert, score, allow, action, meta}
    - object-like with attributes.
    """
    if raw is None:
        return None

    if isinstance(raw, ExpertDecision):
        raw.meta = dict(raw.meta or {})
        return raw

    if isinstance(raw, dict):
        expert = str(raw.get("expert") or fallback_expert)
        score = safe_float(raw.get("score"), 0.0)
        allow = bool(raw.get("allow", False))
        action = str(raw.get("action") or "hold")
        meta = dict(raw.get("meta") or {})
        return ExpertDecision(expert=expert, score=score, allow=allow, action=action, meta=meta)

    expert = str(getattr(raw, "expert", None) or fallback_expert)
    score = safe_float(getattr(raw, "score", 0.0), 0.0)
    allow = bool(getattr(raw, "allow", False))
    action = str(getattr(raw, "action", None) or "hold")
    meta = dict(getattr(raw, "meta", None) or {})

    return ExpertDecision(expert=expert, score=score, allow=allow, action=action, meta=meta)


class BaseExpert:
    """
    Base class for experts.
    - decide(features, context) -> ExpertDecision|dict|None
    """
    name: str = "BASE"

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError


# -------------------------
# Backward compatibility
# -------------------------
# Old code expects "ExpertBase"
ExpertBase = BaseExpert

__all__ = [
    "ExpertDecision",
    "safe_float",
    "coerce_decision",
    "BaseExpert",
    "ExpertBase",
]
