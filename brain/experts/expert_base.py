from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExpertDecision:
    """
    Backward-compatible decision object.

    Legacy positional order in older code:
      ExpertDecision(allow, score, expert, meta)

    Newer code can use keywords:
      ExpertDecision(allow=True, score=0.1, expert="BASELINE", meta={...}, action="hold")
    """
    allow: bool
    score: float
    expert: str
    meta: Dict[str, Any] = field(default_factory=dict)
    action: str = "hold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow": bool(self.allow),
            "score": float(self.score),
            "expert": str(self.expert),
            "action": str(self.action),
            "meta": dict(self.meta or {}),
        }


class BaseExpert:
    """
    Base class for experts.

    Implement either:
      - decide(features, context) -> ExpertDecision
    or legacy:
      - evaluate(features, context) -> ExpertDecision
    """
    name: str = "BASE"

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Optional[ExpertDecision]:
        # Backward compatibility: if subclass still defines evaluate(), use it.
        ev = getattr(self, "evaluate", None)
        if callable(ev):
            return ev(features, context)  # type: ignore[misc]
        raise NotImplementedError("Expert must implement decide() or evaluate().")


# Backward-compatible alias (some files import ExpertBase)
ExpertBase = BaseExpert


def coerce_decision(raw: Any, fallback_expert: str = "UNKNOWN") -> Optional[ExpertDecision]:
    """
    Convert arbitrary expert output -> ExpertDecision.

    Accepts:
    - ExpertDecision
    - dict: {"score":..., "allow":..., "action":..., "meta":..., "expert":...}
    - tuple/list: (score, allow) or (score, allow, action)
    - numeric: score (allow inferred score > 0)
    - None -> None
    """
    if raw is None:
        return None

    if isinstance(raw, ExpertDecision):
        return raw

    # dict payload
    if isinstance(raw, dict):
        expert = str(raw.get("expert", fallback_expert))
        score = float(raw.get("score", 0.0) or 0.0)
        allow = bool(raw.get("allow", score > 0))
        action = str(raw.get("action", "hold") or "hold")
        meta = raw.get("meta") or {}
        # merge extra keys into meta (optional)
        extra = {k: v for k, v in raw.items() if k not in {"expert", "score", "allow", "action", "meta"}}
        if extra:
            meta = {**meta, **extra}
        return ExpertDecision(expert=expert, score=score, allow=allow, action=action, meta=meta)

    # tuple/list
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        score = float(raw[0] or 0.0)
        allow = bool(raw[1])
        action = str(raw[2]) if len(raw) >= 3 and raw[2] is not None else "hold"
        return ExpertDecision(expert=str(fallback_expert), score=score, allow=allow, action=action, meta={})

    # numeric score
    if isinstance(raw, (int, float)):
        score = float(raw)
        return ExpertDecision(expert=str(fallback_expert), score=score, allow=(score > 0), action="hold", meta={})

    # unknown object -> try attribute access
    try:
        score = float(getattr(raw, "score"))
        allow = bool(getattr(raw, "allow", score > 0))
        action = str(getattr(raw, "action", "hold"))
        expert = str(getattr(raw, "expert", fallback_expert))
        meta = getattr(raw, "meta", {}) or {}
        return ExpertDecision(expert=expert, score=score, allow=allow, action=action, meta=dict(meta))
    except Exception:
        return ExpertDecision(
            expert=str(fallback_expert),
            score=0.0,
            allow=False,
            action="hold",
            meta={"coerce_error": repr(raw)},
        )
