from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass(init=False)
class ExpertDecision:
    """
    Normalized decision object returned by experts / gate.

    Backward compatible with older positional constructor:
        ExpertDecision(allow, score, expert, meta)
    New style:
        ExpertDecision(expert="X", score=0.7, allow=True, action="hold", meta={...})
    """

    expert: str
    score: float
    allow: bool
    action: str
    meta: Dict[str, Any]

    def __init__(
        self,
        *args: Any,
        expert: Optional[str] = None,
        score: float = 0.0,
        allow: bool = False,
        action: str = "hold",
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Support old positional: (allow, score, expert, meta)
        if args:
            # args[0]=allow, args[1]=score, args[2]=expert, args[3]=meta
            if len(args) >= 1 and "allow" not in kwargs and allow is False:
                allow = bool(args[0])
            if len(args) >= 2 and "score" not in kwargs and score == 0.0:
                try:
                    score = float(args[1])
                except Exception:
                    score = 0.0
            if len(args) >= 3 and expert is None:
                expert = str(args[2])
            if len(args) >= 4 and meta is None:
                try:
                    meta = dict(args[3]) if args[3] is not None else {}
                except Exception:
                    meta = {}

        # Also allow passing expert/score/allow/action/meta via kwargs (compat)
        if expert is None:
            expert = str(kwargs.get("expert", "UNKNOWN"))

        if meta is None:
            meta = kwargs.get("meta") or {}
        else:
            # merge any extra keys into meta (optional)
            pass

        # merge extra keys into meta (optional)
        extra = {k: v for k, v in kwargs.items() if k not in {"expert", "score", "allow", "action", "meta"}}
        if extra:
            meta = {**meta, **extra}

        self.expert = str(expert)
        self.score = float(score or 0.0)
        self.allow = bool(allow)
        self.action = str(action or "hold")
        self.meta = dict(meta)


class BaseExpert(Protocol):
    """Minimal interface for an Expert."""

    name: str

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any: ...


# Backward-compatible alias (some modules import ExpertBase)
ExpertBase = BaseExpert


def coerce_decision(raw: Any, fallback_expert: str = "UNKNOWN") -> Optional[ExpertDecision]:
    """Convert arbitrary expert output -> ExpertDecision."""
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
        extra = {k: v for k, v in raw.items() if k not in {"expert", "score", "allow", "action", "meta"}}
        if extra:
            meta = {**meta, **extra}
        return ExpertDecision(expert=expert, score=score, allow=allow, action=action, meta=meta)

    # tuple/list: (score, allow) or (score, allow, action)
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        score = float(raw[0] or 0.0)
        allow = bool(raw[1])
        action = str(raw[2]) if len(raw) >= 3 and raw[2] is not None else "hold"
        return ExpertDecision(expert=str(fallback_expert), score=score, allow=allow, action=action, meta={})

    # numeric score
    if isinstance(raw, (int, float)):
        score = float(raw)
        return ExpertDecision(expert=str(fallback_expert), score=score, allow=(score > 0), action="hold", meta={})

    # unknown object -> attribute access
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
