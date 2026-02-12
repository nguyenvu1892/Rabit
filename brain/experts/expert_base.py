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
    score: float = 0.0
    allow: bool = True
    action: str = "hold"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert": str(self.expert),
            "score": float(self.score) if self.score is not None else 0.0,
            "allow": bool(self.allow),
            "action": str(self.action) if self.action else "hold",
            "meta": dict(self.meta) if isinstance(self.meta, dict) else {"meta": self.meta},
        }

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


class ExpertBase(Protocol):
    """Minimal interface for an Expert."""

    name: str

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[ExpertDecision]:
        ...


def coerce_decision(raw: Any, fallback_expert: str = "UNKNOWN") -> Optional[ExpertDecision]:
    """
    Normalize anything into ExpertDecision.
    Accepts:
      - ExpertDecision
      - dict-like {expert, score, allow, action, meta}
      - tuple/list ("buy"/"sell"/"hold", score)
    """
    if raw is None:
        return None

    if isinstance(raw, ExpertDecision):
        # normalize missing action in legacy objects (just in case)
        if not getattr(raw, "action", None):
            raw.action = "hold"
        return raw

    # dict-like
    if isinstance(raw, dict):
        expert = str(raw.get("expert", fallback_expert))
        score = raw.get("score", 0.0)
        allow = raw.get("allow", True)
        action = raw.get("action", "hold") or "hold"
        meta = raw.get("meta", {})
        if not isinstance(meta, dict):
            meta = {"meta": meta}
        try:
            score_f = float(score) if score is not None else 0.0
        except Exception:
            score_f = 0.0
        return ExpertDecision(expert=expert, score=score_f, allow=bool(allow), action=str(action), meta=meta)

    # tuple/list shorthand: (action, score)
    if isinstance(raw, (tuple, list)) and len(raw) >= 1:
        action = raw[0] if len(raw) >= 1 else "hold"
        score = raw[1] if len(raw) >= 2 else 0.0
        try:
            score_f = float(score) if score is not None else 0.0
        except Exception:
            score_f = 0.0
        return ExpertDecision(expert=str(fallback_expert), score=score_f, allow=True, action=str(action), meta={})

    # fallback: not supported
    return None