# brain/experts/expert_gate.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

try:
    # canonical (5.1.x)
    from brain.experts.expert_base import ExpertDecision, ExpertBase
except Exception:
    # ultra-compat fallback (avoid import crash)
    @dataclass
    class ExpertDecision:  # type: ignore
        expert: str
        score: float = 0.0
        allow: bool = True
        action: str = "hold"
        meta: Dict[str, Any] = field(default_factory=dict)

    class ExpertBase:  # type: ignore
        name: str = "BASE"
        def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Any:
            return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _coerce_decision(
    raw: Any,
    *,
    fallback_expert: str = "UNKNOWN",
    fallback_meta: Optional[Dict[str, Any]] = None
) -> Optional[ExpertDecision]:
    """
    Compat layer:
    - ExpertDecision -> passthrough
    - dict -> parse keys {expert,name,score,allow,action,meta}
    - tuple/list -> (allow, score, action?) or (score, action?) etc.
    - bool/float/int -> score only (bool means allow with score 0/1)
    - None -> None
    """
    if raw is None:
        return None

    if isinstance(raw, ExpertDecision):
        # ensure fields exist
        if getattr(raw, "allow", None) is None:
            raw.allow = True  # type: ignore
        if getattr(raw, "action", None) is None:
            raw.action = "hold"  # type: ignore
        if getattr(raw, "meta", None) is None:
            raw.meta = {}  # type: ignore
        return raw

    if isinstance(raw, dict):
        expert = str(raw.get("expert") or raw.get("name") or fallback_expert)
        score = _safe_float(raw.get("score"), 0.0)
        allow = bool(raw.get("allow", True))
        action = str(raw.get("action") or "hold")
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        # preserve any other keys
        for k, v in raw.items():
            if k not in ("expert", "name", "score", "allow", "action", "meta"):
                meta.setdefault(k, v)
        if fallback_meta:
            meta = {**fallback_meta, **meta}
        return ExpertDecision(expert=expert, score=score, allow=allow, action=action, meta=meta)

    if isinstance(raw, (tuple, list)):
        # Try common patterns
        # (allow, score, action, meta)
        allow = True
        score = 0.0
        action = "hold"
        meta: Dict[str, Any] = {}

        if len(raw) >= 1:
            # first could be allow or score
            if isinstance(raw[0], bool):
                allow = bool(raw[0])
            else:
                score = _safe_float(raw[0], 0.0)

        if len(raw) >= 2:
            # second could be score or action
            if isinstance(raw[1], (int, float)) and not isinstance(raw[1], bool):
                score = _safe_float(raw[1], score)
            elif isinstance(raw[1], str):
                action = raw[1]
            else:
                # ignore
                pass

        if len(raw) >= 3:
            if isinstance(raw[2], str):
                action = raw[2]
            elif isinstance(raw[2], (int, float)) and not isinstance(raw[2], bool):
                score = _safe_float(raw[2], score)
            elif isinstance(raw[2], dict):
                meta = raw[2]

        if len(raw) >= 4 and isinstance(raw[3], dict):
            meta = raw[3]

        if fallback_meta:
            meta = {**fallback_meta, **meta}

        return ExpertDecision(expert=fallback_expert, score=score, allow=allow, action=action, meta=meta)

    if isinstance(raw, bool):
        return ExpertDecision(expert=fallback_expert, score=1.0 if raw else 0.0, allow=True, action="hold",
                              meta=fallback_meta or {})

    if isinstance(raw, (int, float)):
        return ExpertDecision(expert=fallback_expert, score=_safe_float(raw, 0.0), allow=True, action="hold",
                              meta=fallback_meta or {})

    # unknown type -> best effort
    return ExpertDecision(
        expert=fallback_expert,
        score=0.0,
        allow=True,
        action="hold",
        meta={**(fallback_meta or {}), "raw_type": str(type(raw)), "raw": repr(raw)},
    )


class ExpertGate:
    """
    Collect expert decisions, apply optional weights, pick best.
    IMPORTANT: Never return (None, []) in normal flow — always fallback to HOLD
    so the whole system doesn't become deny=100%.
    """

    def __init__(
        self,
        registry: Any,
        weight_store: Optional[Any] = None,
        *,
        debug: bool = False,
    ) -> None:
        self.registry = registry
        self.weight_store = weight_store
        self.debug = debug

    def _iter_experts(self) -> List[Any]:
        # compat registry.get_all() may exist; else registry.experts or registry list
        try:
            xs = self.registry.get_all()
            return list(xs) if xs else []
        except Exception:
            pass
        try:
            xs = getattr(self.registry, "experts", None)
            if xs:
                return list(xs)
        except Exception:
            pass
        if isinstance(self.registry, (list, tuple)):
            return list(self.registry)
        return []

    def pick(
        self,
        features: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[ExpertDecision, List[ExpertDecision]]:
        regime = None
        try:
            regime = context.get("regime")
        except Exception:
            regime = None

        decisions: List[ExpertDecision] = []
        experts = self._iter_experts()

        if self.debug:
            names = []
            for e in experts:
                names.append(getattr(e, "name", e.__class__.__name__))
            print("DEBUG REGISTRY EXPERTS:", names)

        for exp in experts:
            name = str(getattr(exp, "name", exp.__class__.__name__))
            try:
                raw = exp.decide(features, context)
            except Exception as e:
                # keep error as a "soft" decision (allow True but hold)
                dec = ExpertDecision(
                    expert=name,
                    score=0.0,
                    allow=True,
                    action="hold",
                    meta={"error": repr(e), "reason": "expert_exception"},
                )
                decisions.append(dec)
                continue

            dec = _coerce_decision(raw, fallback_expert=name, fallback_meta={"reason": "coerced"})
            if dec is None:
                continue

            # Apply weight_store if present: adjusted_score = raw_score * w
            w = 1.0
            if self.weight_store is not None and hasattr(self.weight_store, "get"):
                try:
                    w = _safe_float(self.weight_store.get(dec.expert, regime), 1.0)
                except Exception:
                    w = 1.0

            # keep raw score in meta for trace
            try:
                dec.meta = dec.meta or {}
                dec.meta.setdefault("raw_score", dec.score)
                dec.meta.setdefault("w", w)
                dec.meta.setdefault("regime", regime)
            except Exception:
                pass

            dec.score = _safe_float(dec.score, 0.0) * _safe_float(w, 1.0)
            decisions.append(dec)

        # HARD FALLBACK: if nothing, create a safe HOLD allow=True
        if not decisions:
            fb = ExpertDecision(
                expert="FALLBACK",
                score=0.0,
                allow=True,
                action="hold",
                meta={"reason": "no_expert_decision"},
            )
            if self.debug:
                print("DEBUG FALLBACK: no decisions -> HOLD")
            return fb, [fb]

        # Prefer allow=True decisions; if none, still fallback to HOLD allow=True
        allow_pool = [d for d in decisions if bool(getattr(d, "allow", True))]
        pool = allow_pool if allow_pool else decisions

        # pick best by score
        best = None
        best_score = float("-inf")
        for d in pool:
            s = _safe_float(getattr(d, "score", 0.0), 0.0)
            if s > best_score:
                best_score = s
                best = d

        if best is None:
            best = ExpertDecision(
                expert="FALLBACK",
                score=0.0,
                allow=True,
                action="hold",
                meta={"reason": "best_none_after_pool"},
            )
            decisions.append(best)

        if self.debug:
            try:
                print("DEBUG BEST_EXPERT:", best.expert)
                print("DEBUG BEST_SCORE:", best.score)
                print("DEBUG BEST_META:", best.meta)
            except Exception:
                pass

        return best, decisions
