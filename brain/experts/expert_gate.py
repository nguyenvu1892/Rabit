# brain/experts/expert_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from brain.experts.expert_base import ExpertDecision
except Exception:
    # Ultra fallback: never crash import
    from dataclasses import dataclass, field
    from typing import Any, Dict

    @dataclass
    class ExpertDecision:  # type: ignore
        expert: str = "UNKNOWN"
        allow: bool = True
        score: float = 0.0
        action: str = "hold"
        meta: Dict[str, Any] = field(default_factory=dict)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@dataclass
class GateOut:
    """
    Internal helper: output of gate selection.
    """
    best: ExpertDecision
    debug: Dict[str, Any]


class ExpertGate:
    """
    Gate evaluates multiple experts and chooses the best.

    COMPAT:
    - expose evaluate(...) and evaluate_trade(...) and decide(...)
    - never crash if registry shape differs
    - never return empty: if no experts, return a baseline allow=True decision
    """

    def __init__(self, registry: Any, weight_store: Optional[Any] = None, debug: bool = False, **kwargs: Any) -> None:
        self.registry = registry
        self.weight_store = weight_store
        self.debug = bool(debug)
        # Accept/ignore compat kwargs so older constructors don't crash
        self._compat_kwargs = dict(kwargs)

    # ----------------------------
    # Registry helpers (compat)
    # ----------------------------
    def _get_experts(self) -> List[Any]:
        r = self.registry
        if r is None:
            return []

        # Common patterns
        if hasattr(r, "get_all"):
            try:
                xs = r.get_all()
                return list(xs) if xs is not None else []
            except Exception:
                pass

        if hasattr(r, "all"):
            try:
                xs = r.all()
                return list(xs) if xs is not None else []
            except Exception:
                pass

        if hasattr(r, "_experts"):
            try:
                xs = getattr(r, "_experts")
                return list(xs) if xs is not None else []
            except Exception:
                pass

        return []

    def _baseline(self) -> ExpertDecision:
        return ExpertDecision(expert="BASELINE", allow=True, score=0.0001, action="hold", meta={"reason": "gate_baseline"})

    # ----------------------------
    # Main evaluation
    # ----------------------------
    def evaluate_trade(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        context = context or {}
        experts = self._get_experts()

        if not experts:
            # never deny 100% due to empty registry
            d = self._baseline()
            d.meta["context"] = context
            return d

        best: Optional[ExpertDecision] = None
        best_weighted = -1e18
        dbg_rows: List[Dict[str, Any]] = []

        for e in experts:
            name = getattr(e, "name", getattr(e, "__class__", type(e)).__name__)
            try:
                # Some experts use decide(features, context), some decide(features)
                try:
                    out = e.decide(features, context=context)
                except TypeError:
                    out = e.decide(features)
            except Exception as ex:
                out = ExpertDecision(expert=str(name), allow=False, score=0.0, action="hold", meta={"error": str(ex)})

            # Normalize output into ExpertDecision
            if not isinstance(out, ExpertDecision):
                # try to coerce dict-like
                if isinstance(out, dict):
                    out = ExpertDecision(**out)
                else:
                    out = ExpertDecision(expert=str(name), allow=False, score=0.0, action="hold", meta={"raw": str(out)})

            # Ensure expert name
            if not out.expert or out.expert == "UNKNOWN":
                out.expert = str(name)

            raw = _safe_float(getattr(out, "score", 0.0), 0.0)

            # weight lookup (optional)
            w = 1.0
            if self.weight_store is not None:
                try:
                    # weight_store might expose get(key) or get_weight(expert, regime)
                    regime = context.get("regime") or context.get("market_regime") or "unknown"
                    key = f"{out.expert}|{regime}"
                    if hasattr(self.weight_store, "get"):
                        w = _safe_float(self.weight_store.get("expert_regime", key, default=1.0), 1.0)
                    elif hasattr(self.weight_store, "get_weight"):
                        w = _safe_float(self.weight_store.get_weight(out.expert, regime), 1.0)
                except Exception:
                    w = 1.0

            weighted = raw * w

            if self.debug:
                dbg_rows.append(
                    {
                        "expert": out.expert,
                        "allow": bool(out.allow),
                        "raw_score": raw,
                        "weight": w,
                        "weighted": weighted,
                    }
                )

            # Selection policy:
            # - Primary: higher weighted score
            # - Secondary: prefer allow=True if tie-ish
            if best is None:
                best, best_weighted = out, weighted
            else:
                if weighted > best_weighted + 1e-12:
                    best, best_weighted = out, weighted
                elif abs(weighted - best_weighted) <= 1e-12:
                    # tie: prefer allow=True
                    if bool(out.allow) and not bool(best.allow):
                        best, best_weighted = out, weighted

        if best is None:
            best = self._baseline()

        # Attach debug meta
        best.meta = best.meta or {}
        best.meta.setdefault("context", context)
        if self.debug:
            best.meta["gate_debug"] = dbg_rows
            best.meta["gate_best_weighted"] = best_weighted

        return best

    # ----------------------------
    # COMPAT ALIASES (do not remove)
    # ----------------------------
    def evaluate(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        # older code calls gate.evaluate(...)
        return self.evaluate_trade(features, context=context)

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        # some code calls gate.decide(...)
        return self.evaluate_trade(features, context=context)