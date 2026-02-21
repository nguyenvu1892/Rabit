# brain/experts/experts_basic.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --- Robust imports (NEVER crash this module) ---
try:
    from brain.experts.expert_base import ExpertBase, ExpertDecision
except Exception:
    # Ultra-compat fallback: keep system alive even if expert_base is unstable.
    class ExpertBase:  # type: ignore
        name: str = "BASE"

        def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
            return None

    @dataclass
    class ExpertDecision:  # type: ignore
        expert: str
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


# ----------------------------
# Experts
# ----------------------------
class TrendMAExpert(ExpertBase):
    name = "TREND_MA"

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        context = context or {}
        candles: List[Dict[str, Any]] = features.get("candles") or []
        if len(candles) < 50:
            return ExpertDecision(
                expert=self.name,
                allow=False,
                score=0.0,
                action="hold",
                meta={"reason": "not_enough_data", "n": len(candles)},
            )

        closes = [_safe_float(x.get("c", x.get("close")), 0.0) for x in candles[-50:]]
        ma_fast = sum(closes[-10:]) / 10.0
        ma_slow = sum(closes) / 50.0

        if ma_fast > ma_slow:
            score, allow = 0.65, True
        else:
            score, allow = 0.35, False

        # regime booster (tùy ý)
        if (context.get("regime") or "") in ("trend_up", "trend_down", "trend"):
            score += 0.15

        return ExpertDecision(
            expert=self.name,
            allow=allow,
            score=min(score, 1.0),
            action="hold",
            meta={"ma_fast": ma_fast, "ma_slow": ma_slow, "regime": context.get("regime")},
        )


class MeanRevertExpert(ExpertBase):
    name = "MEAN_REVERT"

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        context = context or {}
        candles: List[Dict[str, Any]] = features.get("candles") or []
        if len(candles) < 60:
            return ExpertDecision(
                expert=self.name,
                allow=False,
                score=0.0,
                action="hold",
                meta={"reason": "not_enough_data", "n": len(candles)},
            )

        closes = [_safe_float(x.get("c", x.get("close")), 0.0) for x in candles[-60:]]
        mean = sum(closes) / 60.0
        last = closes[-1]
        dist = abs(last - mean)

        base = 0.65 if (context.get("regime") or "") == "range" else 0.40
        score = min(1.0, base + (dist / 10.0))
        allow = score >= 0.7

        return ExpertDecision(
            expert=self.name,
            allow=allow,
            score=score,
            action="hold",
            meta={"mean": mean, "last": last, "dist": dist, "regime": context.get("regime")},
        )


class BreakoutExpert(ExpertBase):
    name = "BREAKOUT"

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        context = context or {}
        candles: List[Dict[str, Any]] = features.get("candles") or []
        if len(candles) < 40:
            return ExpertDecision(
                expert=self.name,
                allow=False,
                score=0.0,
                action="hold",
                meta={"reason": "not_enough_data", "n": len(candles)},
            )

        closes = [_safe_float(x.get("c", x.get("close")), 0.0) for x in candles[-40:]]
        hi, lo, last = max(closes[:-1]), min(closes[:-1]), closes[-1]

        broke = (last > hi) or (last < lo)
        score = 0.75 if broke else 0.2

        if (context.get("regime") or "") == "breakout":
            score += 0.15

        allow = score >= 0.8

        return ExpertDecision(
            expert=self.name,
            allow=allow,
            score=min(score, 1.0),
            action="hold",
            meta={"hi": hi, "lo": lo, "last": last, "regime": context.get("regime")},
        )


class BaselineExpert(ExpertBase):
    name = "BASELINE"

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        # Always allow with tiny score so the system never goes empty.
        return ExpertDecision(
            expert=self.name,
            allow=True,
            score=0.0001,
            action="hold",
            meta={"reason": "baseline"},
        )


# Default experts used by DecisionEngine/registry bootstrap
DEFAULT_EXPERTS: List[Any] = [
    BaselineExpert(),
    TrendMAExpert(),
    MeanRevertExpert(),
    BreakoutExpert(),
]


def register_basic_experts(registry: Any) -> None:
    """Registry implementations differ, so we support common method names."""
    if hasattr(registry, "register"):
        for e in DEFAULT_EXPERTS:
            registry.register(e)
        return
    if hasattr(registry, "add"):
        for e in DEFAULT_EXPERTS:
            registry.add(e)
        return

    # last resort: store on attribute
    if not hasattr(registry, "_experts"):
        setattr(registry, "_experts", [])
    registry._experts.extend(DEFAULT_EXPERTS)