# brain/experts/experts_basic.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from brain.experts.expert_base import ExpertBase, ExpertDecision


class TrendMAExpert:
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

        closes = [float(x.get("c", x.get("close"))) for x in candles[-50:]]
        ma_fast = sum(closes[-10:]) / 10.0
        ma_slow = sum(closes) / 50.0

        if ma_fast > ma_slow:
            score, allow = 0.65, True
        else:
            score, allow = 0.35, False

        if (context.get("regime") or "") == "trend":
            score += 0.15

        return ExpertDecision(
            expert=self.name,
            allow=allow,
            score=min(score, 1.0),
            action="hold",
            meta={"ma_fast": ma_fast, "ma_slow": ma_slow, "regime": context.get("regime")},
        )


class MeanRevertExpert:
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

        closes = [float(x.get("c", x.get("close"))) for x in candles[-60:]]
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


class BreakoutExpert:
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

        closes = [float(x.get("c", x.get("close"))) for x in candles[-40:]]
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


class BaselineExpert:
    name = "BASELINE"

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        return ExpertDecision(
            expert=self.name,
            allow=True,
            score=0.0001,  # tiny >0 so it can be selected if needed
            action="hold",
            meta={"reason": "baseline"},
        )


# Default experts used by DecisionEngine/registry bootstrap
DEFAULT_EXPERTS: List[ExpertBase] = [
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
