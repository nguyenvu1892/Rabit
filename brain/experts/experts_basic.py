# brain/experts/experts_basic.py
from __future__ import annotations

from typing import Any, Dict, List

from brain.experts.expert_base import ExpertBase, ExpertDecision


class TrendMAExpert(ExpertBase):
    name = "TREND_MA"

    def evaluate(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        candles: List[Dict[str, Any]] = trade_features.get("candles") or []
        if len(candles) < 50:
            return ExpertDecision(False, 0.0, self.name, {"reason": "not_enough_data", "n": len(candles)})

        closes = [float(x.get("c", x.get("close"))) for x in candles[-50:]]
        ma_fast = sum(closes[-10:]) / 10.0
        ma_slow = sum(closes) / 50.0

        if ma_fast > ma_slow:
            score, allow = 0.65, True
        else:
            score, allow = 0.35, False

        if context.get("regime") == "trend":
            score += 0.15

        return ExpertDecision(allow=allow, score=min(score, 1.0), expert=self.name,
                              meta={"ma_fast": ma_fast, "ma_slow": ma_slow, "regime": context.get("regime")})


class MeanRevertExpert(ExpertBase):
    name = "MEAN_REVERT"
    def evaluate(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        candles: List[Dict[str, Any]] = trade_features.get("candles") or []
        if len(candles) < 60:
            return ExpertDecision(False, 0.0, self.name, {"reason": "not_enough_data", "n": len(candles)})

        closes = [float(x.get("c", x.get("close"))) for x in candles[-60:]]
        mean = sum(closes) / 60.0
        last = closes[-1]
        dist = abs(last - mean)

        base = 0.65 if context.get("regime") == "range" else 0.40
        score = min(1.0, base + (dist / 10.0))
        allow = score >= 0.7
        return ExpertDecision(allow=allow, score=score, expert=self.name,
                              meta={"mean": mean, "last": last, "dist": dist, "regime": context.get("regime")})


class BreakoutExpert(ExpertBase):
    name = "BREAKOUT"
    def evaluate(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        candles: List[Dict[str, Any]] = trade_features.get("candles") or []
        if len(candles) < 40:
            return ExpertDecision(False, 0.0, self.name, {"reason": "not_enough_data", "n": len(candles)})

        closes = [float(x.get("c", x.get("close"))) for x in candles[-40:]]
        hi, lo, last = max(closes[:-1]), min(closes[:-1]), closes[-1]
        broke = (last > hi) or (last < lo)

        score = 0.75 if broke else 0.2
        if context.get("regime") == "breakout":
            score += 0.15

        allow = score >= 0.8
        return ExpertDecision(allow=allow, score=min(score, 1.0), expert=self.name,
                              meta={"hi": hi, "lo": lo, "last": last, "regime": context.get("regime")})


# ✅ để DecisionEngine/ExpertRegistry có thể “register default” rất tiện
DEFAULT_EXPERTS = [TrendMAExpert(), MeanRevertExpert(), BreakoutExpert()]

class BaselineExpert(ExpertBase):
    name = "BASELINE"

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        # Cho phép decision để engine không deny 100%
        return ExpertDecision(
            allow=True,
            score=0.0,
            expert=self.name,
            meta={"reason": "baseline"},
            action="hold",   # nếu hệ thống của bro có dùng action
        )

DEFAULT_EXPERTS: List[ExpertBase] = [BaselineExpert()]

def register_basic_experts(registry: Any) -> None:
    """
    DecisionEngine expects this function.
    Registry implementations differ, so we support common method names.
    """
    experts = [TrendMAExpert(), MeanRevertExpert(), BreakoutExpert()]

    if hasattr(registry, "register"):
        for e in experts:
            registry.register(e)
        return

    # fallback: some registries use add()
    if hasattr(registry, "add"):
        for e in experts:
            registry.add(e)
        return

    # last resort: store on a known attribute
    if not hasattr(registry, "_experts"):
        setattr(registry, "_experts", [])
    registry._experts.extend(experts)
