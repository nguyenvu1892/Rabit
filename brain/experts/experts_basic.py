from __future__ import annotations
from typing import Any, Dict, List
from brain.experts.expert_base import BaseExpert, ExpertDecision


class TrendMAExpert(BaseExpert):
    name = "TREND_MA"

    def evaluate(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        candles: List[Dict[str, Any]] = trade_features["candles"]
        if len(candles) < 50:
            return ExpertDecision(False, 0.0, self.name, {"reason": "not_enough_data"})

        closes = [float(x.get("c", x.get("close"))) for x in candles[-50:]]
        ma_fast = sum(closes[-10:]) / 10.0
        ma_slow = sum(closes) / 50.0

        score = 0.0
        allow = False
        if ma_fast > ma_slow:
            score = 0.65
            allow = True
        else:
            score = 0.35
            allow = False

        # context bonus
        if context.get("regime") == "trend":
            score += 0.15

        return ExpertDecision(allow=allow, score=min(score, 1.0), expert=self.name,
                             meta={"ma_fast": ma_fast, "ma_slow": ma_slow, "regime": context.get("regime")})


class MeanRevertExpert(BaseExpert):
    name = "MEAN_REVERT"

    def evaluate(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        candles: List[Dict[str, Any]] = trade_features["candles"]
        if len(candles) < 60:
            return ExpertDecision(False, 0.0, self.name, {"reason": "not_enough_data"})

        closes = [float(x.get("c", x.get("close"))) for x in candles[-60:]]
        mean = sum(closes) / 60.0
        last = closes[-1]
        dist = abs(last - mean)

        # range regime ưu tiên mean reversion
        base = 0.40
        if context.get("regime") == "range":
            base = 0.65

        # càng xa mean score càng cao
        score = min(1.0, base + (dist / 10.0))
        allow = score >= 0.7

        return ExpertDecision(allow=allow, score=score, expert=self.name,
                             meta={"mean": mean, "last": last, "dist": dist, "regime": context.get("regime")})


class BreakoutExpert(BaseExpert):
    name = "BREAKOUT"

    def evaluate(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        candles: List[Dict[str, Any]] = trade_features["candles"]
        if len(candles) < 40:
            return ExpertDecision(False, 0.0, self.name, {"reason": "not_enough_data"})

        closes = [float(x.get("c", x.get("close"))) for x in candles[-40:]]
        hi = max(closes[:-1])
        lo = min(closes[:-1])
        last = closes[-1]

        broke_up = last > hi
        broke_dn = last < lo

        score = 0.2
        if broke_up or broke_dn:
            score = 0.75

        if context.get("regime") == "breakout":
            score += 0.15

        allow = score >= 0.8
        return ExpertDecision(allow=allow, score=min(score, 1.0), expert=self.name,
                             meta={"hi": hi, "lo": lo, "last": last, "regime": context.get("regime")})
