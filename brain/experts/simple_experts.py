# brain/experts/simple_experts.py
from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass
import math

from brain.experts.expert_base import ExpertSignal

def _ema(values: List[float], alpha: float) -> float:
    if not values:
        return 0.0
    x = values[0]
    for v in values[1:]:
        x = alpha * v + (1 - alpha) * x
    return x

def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(max(v, 0.0))

@dataclass
class TrendExpert:
    name: str = "TREND_MA"

    def evaluate(self, trade_features: Dict[str, Any]) -> ExpertSignal:
        w = trade_features.get("candles_window", [])
        if len(w) < 120:
            return ExpertSignal("neutral", 0.0, ["not_enough_data"])

        closes = [float(c["c"]) for c in w]
        fast = _ema(closes[-60:], alpha=2/(20+1))
        slow = _ema(closes[-200:], alpha=2/(80+1))
        diff = fast - slow

        # confidence based on separation
        std = _std(closes[-200:])
        conf = min(1.0, abs(diff) / (std + 1e-9))

        if conf < 0.15:
            return ExpertSignal("neutral", conf, [f"ma_diff_small:{diff:.4f}"])

        side = "long" if diff > 0 else "short"
        return ExpertSignal(side, conf, [f"ma_diff:{diff:.4f}", f"conf:{conf:.3f}"])

@dataclass
class MeanReversionExpert:
    name: str = "RANGE_ZSCORE"

    def evaluate(self, trade_features: Dict[str, Any]) -> ExpertSignal:
        w = trade_features.get("candles_window", [])
        if len(w) < 120:
            return ExpertSignal("neutral", 0.0, ["not_enough_data"])

        closes = [float(c["c"]) for c in w[-200:]]
        m = sum(closes) / len(closes)
        sd = _std(closes)
        if sd < 1e-9:
            return ExpertSignal("neutral", 0.0, ["sd_too_small"])

        z = (closes[-1] - m) / sd
        # revert: high z -> short, low z -> long
        az = abs(z)
        conf = min(1.0, az / 2.5)

        if az < 1.2:
            return ExpertSignal("neutral", conf, [f"z_small:{z:.2f}"])

        side = "short" if z > 0 else "long"
        return ExpertSignal(side, conf, [f"z:{z:.2f}", f"conf:{conf:.3f}"])

@dataclass
class DummySMCExpert:
    """
    Placeholder for SMC/ICT/FVG later.
    For now returns neutral to keep framework stable.
    """
    name: str = "SMC_PLACEHOLDER"

    def evaluate(self, trade_features: Dict[str, Any]) -> ExpertSignal:
        return ExpertSignal("neutral", 0.0, ["placeholder"])
