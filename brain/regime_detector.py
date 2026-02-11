# brain/regime_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RegimeResult:
    regime: str               # "TREND_STRONG" | "TREND_WEAK" | "RANGE" | "VOLATILITY_SPIKE" | "MIXED" | "UNKNOWN"
    confidence: float         # 0..1
    metrics: Dict[str, float] # slope/vol/atr/...


class RegimeDetector:
    """
    Lightweight, rule-based regime detector.
    - Backward-compatible with older detect_regime(candles)->dict usage.
    - New API: detect(candles)->RegimeResult.
    """

    def __init__(
        self,
        window: int = 50,
        trend_slope_strong: float = 8.0,
        trend_slope_weak: float = 4.0,
        breakout_vol: float = 2.0,
        breakout_slope: float = 8.0,
        vol_spike_mult: float = 2.0,
    ) -> None:
        self.window = int(window)
        self.trend_slope_strong = float(trend_slope_strong)
        self.trend_slope_weak = float(trend_slope_weak)
        self.breakout_vol = float(breakout_vol)
        self.breakout_slope = float(breakout_slope)
        self.vol_spike_mult = float(vol_spike_mult)

    def detect(self, candles: List[Dict[str, Any]]) -> RegimeResult:
        if not candles or len(candles) < max(10, self.window):
            return RegimeResult("UNKNOWN", 0.0, {"slope": 0.0, "vol": 0.0})

        w = candles[-self.window :]
        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []

        for x in w:
            c = x.get("c", x.get("close", x.get("C", None)))
            h = x.get("h", x.get("high", x.get("H", None)))
            l = x.get("l", x.get("low", x.get("L", None)))
            if c is None:
                continue
            closes.append(float(c))
            if h is not None:
                highs.append(float(h))
            if l is not None:
                lows.append(float(l))

        if len(closes) < 10:
            return RegimeResult("UNKNOWN", 0.0, {"slope": 0.0, "vol": 0.0})

        slope = closes[-1] - closes[0]
        diffs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
        vol = sum(diffs) / max(1, len(diffs))

        # ATR-ish proxy from high/low if present, else fallback to vol
        if highs and lows and len(highs) == len(lows) == len(w):
            ranges = [abs(highs[i] - lows[i]) for i in range(len(highs))]
            atr = sum(ranges) / max(1, len(ranges))
        else:
            atr = vol

        # volatility spike vs median diff
        diffs_sorted = sorted(diffs) if diffs else [0.0]
        mid = diffs_sorted[len(diffs_sorted) // 2] if diffs_sorted else 0.0
        vol_spike = (mid > 0.0) and (vol > self.vol_spike_mult * mid)

        # classify
        regime: str
        if vol > self.breakout_vol and abs(slope) > self.breakout_slope:
            regime = "TREND_STRONG"  # treat breakout as strong trend for meta policy
        elif abs(slope) >= self.trend_slope_strong:
            regime = "TREND_STRONG"
        elif abs(slope) >= self.trend_slope_weak:
            regime = "TREND_WEAK"
        else:
            regime = "RANGE"

        if vol_spike and regime == "RANGE":
            regime = "VOLATILITY_SPIKE"

        # confidence heuristic (0..1)
        # stronger slope + cleaner move => higher confidence
        slope_strength = min(1.0, abs(slope) / max(1e-9, self.trend_slope_strong))
        vol_strength = min(1.0, vol / max(1e-9, self.breakout_vol))
        base_conf = 0.35 + 0.45 * slope_strength + 0.20 * (1.0 - min(1.0, vol_strength))
        if regime in ("TREND_STRONG", "TREND_WEAK"):
            conf = min(1.0, 0.45 + 0.55 * slope_strength)
        elif regime == "VOLATILITY_SPIKE":
            conf = 0.55
        elif regime == "RANGE":
            conf = max(0.30, min(0.80, base_conf))
        else:
            conf = 0.0

        metrics = {
            "slope": float(slope),
            "vol": float(vol),
            "atr": float(atr),
        }
        return RegimeResult(regime, float(conf), metrics)


def detect_regime(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for older code paths.
    Returns dict like:
      {"regime": "...", "vol": ..., "slope": ...}
    """
    r = RegimeDetector().detect(candles)
    out = {"regime": r.regime}
    out.update({k: float(v) for k, v in r.metrics.items()})
    out["confidence"] = float(r.confidence)
    return out
