# brain/regime_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RegimeResult:
    regime: str = "unknown"
    vol: float = 0.0
    slope: float = 0.0
    confidence: float = 0.0


class RegimeDetector:
    """
    Minimal regime detector:
    - expects features may contain keys like 'vol', 'slope'
    - returns RegimeResult(regime in {range, trend, breakout, unknown})
    """

    def __init__(
        self,
        *,
        vol_breakout: float = 1.2,
        slope_trend: float = 0.15,
        debug: bool = False,
    ) -> None:
        self.vol_breakout = float(vol_breakout)
        self.slope_trend = float(slope_trend)
        self.debug = debug

    def detect(self, features: Dict[str, Any]) -> RegimeResult:
        vol = 0.0
        slope = 0.0

        try:
            vol = float(features.get("vol", 0.0))
        except Exception:
            vol = 0.0

        try:
            slope = float(features.get("slope", 0.0))
        except Exception:
            slope = 0.0

        # heuristic
        regime = "range"
        conf = 0.5

        if abs(slope) >= self.slope_trend:
            regime = "trend"
            conf = min(1.0, 0.5 + abs(slope))
        elif vol >= self.vol_breakout:
            regime = "breakout"
            conf = min(1.0, 0.4 + (vol - self.vol_breakout))
        else:
            regime = "range"
            conf = 0.55

        if not regime:
            regime = "unknown"
            conf = 0.0

        if self.debug:
            print("REGIME:", regime, "vol:", vol, "slope:", slope, "conf:", conf)

        return RegimeResult(regime=regime, vol=vol, slope=slope, confidence=float(conf))
