# brain/regime_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    vol: float
    slope: float
    confidence: float


class RegimeDetector:
    """
    A simple regime detector using:
    - log returns volatility (vol)
    - linear slope proxy (trend)
    - breakout when vol high and slope meaningful
    """

    def __init__(
        self,
        vol_win: int = 30,
        slope_win: int = 30,
        vol_hi: float = 0.006,
        slope_hi: float = 0.08,
        conf_scale: float = 1.0,
    ) -> None:
        self.vol_win = int(vol_win)
        self.slope_win = int(slope_win)
        self.vol_hi = float(vol_hi)
        self.slope_hi = float(slope_hi)
        self.conf_scale = float(conf_scale)

    def detect(self, candles: List[Dict]) -> RegimeResult:
        if not candles or len(candles) < max(self.vol_win, self.slope_win) + 2:
            return RegimeResult("unknown", 0.0, 0.0, 0.0)

        closes = []
        for c in candles:
            v = c.get("close", None)
            if v is None:
                v = c.get("c", None)
            if v is None:
                v = c.get("Close", None)
            if v is None:
                continue
            try:
                closes.append(float(v))
            except Exception:
                continue

        if len(closes) < max(self.vol_win, self.slope_win) + 2:
            return RegimeResult("unknown", 0.0, 0.0, 0.0)

        # --- volatility ---
        import math

        rets = []
        for i in range(1, len(closes)):
            a = closes[i - 1]
            b = closes[i]
            if a <= 0 or b <= 0:
                continue
            rets.append(math.log(b / a))
        if len(rets) < self.vol_win:
            return RegimeResult("unknown", 0.0, 0.0, 0.0)

        win_rets = rets[-self.vol_win :]
        mean_r = sum(win_rets) / max(1, len(win_rets))
        var = sum((x - mean_r) ** 2 for x in win_rets) / max(1, len(win_rets))
        vol = math.sqrt(var)

        # --- slope proxy ---
        sw = min(self.slope_win, len(closes) - 1)
        y0 = closes[-sw - 1]
        y1 = closes[-1]
        slope = (y1 - y0) / max(1e-9, abs(y0))

        # --- regime ---
        abs_slope = abs(slope)
        regime = "range"
        if vol >= self.vol_hi and abs_slope >= self.slope_hi:
            regime = "breakout"
        elif abs_slope >= (0.5 * self.slope_hi):
            regime = "trend"
        else:
            regime = "range"

        # normalize strictly
        if regime not in ("range", "trend", "breakout"):
            regime = "unknown"

        # --- confidence ---
        v_score = min(1.0, vol / max(1e-9, self.vol_hi))
        s_score = min(1.0, abs_slope / max(1e-9, self.slope_hi))
        confidence = float(min(1.0, (0.6 * v_score + 0.4 * s_score) * self.conf_scale))

        return RegimeResult(regime, float(vol), float(slope), confidence)
