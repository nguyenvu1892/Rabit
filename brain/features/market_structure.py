# brain/feature/market_structure.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass
class MarketStructureFeatures:
    """
    Deterministic market-structure features from OHLCV candles.
    candles items expected keys: o,h,l,c (v optional)
    """
    name: str = "ms"
    lookback: int = 20
    breakout_lookback: int = 10
    slope_threshold: float = 0.0001  # relative threshold (scaled later)

    def compute(self, candles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if candles is None or len(candles) < 3:
            return {}

        closes = [float(x["c"]) for x in candles]
        highs = [float(x["h"]) for x in candles]
        lows = [float(x["l"]) for x in candles]

        n = min(self.lookback, len(candles))
        recent_closes = closes[-n:]

        # --- slope estimate (simple) ---
        first = recent_closes[0]
        last = recent_closes[-1]
        if abs(first) < 1e-9:
            slope = 0.0
        else:
            slope = (last - first) / abs(first)  # relative

        # --- trend_state ---
        if slope > self.slope_threshold:
            trend_state = "up"
        elif slope < -self.slope_threshold:
            trend_state = "down"
        else:
            trend_state = "range"

        # --- HH/HL vs LH/LL using last two swings (simplified: compare last 2 highs/lows) ---
        # Use last 3 candles highs/lows for a very stable minimal v1.
        h1, h2 = highs[-2], highs[-1]
        l1, l2 = lows[-2], lows[-1]
        if h2 > h1 and l2 > l1:
            structure = "hh_hl"
        elif h2 < h1 and l2 < l1:
            structure = "lh_ll"
        else:
            structure = "mixed"

        # --- breakout check ---
        m = min(self.breakout_lookback, len(candles) - 1)
        prev_high = max(highs[-(m + 1):-1])
        prev_low = min(lows[-(m + 1):-1])
        breakout_up = 1 if last > prev_high else 0
        breakout_down = 1 if last < prev_low else 0

        return {
            "trend_state": trend_state,
            "structure": structure,
            "slope": float(slope),
            "breakout_up": int(breakout_up),
            "breakout_down": int(breakout_down),
            "prev_high": float(prev_high),
            "prev_low": float(prev_low),
        }
