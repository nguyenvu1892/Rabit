# brain/feature/price_action.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / (b if abs(b) > eps else eps)


@dataclass
class PriceActionFeatures:
    """
    Deterministic candle-pattern features using last candle(s).
    candles items expected keys: o,h,l,c (v optional)
    """
    name: str = "pa"
    pinbar_wick_ratio: float = 2.0   # wick >= 2x body
    pinbar_body_max: float = 0.35    # body <= 35% of range
    engulf_min_body_ratio: float = 1.05  # current body >= 1.05x prev body

    def compute(self, candles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if candles is None or len(candles) < 2:
            return {}

        prev = candles[-2]
        cur = candles[-1]

        o = float(cur["o"]); h = float(cur["h"]); l = float(cur["l"]); c = float(cur["c"])
        po = float(prev["o"]); ph = float(prev["h"]); pl = float(prev["l"]); pc = float(prev["c"])

        rng = max(h - l, 1e-9)
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        body_ratio = body / rng
        upper_wick_ratio = _safe_div(upper_wick, body)
        lower_wick_ratio = _safe_div(lower_wick, body)

        # candle direction
        bull = 1 if c > o else 0
        bear = 1 if c < o else 0

        # pinbar rules (objective)
        pinbar_bull = 1 if (lower_wick_ratio >= self.pinbar_wick_ratio and body_ratio <= self.pinbar_body_max and bull == 1) else 0
        pinbar_bear = 1 if (upper_wick_ratio >= self.pinbar_wick_ratio and body_ratio <= self.pinbar_body_max and bear == 1) else 0

        # engulfing rules (objective)
        prev_body = abs(pc - po)
        cur_body = abs(c - o)

        bullish_engulf = 1 if (
            c > o and pc < po and            # cur bull, prev bear
            o <= pc and c >= po and          # body engulfs
            _safe_div(cur_body, max(prev_body, 1e-9)) >= self.engulf_min_body_ratio
        ) else 0

        bearish_engulf = 1 if (
            c < o and pc > po and            # cur bear, prev bull
            o >= pc and c <= po and          # body engulfs
            _safe_div(cur_body, max(prev_body, 1e-9)) >= self.engulf_min_body_ratio
        ) else 0

        # close position in range (0..1)
        close_pos = (c - l) / rng  # 0 near low, 1 near high

        return {
            "bull": int(bull),
            "bear": int(bear),
            "range": float(rng),
            "body": float(body),
            "body_ratio": float(body_ratio),
            "upper_wick": float(upper_wick),
            "lower_wick": float(lower_wick),
            "upper_wick_ratio": float(upper_wick_ratio),
            "lower_wick_ratio": float(lower_wick_ratio),
            "close_pos": float(close_pos),

            "pinbar_bull": int(pinbar_bull),
            "pinbar_bear": int(pinbar_bear),
            "bullish_engulf": int(bullish_engulf),
            "bearish_engulf": int(bearish_engulf),
        }
