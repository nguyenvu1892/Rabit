from __future__ import annotations
from typing import Any, Dict, List


def detect_regime(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    candles: list dict with keys: ts,o,h,l,c,v (hoặc close/...)
    Output: context dict { regime: 'trend'|'range'|'breakout', vol:..., slope:... }
    """
    if len(candles) < 50:
        return {"regime": "unknown", "vol": 0.0, "slope": 0.0}

    closes = [float(x.get("c", x.get("close"))) for x in candles[-50:]]
    # slope đơn giản: close_end - close_start
    slope = closes[-1] - closes[0]
    # vol proxy: avg abs diff
    diffs = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
    vol = sum(diffs) / max(1, len(diffs))

    # rule-based sơ cấp (pass 5.0.7.4)
    if vol > 2.0 and abs(slope) > 8.0:
        regime = "breakout"
    elif abs(slope) > 6.0:
        regime = "trend"
    else:
        regime = "range"

    return {"regime": regime, "vol": vol, "slope": slope}
