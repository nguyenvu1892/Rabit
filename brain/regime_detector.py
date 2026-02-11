from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    vol: float
    slope: float
    confidence: float = 0.5  # simple heuristic, 0..1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "vol": float(self.vol),
            "slope": float(self.slope),
            "confidence": float(self.confidence),
        }


class RegimeDetector:
    """
    Backward-compatible wrapper.
    - New API: detect(candles) -> RegimeResult
    - Old API: detect_regime(candles) -> dict  (still exists below)
    """

    def __init__(self, window: int = 50) -> None:
        self.window = int(window)

    def detect(self, candles: List[Dict[str, Any]]) -> RegimeResult:
        ctx = detect_regime(candles)  # reuse existing logic
        regime = str(ctx.get("regime", "unknown"))
        vol = float(ctx.get("vol", 0.0))
        slope = float(ctx.get("slope", 0.0))

        # very light confidence heuristic
        if regime == "breakout":
            conf = 0.75
        elif regime == "trend":
            conf = 0.65
        elif regime == "range":
            conf = 0.55
        else:
            conf = 0.0

        return RegimeResult(regime=regime, vol=vol, slope=slope, confidence=conf)


def detect_regime(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    candles: list dict with keys: ts,o,h,l,c,v (hoặc close/...)
    Output: context dict { regime: 'trend'|'range'|'breakout', vol:..., slope:... }

    NOTE: giữ nguyên logic cũ để KHÔNG phá hệ thống đã build.
    """
    if len(candles) < 50:
        return {"regime": "unknown", "vol": 0.0, "slope": 0.0}

    tail = candles[-50:]
    closes = [float(x.get("c", x.get("close"))) for x in tail]

    slope = closes[-1] - closes[0]
    diffs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    vol = sum(diffs) / max(1, len(diffs))

    # rule-based sơ cấp (pass 5.0.7.4)
    if vol > 2.0 and abs(slope) > 8.0:
        regime = "breakout"
    elif abs(slope) > 6.0:
        regime = "trend"
    else:
        regime = "range"

    return {"regime": regime, "vol": vol, "slope": slope}
