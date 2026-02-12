# brain/regime_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    vol: float
    slope: float
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": str(self.regime),
            "vol": float(self.vol),
            "slope": float(self.slope),
            "confidence": float(self.confidence),
        }


def _get_close(x: Dict[str, Any]) -> float:
    # support multiple formats: c/close/Close
    return float(x.get("c", x.get("close", x.get("Close", 0.0))) or 0.0)


def _conf_from_rule(regime: str, vol: float, slope: float) -> float:
    a_slope = abs(float(slope))
    vol = float(vol)

    if regime == "breakout":
        return max(0.0, min(1.0, 0.55 + 0.03 * max(0.0, a_slope - 8.0) + 0.10 * max(0.0, vol - 2.0)))
    if regime == "trend":
        return max(0.0, min(1.0, 0.45 + 0.03 * max(0.0, a_slope - 6.0) - 0.05 * max(0.0, 2.0 - vol)))
    if regime == "range":
        return max(0.0, min(1.0, 0.40 + 0.05 * max(0.0, 2.0 - a_slope) - 0.05 * max(0.0, vol - 2.0)))

    return 0.0


def detect_regime(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Output:
      { regime: 'trend'|'range'|'breakout'|'unknown', vol: float, slope: float, confidence: float }

    Keep old logic stable, only normalize + add confidence.
    """
    if not candles or len(candles) < 50:
        return {"regime": "unknown", "vol": 0.0, "slope": 0.0, "confidence": 0.0}

    tail = candles[-50:]
    closes = [_get_close(x) for x in tail]
    slope = float(closes[-1] - closes[0])

    diffs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    vol = float(sum(diffs) / max(1, len(diffs)))

    if vol > 2.0 and abs(slope) > 8.0:
        regime = "breakout"
    elif abs(slope) > 6.0:
        regime = "trend"
    else:
        regime = "range"

    conf = _conf_from_rule(regime, vol, slope)
    return {"regime": regime, "vol": vol, "slope": slope, "confidence": conf}


class RegimeDetector:
    """
    Backward-compatible wrapper:
    - detect(candles) -> RegimeResult
    - detect_regime(candles) -> dict
    """

    def __init__(self, window: int = 50) -> None:
        self.window = int(window)

    def detect(self, candles: List[Dict[str, Any]]) -> RegimeResult:
        ctx = detect_regime(candles)
        return RegimeResult(
            regime=str(ctx.get("regime", "unknown")),
            vol=float(ctx.get("vol", 0.0)),
            slope=float(ctx.get("slope", 0.0)),
            confidence=float(ctx.get("confidence", 0.0)),
        )
