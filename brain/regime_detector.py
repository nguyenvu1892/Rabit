# brain/regime_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    vol: float
    slope: float
    confidence: float


class RegimeDetector:
    """
    Regime logic (giữ nguyên ý tưởng cũ):
      - trend: slope cao và đủ tự tin
      - breakout: vol cao và đủ tự tin
      - range: còn lại
    Upgrade 5.1.8:
      - Nếu features không có vol/slope, tự suy ra từ candles (close series)
        để tránh regime bị "unknown" hoặc vô nghĩa.
    """

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug

    def _extract_closes(self, candles: Any) -> List[float]:
        closes: List[float] = []
        if candles is None:
            return closes

        # candles can be: list[dict], list[list/tuple], list[float]
        try:
            for c in candles:
                if c is None:
                    continue
                if isinstance(c, (int, float)):
                    closes.append(float(c))
                    continue
                if isinstance(c, dict):
                    # common keys
                    if "close" in c:
                        closes.append(float(c["close"]))
                    elif "c" in c:
                        closes.append(float(c["c"]))
                    else:
                        # try last numeric value
                        for k in ("Close", "CLOSE", "price", "last"):
                            if k in c:
                                closes.append(float(c[k]))
                                break
                    continue
                if isinstance(c, (list, tuple)):
                    # heuristics: often [t, o, h, l, c, v] or [o,h,l,c]
                    if len(c) >= 5:
                        closes.append(float(c[4]))
                    elif len(c) >= 4:
                        closes.append(float(c[3]))
        except Exception:
            return closes

        return closes

    def _compute_slope_vol(self, closes: List[float]) -> (float, float):
        n = len(closes)
        if n < 3:
            return 0.0, 0.0

        # slope: normalized delta over window
        first = closes[0]
        last = closes[-1]
        denom = abs(first) if abs(first) > 1e-9 else 1.0
        slope = (last - first) / denom  # normalized

        # vol: mean absolute return
        rets: List[float] = []
        for i in range(1, n):
            prev = closes[i - 1]
            cur = closes[i]
            d = abs(prev) if abs(prev) > 1e-9 else 1.0
            rets.append((cur - prev) / d)

        if not rets:
            vol = 0.0
        else:
            vol = sum(abs(r) for r in rets) / float(len(rets))

        return float(slope), float(vol)

    def detect(self, features: Dict[str, Any]) -> RegimeResult:
        # Hardening: features may not be dict in some call paths → avoid crashing.
        if not isinstance(features, dict):
            return RegimeResult(regime="unknown", vol=0.0, slope=0.0, confidence=0.0)

        # Prefer explicit vol/slope if provided
        vol = 0.0
        slope = 0.0
        try:
            if "vol" in features and features["vol"] is not None:
                vol = float(features.get("vol") or 0.0)
            if "slope" in features and features["slope"] is not None:
                slope = float(features.get("slope") or 0.0)
        except Exception:
            vol, slope = 0.0, 0.0

        # If missing -> infer from candles
        if (vol == 0.0 and slope == 0.0) and ("candles" in features):
            closes = self._extract_closes(features.get("candles"))
            s, v = self._compute_slope_vol(closes)
            slope = float(slope or s)
            vol = float(vol or v)

        # Base regime
        regime = "range"
        if abs(slope) > 0.002:
            regime = "trend"
        if vol > 0.003:
            regime = "breakout"

        # Confidence (giữ logic cũ, nhưng đảm bảo không luôn = 0)
        confidence = 0.55
        confidence += min(abs(slope) * 40.0, 0.25)
        confidence += min(vol * 40.0, 0.20)
        if confidence > 0.99:
            confidence = 0.99
        if confidence < 0.0:
            confidence = 0.0

        if self.debug:
            try:
                print("DEBUG REGIME:", regime, "vol=", vol, "slope=", slope, "conf=", confidence)
            except Exception:
                pass

        return RegimeResult(regime=regime, vol=float(vol), slope=float(slope), confidence=float(confidence))