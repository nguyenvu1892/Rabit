# brain/regime_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import math


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    vol: float
    slope: float
    confidence: float


class RegimeDetector:

    def __init__(
        self,
        *,
        debug: bool = False,
        window: int = 32,
        min_slope_th: float = 0.0010,
        min_vol_th: float = 0.0015,
    ) -> None:
        self.debug = bool(debug)
        self.window = int(window) if window and window > 2 else 32
        self.min_slope_th = float(min_slope_th)
        self.min_vol_th = float(min_vol_th)

    def _dprint(self, *args: Any) -> None:
        if self.debug:
            try:
                print(*args)
            except Exception:
                pass

    # ==============================
    # FIX: parse tab-separated candle
    # ==============================
    def _extract_closes(self, candles: Any) -> List[float]:
        closes: List[float] = []

        if candles is None:
            return closes

        try:
            for c in candles:

                # case 1: dict with tab-separated string value
                if isinstance(c, dict) and len(c) == 1:
                    raw = list(c.values())[0]
                    if isinstance(raw, str):
                        parts = raw.split("\t")
                        if len(parts) >= 6:
                            closes.append(float(parts[5]))
                            continue

                # case 2: dict normal OHLC
                if isinstance(c, dict):
                    for k in ("close", "c", "Close", "CLOSE"):
                        if k in c:
                            closes.append(float(c[k]))
                            break
                    continue

                # case 3: list/tuple OHLC
                if isinstance(c, (list, tuple)):
                    if len(c) >= 5:
                        closes.append(float(c[4]))
                    elif len(c) >= 4:
                        closes.append(float(c[3]))
                    continue

        except Exception:
            return closes

        return closes

    def _compute_slope_vol(self, closes: List[float]) -> (float, float):
        n = len(closes)
        if n < 3:
            return 0.0, 0.0

        w = closes[-self.window:] if n > self.window else closes
        if len(w) < 3:
            return 0.0, 0.0

        first = w[0]
        last = w[-1]

        denom = abs(first) if abs(first) > 1e-9 else 1.0
        slope = (last - first) / denom

        rets = []
        for i in range(1, len(w)):
            prev = w[i - 1]
            cur = w[i]
            d = abs(prev) if abs(prev) > 1e-9 else 1.0
            rets.append((cur - prev) / d)

        if not rets:
            return float(slope), 0.0

        vol = sum(abs(r) for r in rets) / len(rets)
        return float(slope), float(vol)

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except Exception:
            return 0.5

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def detect(self, features: Dict[str, Any]) -> RegimeResult:
        if not isinstance(features, dict):
            return RegimeResult("unknown", 0.0, 0.0, 0.0)

        candles = features.get("candles")
        closes = self._extract_closes(candles)

        slope, vol = self._compute_slope_vol(closes)

        abs_vol = abs(vol)
        abs_slope = abs(slope)

        slope_th = max(self.min_slope_th, 2.6 * abs_vol)
        vol_th = max(self.min_vol_th, 2.2 * abs_vol + self.min_vol_th * 0.5)

        if abs_slope > slope_th:
            regime = "trend_up" if slope > 0 else "trend_down"
        else:
            regime = "volatile" if abs_vol > vol_th else "range"

        rel_trend = abs_slope / (slope_th + 1e-12)
        rel_vol = abs_vol / (vol_th + 1e-12)

        trend_conf = self._sigmoid(2.0 * (rel_trend - 1.0))
        vol_conf = self._sigmoid(2.0 * (rel_vol - 1.0))

        confidence = 0.52
        if regime.startswith("trend_"):
            confidence += 0.28 * trend_conf + 0.10 * vol_conf
        elif regime == "volatile":
            confidence += 0.25 * vol_conf + 0.05 * (1.0 - trend_conf)
        else:
            confidence += 0.18 * (1.0 - trend_conf) + 0.08 * (1.0 - vol_conf)

        confidence = self._clamp(confidence, 0.20, 0.99)

        self._dprint(
            "REGIME DEBUG | slope=", slope,
            "vol=", vol,
            "=>", regime
        )

        return RegimeResult(regime, vol, slope, confidence)