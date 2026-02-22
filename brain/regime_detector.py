# brain/regime_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
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

    def _compute_slope_vol(self, closes: List[float]) -> Tuple[float, float]:
        n = len(closes)
        if n < 3:
            return 0.0, 0.0
        w = closes[-self.window :] if n > self.window else closes
        if len(w) < 3:
            return 0.0, 0.0

        first = w[0]
        last = w[-1]
        denom = abs(first) if abs(first) > 1e-9 else 1.0
        slope = (last - first) / denom

        rets: List[float] = []
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

    # ==========================================================
    # COMPAT ADD: detect from precomputed features (FeaturePack)
    # - DO NOT remove old candle-based behavior
    # - If slope/vol already exist in features, use them
    # - Else try series keys (closes/close_series/etc)
    # - Else fallback to candles extraction (old logic)
    # ==========================================================
    def _pick_first_float(self, features: Dict[str, Any], keys: List[str]) -> float | None:
        for k in keys:
            if k in features:
                v = features.get(k)
                try:
                    if v is None:
                        continue
                    # allow dict wrapper
                    if isinstance(v, dict):
                        # common: {"value": x}
                        for kk in ("value", "v"):
                            if kk in v:
                                v = v[kk]
                                break
                    return float(v)
                except Exception:
                    continue
        return None

    def _extract_series_from_features(self, features: Dict[str, Any]) -> List[float]:
        # common series keys from feature packs
        for k in (
            "closes",
            "close_series",
            "close_list",
            "prices",
            "price_series",
            "series_close",
            "close",
        ):
            if k in features:
                v = features.get(k)
                # if single float, ignore
                if isinstance(v, (int, float)):
                    continue
                if isinstance(v, (list, tuple)):
                    out: List[float] = []
                    for x in v:
                        try:
                            out.append(float(x))
                        except Exception:
                            pass
                    if len(out) >= 3:
                        return out
        return []

    def detect(self, features: Dict[str, Any]) -> RegimeResult:
        if not isinstance(features, dict):
            return RegimeResult("unknown", 0.0, 0.0, 0.0)

        # ---------- COMPAT PATH 1: use precomputed slope/vol if available ----------
        # keep it generous because pack naming may vary
        slope = self._pick_first_float(
            features,
            keys=[
                "slope",
                "trend_slope",
                "ema_slope",
                "price_slope",
                "ret_slope",
                "linreg_slope",
            ],
        )
        vol = self._pick_first_float(
            features,
            keys=[
                "vol",
                "volatility",
                "ret_vol",
                "return_vol",
                "atr",
                "natr",
                "std",
                "std_ret",
                "range_vol",
            ],
        )

        if slope is None or vol is None:
            # ---------- COMPAT PATH 2: compute from series keys if present ----------
            closes_series = self._extract_series_from_features(features)
            if closes_series:
                s2, v2 = self._compute_slope_vol(closes_series)
                slope = s2 if slope is None else slope
                vol = v2 if vol is None else vol

        if slope is None or vol is None:
            # ---------- OLD PATH: fallback to candles (DO NOT REMOVE) ----------
            candles = features.get("candles")
            closes = self._extract_closes(candles)
            s3, v3 = self._compute_slope_vol(closes)
            slope = s3 if slope is None else slope
            vol = v3 if vol is None else vol

        # still None? hard fallback
        if slope is None:
            slope = 0.0
        if vol is None:
            vol = 0.0

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

        self._dprint("REGIME DEBUG | slope=", slope, "vol=", vol, "=>", regime)
        return RegimeResult(regime, float(vol), float(slope), float(confidence))