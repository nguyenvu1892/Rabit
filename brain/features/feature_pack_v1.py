# brain/features/feature_pack_v1.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import math


class FeaturePackV1:
    """
    FeaturePack-v1 (Foundation Layer)

    Goals:
    - Stable schema (fixed feature names)
    - Normalized values (ATR/close/zscore based)
    - No hard trading rules
    - Safe against bad candle formats

    Supported candle formats:
    1) dict with open/high/low/close (and optional volume)
    2) dict with o/h/l/c (and optional v)
    3) dict with single key/value string containing tab-separated fields:
       {'<DATE>\\t<TIME>\\t<OPEN>\\t<HIGH>\\t<LOW>\\t<CLOSE>\\t<TICKVOL>\\t<VOL>\\t<SPREAD>': '2021.11.17\\t04:45:00\\t1854.20\\t...'}
       or similar where the VALUE is the row string.
    """

    @staticmethod
    def compute(candles: List[Any]) -> Dict[str, float]:
        if not candles or len(candles) < 210:
            return {}

        ohlc = [FeaturePackV1._parse_candle(c) for c in candles]
        ohlc = [x for x in ohlc if x is not None]

        if len(ohlc) < 210:
            return {}

        closes = [x["close"] for x in ohlc]
        highs = [x["high"] for x in ohlc]
        lows = [x["low"] for x in ohlc]
        opens = [x["open"] for x in ohlc]
        volumes = [x.get("volume", 0.0) for x in ohlc]

        close = closes[-1]
        if not math.isfinite(close) or close <= 0:
            return {}

        features: Dict[str, float] = {}

        # ---------- Returns ----------
        features["ret_1"] = FeaturePackV1._safe_div(close - closes[-2], closes[-2])
        features["ret_5"] = FeaturePackV1._safe_div(close - closes[-6], closes[-6])

        # ---------- Candle structure ----------
        body = abs(close - opens[-1])
        rng = highs[-1] - lows[-1]
        features["range_nrm"] = FeaturePackV1._safe_div(rng, close)
        features["body_nrm"] = FeaturePackV1._safe_div(body, close)

        wick_up = highs[-1] - max(close, opens[-1])
        wick_dn = min(close, opens[-1]) - lows[-1]
        features["wick_up_nrm"] = FeaturePackV1._safe_div(wick_up, close)
        features["wick_dn_nrm"] = FeaturePackV1._safe_div(wick_dn, close)

        # ---------- ATR ----------
        atr14 = FeaturePackV1._atr(ohlc, 14)
        features["atr_14_nrm"] = FeaturePackV1._safe_div(atr14, close)

        # ---------- Volatility ----------
        rets = [FeaturePackV1._safe_div(closes[i] - closes[i - 1], closes[i - 1]) for i in range(1, len(closes))]
        features["volatility_14"] = FeaturePackV1._mean([abs(r) for r in rets[-14:]])

        # ---------- EMA ----------
        ema20 = FeaturePackV1._ema(closes, 20)
        ema50 = FeaturePackV1._ema(closes, 50)
        ema200 = FeaturePackV1._ema(closes, 200)

        features["ema_20_nrm"] = FeaturePackV1._safe_div(close - ema20, close)
        features["ema_50_nrm"] = FeaturePackV1._safe_div(close - ema50, close)
        features["ema_200_nrm"] = FeaturePackV1._safe_div(close - ema200, close)

        ema20_prev = FeaturePackV1._ema(closes[:-1], 20)
        features["ema20_slope"] = FeaturePackV1._safe_div(ema20 - ema20_prev, close)

        # ---------- RSI ----------
        features["rsi_14"] = FeaturePackV1._rsi(closes, 14) / 100.0

        # ---------- MACD ----------
        macd_line, signal, hist = FeaturePackV1._macd(closes)
        features["macd_hist_nrm"] = FeaturePackV1._safe_div(hist, close)
        macd_prev_hist = FeaturePackV1._macd(closes[:-1])[2]
        features["macd_slope"] = hist - macd_prev_hist

        # ---------- Volume ----------
        if any(v > 0 for v in volumes):
            features["vol_z_50"] = FeaturePackV1._zscore(volumes[-50:])
            sma20 = FeaturePackV1._mean(volumes[-20:])
            features["vol_rel_20"] = FeaturePackV1._safe_div(volumes[-1], sma20)

        return features

    # ==============================
    # Parsing + Helpers
    # ==============================

    @staticmethod
    def _parse_candle(c: Any) -> Optional[Dict[str, float]]:
        try:
            # ---- Format 1/2: dict with OHLC keys
            if isinstance(c, dict) and any(k in c for k in ("open", "high", "low", "close", "o", "h", "l", "c")):
                o = c.get("o", c.get("open"))
                h = c.get("h", c.get("high"))
                l = c.get("l", c.get("low"))
                cl = c.get("c", c.get("close"))
                if o is None or h is None or l is None or cl is None:
                    return None
                out = {
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(cl),
                }
                v = c.get("v", c.get("volume", 0.0))
                out["volume"] = float(v) if v is not None else 0.0
                return out

            # ---- Format 3: dict single key/value row string
            if isinstance(c, dict) and len(c) == 1:
                k, v = next(iter(c.items()))
                row = None
                if isinstance(v, str) and "\t" in v:
                    row = v
                elif isinstance(k, str) and "\t" in k:
                    row = k
                if row is None:
                    return None

                parts = row.split("\t")
                # expected: DATE, TIME, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD (>= 6)
                if len(parts) < 6:
                    return None

                # robust indexing (OPEN at 2)
                o = float(parts[2])
                h = float(parts[3])
                l = float(parts[4])
                cl = float(parts[5])

                vol = 0.0
                # VOL usually at index 7
                if len(parts) >= 8:
                    try:
                        vol = float(parts[7])
                    except Exception:
                        vol = 0.0

                return {"open": o, "high": h, "low": l, "close": cl, "volume": vol}

            return None
        except Exception:
            return None

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        try:
            if b == 0:
                return 0.0
            x = float(a) / float(b)
            if not math.isfinite(x):
                return 0.0
            return x
        except Exception:
            return 0.0

    @staticmethod
    def _mean(xs: List[float]) -> float:
        if not xs:
            return 0.0
        return sum(xs) / len(xs)

    @staticmethod
    def _ema(xs: List[float], period: int) -> float:
        if len(xs) < period:
            return xs[-1]
        k = 2 / (period + 1)
        ema = xs[-period]
        for x in xs[-period + 1 :]:
            ema = x * k + ema * (1 - k)
        return ema

    @staticmethod
    def _atr(ohlc: List[Dict[str, float]], period: int) -> float:
        trs = []
        for i in range(1, len(ohlc)):
            h = ohlc[i]["high"]
            l = ohlc[i]["low"]
            pc = ohlc[i - 1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        if len(trs) < period:
            return trs[-1] if trs else 0.0
        return FeaturePackV1._mean(trs[-period:])

    @staticmethod
    def _rsi(closes: List[float], period: int) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains = []
        losses = []
        for i in range(-period, 0):
            diff = closes[i] - closes[i - 1]
            if diff > 0:
                gains.append(diff)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(diff))
        avg_gain = FeaturePackV1._mean(gains)
        avg_loss = FeaturePackV1._mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _macd(closes: List[float]):
        # Lightweight MACD: EMA12 - EMA26
        ema12 = FeaturePackV1._ema(closes, 12)
        ema26 = FeaturePackV1._ema(closes, 26)
        macd_line = ema12 - ema26

        # Approx signal using macd series proxy (keep it stable without storing full history)
        # For foundation layer, we can set signal=macd_line (hist=0). Later we can upgrade to proper MACD series.
        signal = macd_line
        hist = macd_line - signal
        return macd_line, signal, hist

    @staticmethod
    def _zscore(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        mean = FeaturePackV1._mean(xs)
        var = FeaturePackV1._mean([(x - mean) ** 2 for x in xs])
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        return (xs[-1] - mean) / std