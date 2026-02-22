# brain/features/feature_pack_v1.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _get(c: Any, key: str, default: Any = None) -> Any:
    # candle can be dict-like or object with attribute
    if isinstance(c, dict):
        return c.get(key, default)
    return getattr(c, key, default)


def _extract_ohlcv(candles: List[Any]) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    o, h, l, cl, v = [], [], [], [], []
    for c in candles:
        o.append(_safe_float(_get(c, "open", _get(c, "OPEN", 0.0)), 0.0))
        h.append(_safe_float(_get(c, "high", _get(c, "HIGH", 0.0)), 0.0))
        l.append(_safe_float(_get(c, "low", _get(c, "LOW", 0.0)), 0.0))
        cl.append(_safe_float(_get(c, "close", _get(c, "CLOSE", 0.0)), 0.0))
        v.append(_safe_float(_get(c, "volume", _get(c, "VOL", _get(c, "tick_volume", 0.0))), 0.0))
    return o, h, l, cl, v


def _ema(series: List[float], period: int) -> float:
    if not series or period <= 1:
        return series[-1] if series else 0.0
    alpha = 2.0 / (period + 1.0)
    e = series[0]
    for x in series[1:]:
        e = alpha * x + (1.0 - alpha) * e
    return e


def _rsi_wilder(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 2:
        return 50.0
    gains, losses = 0.0, 0.0
    # initial window
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / period
    avg_loss = losses / period

    # Wilder smoothing to the end
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        gain = d if d > 0 else 0.0
        loss = (-d) if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss <= 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(h: float, l: float, prev_close: float) -> float:
    return max(h - l, abs(h - prev_close), abs(l - prev_close))


def _atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    n = len(closes)
    if n < 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, n):
        trs.append(_true_range(highs[i], lows[i], closes[i - 1]))
    if len(trs) < period:
        return sum(trs) / max(1, len(trs))
    # Wilder ATR
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / max(1, len(xs))
    return m, math.sqrt(max(0.0, var))


def _linear_slope(xs: List[float]) -> float:
    # slope of y over x=0..n-1 (least squares), returns slope per step
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(xs) / n
    num, den = 0.0, 0.0
    for i, y in enumerate(xs):
        dx = i - x_mean
        num += dx * (y - y_mean)
        den += dx * dx
    if den <= 1e-12:
        return 0.0
    return num / den


@dataclass
class FeaturePackV1Config:
    ema_fast: int = 12
    ema_slow: int = 26
    ema_trend: int = 50
    macd_signal: int = 9
    rsi_period: int = 14
    atr_period: int = 14
    vol_window: int = 30
    slope_window: int = 30


class FeaturePackV1:
    """
    FeaturePackV1:
    - Pure feature extraction (NO entry rules)
    - Stable + deterministic
    - Consumes candles window, returns flat dict of numeric features
    """

    cfg = FeaturePackV1Config()

    @classmethod
    def compute(cls, candles: List[Any], cfg: Optional[FeaturePackV1Config] = None) -> Dict[str, Any]:
        cfg = cfg or cls.cfg
        if not candles:
            return {"fpv1_ok": False}

        o, h, l, c, v = _extract_ohlcv(candles)
        last_close = c[-1] if c else 0.0
        if last_close == 0.0:
            return {"fpv1_ok": False}

        # returns
        rets: List[float] = []
        for i in range(1, len(c)):
            prev = c[i - 1]
            cur = c[i]
            if prev != 0:
                rets.append((cur - prev) / prev)
        ret_m, ret_s = _mean_std(rets[-cfg.vol_window :])

        # trend slope (close)
        slope = _linear_slope(c[-cfg.slope_window :])
        # normalize slope by price
        slope_n = slope / max(1e-9, last_close)

        # EMA / MACD
        ema_fast = _ema(c, cfg.ema_fast)
        ema_slow = _ema(c, cfg.ema_slow)
        macd = ema_fast - ema_slow

        # signal line: EMA of macd series (approx by recomputing MACD history)
        # (stable + simple)
        macd_hist_series: List[float] = []
        # keep it cheap: compute over last 3*slow
        start = max(0, len(c) - cfg.ema_slow * 3)
        for i in range(start, len(c)):
            ema_f_i = _ema(c[start : i + 1], cfg.ema_fast)
            ema_s_i = _ema(c[start : i + 1], cfg.ema_slow)
            macd_hist_series.append(ema_f_i - ema_s_i)
        macd_signal = _ema(macd_hist_series, cfg.macd_signal) if macd_hist_series else 0.0
        macd_hist = macd - macd_signal

        ema_trend = _ema(c, cfg.ema_trend)
        ema_gap = (last_close - ema_trend) / max(1e-9, ema_trend)

        # RSI / ATR
        rsi = _rsi_wilder(c, cfg.rsi_period)
        atr = _atr(h, l, c, cfg.atr_period)
        atr_n = atr / max(1e-9, last_close)

        # volume stats (optional)
        vol_m, vol_s = _mean_std(v[-cfg.vol_window :])

        # structure-ish placeholders (not rules)
        # - range compression proxy: lower vol + low atr
        compression = 1.0 / (1.0 + (abs(ret_s) + atr_n))
        # - breakout pressure proxy: abs(macd_hist) + abs(slope_n)
        breakout_pressure = abs(macd_hist) + abs(slope_n)

        return {
            "fpv1_ok": True,
            # prices
            "px_close": last_close,
            # trend/vol
            "ret_mean": ret_m,
            "ret_std": ret_s,
            "slope": slope,
            "slope_n": slope_n,
            # ema/macd
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "ema_trend": ema_trend,
            "ema_gap": ema_gap,
            # rsi/atr
            "rsi": rsi,
            "atr": atr,
            "atr_n": atr_n,
            # volume
            "vol_mean": vol_m,
            "vol_std": vol_s,
            # stable “structure proxies”
            "compression": compression,
            "breakout_pressure": breakout_pressure,
        }