# brain/regime_detector.py
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


def _extract_closes(candles: List[Any]) -> List[float]:
    closes: List[float] = []
    for c in candles:
        closes.append(_safe_float(_get(c, "close", _get(c, "CLOSE", 0.0)), 0.0))
    return closes


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / max(1, len(xs))
    return m, math.sqrt(max(0.0, var))


def _linear_slope(xs: List[float]) -> float:
    # least squares slope for x=0..n-1
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
class RegimeDetectorConfig:
    # ---- Feature-based thresholds (preferred) ----
    # slope_n is normalized slope by price: slope / close
    slope_n_trend: float = 0.00025  # trend threshold
    # volatility proxy from fpv1: ret_std or atr_n
    vol_range_max: float = 0.00120  # if vol is low and slope is small => range
    # breakout_pressure proxy: abs(macd_hist) + abs(slope_n)
    breakout_pressure_trend: float = 0.00035

    # confidence shaping
    conf_clip_min: float = 0.05
    conf_clip_max: float = 0.99

    # ---- Legacy candle-based thresholds (compat) ----
    slope_n_trend_legacy: float = 0.00025
    vol_range_max_legacy: float = 0.00120

    # windows (legacy)
    vol_window: int = 30
    slope_window: int = 30


class RegimeDetector:
    """
    RegimeDetector (5.1.9):
    - Prefer detect from FeaturePackV1 output (features dict)
    - Keep legacy candle-based detection for compat
    - Returns (regime: str, confidence: float)

    Regime keys:
      - "trend_up"
      - "trend_down"
      - "range"
      - "unknown" (fallback)
    """

    def __init__(self, cfg: Optional[RegimeDetectorConfig] = None, debug: bool = False):
        self.cfg = cfg or RegimeDetectorConfig()
        self.debug = bool(debug)

    # ----------------------------
    # Public API (keep stable)
    # ----------------------------
    def detect(
        self,
        candles: Optional[List[Any]] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        """
        Compat entrypoint:
        - If features provided and fpv1_ok => feature-based detect
        - Else if candles provided => legacy detect
        - Else => unknown
        """
        # Prefer feature-based (new foundation)
        if isinstance(features, dict):
            reg, conf = self.detect_from_features(features)
            if reg != "unknown":
                return reg, conf

        # Legacy fallback
        if candles:
            return self._detect_from_candles_legacy(candles)

        return "unknown", 0.0

    # ----------------------------
    # New: feature-based detection
    # ----------------------------
    def detect_from_features(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Uses FeaturePackV1 outputs if available.
        Expected keys (best-effort):
          fpv1_ok, slope_n, ret_std, atr_n, breakout_pressure, ema_gap, macd_hist
        """
        cfg = self.cfg

        fp_ok = bool(features.get("fpv1_ok", False))
        # NOTE: We still try even if fpv1_ok missing, but fp_ok helps avoid garbage.
        slope_n = _safe_float(features.get("slope_n", None), 0.0)
        ret_std = _safe_float(features.get("ret_std", None), 0.0)
        atr_n = _safe_float(features.get("atr_n", None), 0.0)
        breakout_pressure = _safe_float(features.get("breakout_pressure", None), 0.0)
        ema_gap = _safe_float(features.get("ema_gap", None), 0.0)
        macd_hist = _safe_float(features.get("macd_hist", None), 0.0)

        # If no fpv1_ok and we also have no meaningful features → unknown
        if (not fp_ok) and (abs(slope_n) <= 1e-12) and (ret_std <= 1e-12) and (atr_n <= 1e-12):
            return "unknown", 0.0

        # Vol proxy: prefer ret_std, else atr_n
        vol = ret_std if ret_std > 0 else atr_n

        # Determine direction preference:
        # - slope_n sign is primary
        # - ema_gap/macd_hist can reinforce
        direction_score = slope_n
        # small reinforcement without dominating
        direction_score += 0.25 * _safe_float(ema_gap, 0.0)
        direction_score += 0.10 * (_safe_float(macd_hist, 0.0))

        # Trend condition: slope strong enough and breakout pressure enough
        is_trend = (abs(slope_n) >= cfg.slope_n_trend) and (breakout_pressure >= cfg.breakout_pressure_trend)

        # Range condition: slope small and volatility low
        is_range = (abs(slope_n) < cfg.slope_n_trend) and (vol <= cfg.vol_range_max)

        if is_trend:
            regime = "trend_up" if direction_score >= 0 else "trend_down"
            conf = self._confidence_trend(abs(slope_n), breakout_pressure)
            return regime, conf

        if is_range:
            conf = self._confidence_range(vol, abs(slope_n))
            return "range", conf

        # Ambiguous: choose by slope sign but with lower confidence
        if abs(slope_n) > 0:
            regime = "trend_up" if direction_score >= 0 else "trend_down"
            conf = self._confidence_ambiguous(abs(slope_n), vol)
            return regime, conf

        return "unknown", 0.0

    # ----------------------------
    # Legacy: candle-based detection
    # ----------------------------
    def _detect_from_candles_legacy(self, candles: List[Any]) -> Tuple[str, float]:
        """
        Legacy detection from candles only.
        Kept to avoid breaking old logic. Uses slope_n + return std.
        """
        cfg = self.cfg
        closes = _extract_closes(candles)
        if len(closes) < 5:
            return "unknown", 0.0

        last = closes[-1]
        if last == 0.0:
            return "unknown", 0.0

        # returns std
        rets: List[float] = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            cur = closes[i]
            if prev != 0:
                rets.append((cur - prev) / prev)
        _, ret_s = _mean_std(rets[-cfg.vol_window :])

        # slope_n
        slope = _linear_slope(closes[-cfg.slope_window :])
        slope_n = slope / max(1e-9, last)

        # decide
        if abs(slope_n) >= cfg.slope_n_trend_legacy:
            regime = "trend_up" if slope_n >= 0 else "trend_down"
            # confidence from slope magnitude (legacy)
            conf = min(0.99, max(0.05, abs(slope_n) / (cfg.slope_n_trend_legacy * 3.0)))
            return regime, conf

        if ret_s <= cfg.vol_range_max_legacy:
            # likely range
            conf = min(0.90, max(0.05, (cfg.vol_range_max_legacy - ret_s) / max(1e-9, cfg.vol_range_max_legacy)))
            return "range", conf

        return "unknown", 0.0

    # ----------------------------
    # Confidence helpers
    # ----------------------------
    def _clip_conf(self, c: float) -> float:
        return float(min(self.cfg.conf_clip_max, max(self.cfg.conf_clip_min, c)))

    def _confidence_trend(self, slope_abs: float, breakout_pressure: float) -> float:
        # Scale relative to thresholds; keep stable 0..1
        a = slope_abs / max(1e-12, self.cfg.slope_n_trend * 4.0)
        b = breakout_pressure / max(1e-12, self.cfg.breakout_pressure_trend * 4.0)
        c = 0.5 * a + 0.5 * b
        return self._clip_conf(min(1.0, c))

    def _confidence_range(self, vol: float, slope_abs: float) -> float:
        # lower vol + lower slope => higher confidence
        v = 1.0 - (vol / max(1e-12, self.cfg.vol_range_max))
        s = 1.0 - (slope_abs / max(1e-12, self.cfg.slope_n_trend))
        c = 0.6 * v + 0.4 * s
        return self._clip_conf(min(1.0, max(0.0, c)))

    def _confidence_ambiguous(self, slope_abs: float, vol: float) -> float:
        # ambiguous => lower confidence
        a = slope_abs / max(1e-12, self.cfg.slope_n_trend * 6.0)
        b = 1.0 - (vol / max(1e-12, self.cfg.vol_range_max * 4.0))
        c = 0.35 * a + 0.25 * max(0.0, b)
        return self._clip_conf(min(0.60, max(0.05, c)))