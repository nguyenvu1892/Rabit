# brain/feature/volume.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import math


def _mean(xs):
    return sum(xs) / max(len(xs), 1)


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(var, 0.0))


@dataclass
class VolumeFeatures:
    """
    Volume-based features.
    candles expected keys: o,h,l,c and optional v
    """
    name: str = "vol"
    lookback: int = 50
    z_clip: float = 10.0

    def compute(self, candles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if candles is None or len(candles) < 2:
            return {}

        # require volume
        if "v" not in candles[-1]:
            return {}

        vols = []
        for x in candles[-min(self.lookback, len(candles)):]:
            if "v" in x and x["v"] is not None:
                vols.append(float(x["v"]))

        if len(vols) < 2:
            return {}

        last_v = float(vols[-1])
        m = _mean(vols[:-1])  # baseline from history excluding current
        s = _std(vols[:-1])

        z = 0.0 if s <= 1e-9 else (last_v - m) / s
        # clip to avoid extreme values dominating
        if z > self.z_clip:
            z = self.z_clip
        if z < -self.z_clip:
            z = -self.z_clip

        vol_ratio = 0.0 if abs(m) <= 1e-9 else last_v / m
        spike = 1 if z >= 2.0 else 0  # simple threshold for "unusual" volume

        return {
            "v": float(last_v),
            "v_mean": float(m),
            "v_std": float(s),
            "v_z": float(z),
            "v_ratio": float(vol_ratio),
            "v_spike": int(spike),
        }
