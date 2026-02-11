# brain/weight_store.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _now_ts() -> int:
    return int(time.time())


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class WeightSnapshot:
    ts: int
    path: str
    changed: int
    note: str = ""


class WeightStore:
    """
    Smart bucket weight store.

    Storage shape:
        weights[bucket][key] = float

    Examples:
        weights["expert"]["London"] = 1.2
        weights["regime:trend"]["London"] = 0.9
        weights["pattern"]["Engulf"] = 1.15
        weights["trend"]["up"] = 1.05

    Backward compatibility:
        get(expert, regime, default)  -> regime treated as bucket, expert treated as key
        bump(expert, delta, regime=...) works
    """

    def __init__(
        self,
        path: Optional[str] = None,
        default_weight: float = 1.0,
        min_w: float = 0.2,
        max_w: float = 5.0,
    ) -> None:
        self.path = path
        self.default_weight = float(default_weight)
        self.min_w = float(min_w)
        self.max_w = float(max_w)

        # main store
        self._w: Dict[str, Dict[str, float]] = {}

        # optional snapshots info
        self.last_snapshot: Optional[WeightSnapshot] = None

    # ---- compatibility alias ----
    @property
    def weights(self) -> Dict[str, Dict[str, float]]:
        return self._w

    # ---- core helpers ----
    def _bucket(self, bucket: Optional[str]) -> str:
        return str(bucket) if bucket else "global"

    def _ensure_bucket(self, bucket: str) -> Dict[str, float]:
        if bucket not in self._w:
            self._w[bucket] = {}
        return self._w[bucket]

    # ---- public API ----
    def get(self, key: str, bucket: Optional[str] = None, default: Optional[float] = None) -> float:
        """
        Preferred usage:
            get(key="London", bucket="expert")
            get(key="Engulf", bucket="pattern")

        Backward compatible usage elsewhere:
            get(expert_name, regime_bucket, default)
        """
        b = self._bucket(bucket)
        d = self.default_weight if default is None else float(default)
        return float(self._w.get(b, {}).get(str(key), d))

    def set(self, key: str, value: float, bucket: Optional[str] = None) -> float:
        b = self._bucket(bucket)
        kv = self._ensure_bucket(b)
        kv[str(key)] = float(value)
        return kv[str(key)]

    def multiplier(
        self,
        bucket: str,
        key: str,
        default: float = 1.0,
        power: float = 1.0,
    ) -> float:
        """
        Read weight and return (weight ** power), with clamping to safety range.
        """
        w = self.get(key=str(key), bucket=str(bucket), default=float(default))
        w = _clamp(float(w), self.min_w, self.max_w)

        p = float(power)
        if p == 1.0:
            return w
        # allow fractional power
        try:
            return float(math.pow(w, p))
        except Exception:
            return w

    def bump(
        self,
        key: str,
        delta: float,
        bucket: Optional[str] = None,
        clamp: bool = True,
    ) -> float:
        b = self._bucket(bucket)
        kv = self._ensure_bucket(b)

        cur = float(kv.get(str(key), self.default_weight))
        nxt = cur + float(delta)

        if clamp:
            nxt = _clamp(nxt, self.min_w, self.max_w)

        kv[str(key)] = float(nxt)
        return float(nxt)

    def decay_bucket_toward(
        self,
        bucket: str,
        target: float = 1.0,
        rate: float = 0.002,
        clamp: bool = True,
    ) -> int:
        """
        Exponential pull toward target:
            w <- w + (target - w) * rate
        """
        b = self._bucket(bucket)
        if b not in self._w:
            return 0

        changed = 0
        tgt = float(target)
        r = float(rate)

        for k, w in list(self._w[b].items()):
            nw = float(w) + (tgt - float(w)) * r
            if clamp:
                nw = _clamp(nw, self.min_w, self.max_w)
            if abs(nw - float(w)) > 1e-12:
                self._w[b][k] = float(nw)
                changed += 1
        return changed

    def clamp_bucket(self, bucket: str, min_w: Optional[float] = None, max_w: Optional[float] = None) -> int:
        b = self._bucket(bucket)
        if b not in self._w:
            return 0
        lo = self.min_w if min_w is None else float(min_w)
        hi = self.max_w if max_w is None else float(max_w)

        changed = 0
        for k, w in list(self._w[b].items()):
            nw = _clamp(float(w), lo, hi)
            if abs(nw - float(w)) > 1e-12:
                self._w[b][k] = float(nw)
                changed += 1
        return changed

    def normalize_bucket_mean(self, bucket: str, target_mean: float = 1.0) -> int:
        """
        Scale bucket so its mean becomes target_mean.
        """
        b = self._bucket(bucket)
        if b not in self._w or not self._w[b]:
            return 0

        vals = [float(v) for v in self._w[b].values()]
        mean = sum(vals) / max(1, len(vals))
        if mean <= 1e-12:
            return 0

        scale = float(target_mean) / mean
        changed = 0
        for k, w in list(self._w[b].items()):
            nw = float(w) * scale
            nw = _clamp(nw, self.min_w, self.max_w)
            if abs(nw - float(w)) > 1e-12:
                self._w[b][k] = float(nw)
                changed += 1
        return changed

    def topk(self, bucket: str, k: int = 5) -> List[Tuple[str, float]]:
        b = self._bucket(bucket)
        items = list(self._w.get(b, {}).items())
        items.sort(key=lambda x: float(x[1]), reverse=True)
        return [(str(a), float(bv)) for a, bv in items[: max(0, int(k))]]

    def bottomk(self, bucket: str, k: int = 5) -> List[Tuple[str, float]]:
        b = self._bucket(bucket)
        items = list(self._w.get(b, {}).items())
        items.sort(key=lambda x: float(x[1]))
        return [(str(a), float(bv)) for a, bv in items[: max(0, int(k))]]

    # ---- IO ----
    def load_json(self, path: Optional[str] = None) -> bool:
        p = path or self.path
        if not p:
            return False
        if not os.path.exists(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # data should be dict[bucket][key]=float
            out: Dict[str, Dict[str, float]] = {}
            if isinstance(data, dict):
                for b, kv in data.items():
                    if isinstance(kv, dict):
                        out[str(b)] = {str(k): _safe_float(v, self.default_weight) for k, v in kv.items()}
            self._w = out
            self.path = p
            return True
        except Exception:
            return False

    def save_json(self, path: Optional[str] = None) -> bool:
        p = path or self.path
        if not p:
            return False
        try:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._w, f, ensure_ascii=False, indent=2, sort_keys=True)
            self.path = p
            return True
        except Exception:
            return False

    def snapshot(self, path: Optional[str] = None, note: str = "") -> bool:
        """
        Save and register last_snapshot metadata.
        """
        p = path or self.path
        ok = self.save_json(p)
        if ok and p:
            self.last_snapshot = WeightSnapshot(ts=_now_ts(), path=p, changed=0, note=note)
        return ok
