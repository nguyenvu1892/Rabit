# brain/weight_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
import json
import os
import time
import math


@dataclass
class WeightCfg:
    # Core learning
    lr: float = 0.03              # giảm nhẹ so với 0.05 để đỡ runaway
    decay: float = 0.001          # kéo dần về 1.0 mỗi lần update
    min_w: float = 0.2
    max_w: float = 10.0
    reward_clip: float = 1.0

    # Stabilizers
    lr_floor: float = 0.005       # LR tối thiểu sau khi giảm theo count
    lr_count_halflife: int = 400  # số lần update/key để LR giảm còn ~1/2
    power_default: float = 0.5    # weight -> multiplier: w**power (0.5 = sqrt)
    power_regime: float = 0.35    # regime tác động yếu hơn expert


class WeightStore:
    """
    Stores weights by (bucket -> key -> weight)
    Example:
      bucket="expert", key="MEAN_REVERT"
      bucket="regime", key="trend_up"

    Persistence:
      {"weights": {bucket: {key: weight}}, "meta": {...}}
    Also accepts legacy:
      {bucket: {key: weight}}
    """

    def __init__(self, path: Optional[str] = None, cfg: Optional[WeightCfg] = None) -> None:
        self.path = path
        self.cfg = cfg or WeightCfg()
        self._w: Dict[str, Dict[str, float]] = {}
        self._n: Dict[str, Dict[str, int]] = {}  # update counts (in-memory only)
        if self.path:
            self.load(self.path)

    # ------------------- IO -------------------
    def load(self, path: str) -> None:
        if not path or not os.path.exists(path):
            self._w = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._w = {}
            return

        if isinstance(data, dict) and "weights" in data and isinstance(data["weights"], dict):
            self._w = data["weights"]
        elif isinstance(data, dict):
            # legacy
            self._w = data
        else:
            self._w = {}

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.path
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        payload = {
            "weights": self._w,
            "meta": {
                "ts": int(time.time()),
                "cfg": {
                    "lr": self.cfg.lr,
                    "decay": self.cfg.decay,
                    "min_w": self.cfg.min_w,
                    "max_w": self.cfg.max_w,
                    "reward_clip": self.cfg.reward_clip,
                    "power_default": self.cfg.power_default,
                    "power_regime": self.cfg.power_regime,
                },
            },
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # backward compatible aliases
    def load_json(self, path: str) -> None:
        self.load(path)

    def save_json(self, path: str) -> None:
        self.save(path)

    # ------------------- core API -------------------
    def get(self, bucket: str, key: str, default: float = 1.0) -> float:
        try:
            return float(self._w.get(bucket, {}).get(key, default))
        except Exception:
            return float(default)

    def set(self, bucket: str, key: str, value: float) -> None:
        v = float(value)
        v = max(self.cfg.min_w, min(self.cfg.max_w, v))
        self._w.setdefault(bucket, {})[str(key)] = v

    def _bump_count(self, bucket: str, key: str) -> int:
        b = self._n.setdefault(bucket, {})
        b[key] = int(b.get(key, 0)) + 1
        return b[key]

    def _effective_lr(self, count: int) -> float:
        # LR giảm theo số lần update/key (anti runaway)
        hl = max(1, int(self.cfg.lr_count_halflife))
        base = float(self.cfg.lr)
        # dạng 1 / (1 + count/hl)
        lr_eff = base / (1.0 + (float(count) / float(hl)))
        return max(float(self.cfg.lr_floor), float(lr_eff))

    def update(self, bucket: str, key: str, reward: float) -> float:
        """
        Stable additive update:
          - clip reward
          - decay weight toward 1.0
          - lr decreases with update count for that key
          - clamp
        """
        r = float(reward)
        rc = float(self.cfg.reward_clip)
        if rc > 0:
            r = max(-rc, min(rc, r))

        k = str(key)
        w = self.get(bucket, k, default=1.0)

        # decay toward 1.0 first
        d = float(self.cfg.decay)
        if d > 0:
            w = w + (1.0 - w) * d

        cnt = self._bump_count(bucket, k)
        lr_eff = self._effective_lr(cnt)

        w2 = w + lr_eff * r
        w2 = max(self.cfg.min_w, min(self.cfg.max_w, float(w2)))
        self._w.setdefault(bucket, {})[k] = float(w2)
        return float(w2)

    # ------------------- stabilize helpers -------------------
    def decay_toward_one(self, bucket: str, rate: float = 0.001) -> None:
        rate = float(rate)
        if rate <= 0:
            return
        m = self._w.get(bucket, {})
        if not isinstance(m, dict):
            return
        for k, v in list(m.items()):
            try:
                vv = float(v)
                vv2 = vv + (1.0 - vv) * rate
                m[k] = max(self.cfg.min_w, min(self.cfg.max_w, float(vv2)))
            except Exception:
                continue

    def topk(self, bucket: str, k: int = 5) -> List[Tuple[str, float]]:
        m = self._w.get(bucket, {})
        if not isinstance(m, dict):
            return []
        items: List[Tuple[str, float]] = []
        for kk, vv in m.items():
            try:
                items.append((str(kk), float(vv)))
            except Exception:
                pass
        items.sort(key=lambda x: x[1], reverse=True)
        return items[: int(k)]

    def bottomk(self, bucket: str, k: int = 5) -> List[Tuple[str, float]]:
        m = self._w.get(bucket, {})
        if not isinstance(m, dict):
            return []
        items: List[Tuple[str, float]] = []
        for kk, vv in m.items():
            try:
                items.append((str(kk), float(vv)))
            except Exception:
                pass
        items.sort(key=lambda x: x[1])
        return items[: int(k)]

    # ------------------- reward shaping -------------------
    @staticmethod
    def outcome_reward(win: bool, pnl: float) -> float:
        """
        Output roughly in [-1, +1].
        """
        base = 0.6 if bool(win) else -0.6
        p = float(pnl)
        if p > 0:
            base += min(0.4, p / 10.0)
        elif p < 0:
            base -= min(0.4, abs(p) / 10.0)
        return float(base)

    # ------------------- multiplier helpers (for gate) -------------------
    def multiplier(self, bucket: str, key: str, default: float = 1.0, power: Optional[float] = None) -> float:
        """
        Convert weight to a SAFE multiplicative factor using power:
          mult = clamp(w) ** power
        power < 1 reduces dominance (sqrt is a good default).
        """
        w = self.get(bucket, key, default=default)
        w = max(self.cfg.min_w, min(self.cfg.max_w, float(w)))
        p = float(self.cfg.power_default if power is None else power)
        try:
            return float(math.pow(w, p))
        except Exception:
            return 1.0
