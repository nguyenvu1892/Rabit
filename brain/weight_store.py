# brain/weight_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import json
import os


@dataclass
class WeightCfg:
    lr: float = 0.05          # learning rate
    decay: float = 0.0005     # per update decay toward 1.0
    min_w: float = 0.2
    max_w: float = 10.0
    reward_clip: float = 1.0  # clip reward magnitude


class WeightStore:
    """
    Stores weights by (bucket -> key -> weight)

    Example:
      bucket="expert", key="MEAN_REVERT"
      bucket="regime", key="trend_up"
      bucket="pattern", key="engulf"

    Persistence format:
      {"weights": {bucket: {key: weight}}}
    but also accepts old plain dict {bucket: {key: weight}}
    """

    def __init__(self, path: Optional[str] = None, cfg: Optional[WeightCfg] = None) -> None:
        self.path = path
        self.cfg = cfg or WeightCfg()
        self._w: Dict[str, Dict[str, float]] = {}

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
            self._w = data
        else:
            self._w = {}

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.path
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        payload = {"weights": self._w}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # --- backward compatible aliases (older callers use load_json/save_json) ---
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

    def update(self, bucket: str, key: str, reward: float) -> float:
        """
        Stable additive update with clipping and light decay toward 1.0:
          w <- decay_to_1(w) + lr * clip(reward)
        """
        r = float(reward)
        rc = float(self.cfg.reward_clip)
        if rc > 0:
            r = max(-rc, min(rc, r))

        w = self.get(bucket, key, default=1.0)

        # decay toward 1.0 first
        d = float(self.cfg.decay)
        if d > 0:
            w = w + (1.0 - w) * d

        lr = float(self.cfg.lr)
        w2 = w + lr * r
        w2 = max(self.cfg.min_w, min(self.cfg.max_w, float(w2)))

        self._w.setdefault(bucket, {})[str(key)] = float(w2)
        return float(w2)

    # ------------------- stabilize helpers (5.0.8.3) -------------------
    def decay_toward_one(self, bucket: str, rate: float = 0.001) -> None:
        """
        Extra decay pass toward 1.0 for a specific bucket (expert/regime).
        Use small rate: 0.0005 ~ 0.005.
        """
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
        Reward shape: win gives +, loss gives -, pnl adds small magnitude.
        Output roughly in [-1, +1].
        """
        base = 0.6 if bool(win) else -0.6
        p = float(pnl)

        if p > 0:
            base += min(0.4, p / 10.0)
        elif p < 0:
            base -= min(0.4, abs(p) / 10.0)

        return float(base)
