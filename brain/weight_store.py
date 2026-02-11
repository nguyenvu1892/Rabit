# brain/weight_store.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import json
import os

@dataclass
class WeightCfg:
    lr: float = 0.05          # learning rate
    decay: float = 0.0005     # per update decay
    min_w: float = 0.2
    max_w: float = 10.0
    reward_clip: float = 1.0  # clip reward magnitude


class WeightStore:
    """
    Stores weights by (bucket -> key -> weight)
    Example:
      bucket="trend", key="up"
      bucket="pattern", key="engulf"
      bucket="expert", key="MEAN_REVERT"
      bucket="regime", key="trend_up" (optional)
    """

    def __init__(self, path: Optional[str] = None, cfg: Optional[WeightCfg] = None) -> None:
        self.path = path
        self.cfg = cfg or WeightCfg()
        self._w: Dict[str, Dict[str, float]] = {}
        if self.path:
            self.load(self.path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            self._w = {}
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # accept both old formats
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

    def get(self, bucket: str, key: str, default: float = 1.0) -> float:
        return float(self._w.get(bucket, {}).get(key, default))

    def set(self, bucket: str, key: str, value: float) -> None:
        v = float(value)
        v = max(self.cfg.min_w, min(self.cfg.max_w, v))
        self._w.setdefault(bucket, {})[key] = v

    def decay_all(self) -> None:
        """Small decay towards 1.0 to prevent drifting forever."""
        d = float(self.cfg.decay)
        if d <= 0:
            return
        for bucket, m in self._w.items():
            for k, v in list(m.items()):
                # decay toward 1.0
                v2 = v + (1.0 - v) * d
                m[k] = max(self.cfg.min_w, min(self.cfg.max_w, float(v2)))

    def update(self, bucket: str, key: str, reward: float) -> float:
        """
        Update weight with clipped reward. Positive reward increases weight, negative decreases.
        Uses: w <- clamp( (1-decay)*w + lr*reward )
        """
        r = float(reward)
        rc = float(self.cfg.reward_clip)
        if rc > 0:
            r = max(-rc, min(rc, r))

        w = self.get(bucket, key, default=1.0)

        # apply light decay toward 1.0 first
        d = float(self.cfg.decay)
        if d > 0:
            w = w + (1.0 - w) * d

        lr = float(self.cfg.lr)
        w2 = w + lr * r

        w2 = max(self.cfg.min_w, min(self.cfg.max_w, float(w2)))
        self._w.setdefault(bucket, {})[key] = w2
        return w2

    @staticmethod
    def outcome_reward(win: bool, pnl: float) -> float:
        """
        Reward shape: win gives +, loss gives -, pnl adds small magnitude (clipped outside).
        Keep simple & stable.
        """
        base = 0.6 if win else -0.6
        # pnl scale: small contribution
        p = float(pnl)
        if p > 0:
            base += min(0.4, p / 10.0)
        elif p < 0:
            base -= min(0.4, abs(p) / 10.0)
        return base
        
    # --- backward compatible aliases (older callers use load_json/save_json) ---
    def load_json(self, path: str) -> None:
        self.load(path)

    def save_json(self, path: str) -> None:
        self.save(path)
