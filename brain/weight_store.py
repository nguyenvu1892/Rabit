# brain/weight_store.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class WeightStore:
    """
    Simple persistent weights store.

    Structure:
    {
      "expert":  {"MEAN_REVERT": 1.12, ...},
      "regime":  {"trend_up": 1.05, ...},
      "pattern": {...},
      "session": {...},
    }

    Design goals:
    - Backward compatible: allow get("KEY") defaulting to group="expert"
    - Defensive: never crash core loop
    """

    path: Optional[str] = None
    weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # learning params (safe defaults)
    lr: float = 0.05            # learning rate (small)
    decay: float = 0.0005       # tiny decay to prevent explosion
    w_min: float = 0.10
    w_max: float = 5.00
    default_weight: float = 1.0

    def ensure_group(self, group: str) -> None:
        if group not in self.weights or not isinstance(self.weights[group], dict):
            self.weights[group] = {}

    # --- Persistence ---
    def load_json(self, path: Optional[str] = None) -> None:
        p = path or self.path
        if not p:
            return
        if not os.path.exists(p):
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # only accept dict[str, dict[str,float]]
            cleaned: Dict[str, Dict[str, float]] = {}
            for g, mp in data.items():
                if not isinstance(mp, dict):
                    continue
                cleaned[g] = {}
                for k, v in mp.items():
                    try:
                        cleaned[g][str(k)] = float(v)
                    except Exception:
                        pass
            self.weights = cleaned

    def save_json(self, path: Optional[str] = None) -> None:
        p = path or self.path
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.weights, f, ensure_ascii=False, indent=2)

    # alias for convenience/back-compat
    def save(self) -> None:
        self.save_json(self.path)

    # --- Read/Write API ---
    def get(self, group_or_key: str, key: Optional[str] = None, default: Optional[float] = None) -> float:
        """
        Supports:
        - get("expert","MEAN_REVERT")
        - get("MEAN_REVERT") -> defaults group="expert"
        """
        if key is None:
            group = "expert"
            k = str(group_or_key)
        else:
            group = str(group_or_key)
            k = str(key)

        self.ensure_group(group)
        if default is None:
            default = self.default_weight

        try:
            return float(self.weights[group].get(k, float(default)))
        except Exception:
            return float(default)

    def set(self, group: str, key: str, value: float) -> None:
        self.ensure_group(group)
        self.weights[group][str(key)] = float(_clamp(float(value), self.w_min, self.w_max))

    def update(self, group: str, key: str, reward: float) -> float:
        """
        Multiplicative update (safe & stable):
        w <- clamp( w * (1 + lr*reward) - decay*w )
        """
        group = str(group)
        key = str(key)
        self.ensure_group(group)

        w0 = self.get(group, key, default=self.default_weight)
        r = float(reward)

        # multiplicative update + tiny decay
        w1 = w0 * (1.0 + self.lr * r)
        w1 = w1 - (self.decay * w1)

        w1 = _clamp(w1, self.w_min, self.w_max)
        self.weights[group][key] = float(w1)
        return float(w1)

    # --- Reward shaping ---
    def outcome_reward(self, win: bool, pnl: float = 0.0) -> float:
        """
        Map outcome -> reward in [-1, +1] (safe).
        Prefer win/loss signal, pnl as bonus.
        """
        base = 1.0 if bool(win) else -1.0

        # pnl bonus: compress strongly to avoid huge updates
        try:
            p = float(pnl)
        except Exception:
            p = 0.0

        # scale pnl to small [-0.5, +0.5] band
        bonus = _clamp(p / 100.0, -0.5, 0.5)
        return float(_clamp(base + bonus, -1.0, 1.0))
