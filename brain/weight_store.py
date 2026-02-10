# brain/weight_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class WeightConfig:
    # learning
    lr: float = 0.05                 # learning rate
    reward_win: float = 1.0
    reward_loss: float = -1.0
    pnl_scale: float = 0.0           # 0.0 = ignore pnl magnitude, >0 use scaled pnl
    # safety bounds
    w_min: float = 0.25
    w_max: float = 3.0


class WeightStore:
    """
    Backward-compatible weight store.

    - Old usage: get(expert_name) -> weight
    - New usage: get(expert_name, regime) -> weight
    - New learning: update_from_outcome(expert_name, regime, win, pnl)
    """

    def __init__(self, path: str = "data/weights.json", cfg: Optional[WeightConfig] = None) -> None:
        self.path = Path(path)
        self.cfg = cfg or WeightConfig()
        self._w: Dict[str, float] = {}
        self.load()

    def _key(self, expert_name: str, regime: Optional[str] = None) -> str:
        # keep stable string key for json
        r = (regime or "ANY").strip() if isinstance(regime, str) else "ANY"
        return f"{expert_name}::{r}"

    def get(self, expert_name: str, regime: Optional[str] = None, default: float = 1.0) -> float:
        """
        Compatible:
          get("TREND_MA") -> reads TREND_MA::ANY if exists else fallback to old TREND_MA
          get("TREND_MA", "RANGE") -> reads TREND_MA::RANGE else fallback ANY
        """
        # new key
        k = self._key(expert_name, regime)
        if k in self._w:
            return float(self._w[k])

        # fallback to ANY regime
        k_any = self._key(expert_name, "ANY")
        if k_any in self._w:
            return float(self._w[k_any])

        # legacy fallback: some older versions stored directly by expert_name
        if expert_name in self._w:
            return float(self._w[expert_name])

        return float(default)

    def set(self, expert_name: str, value: float, regime: Optional[str] = None) -> None:
        v = float(value)
        v = max(self.cfg.w_min, min(self.cfg.w_max, v))
        self._w[self._key(expert_name, regime)] = v

    def update_from_outcome(
        self,
        expert_name: str,
        regime: Optional[str],
        win: bool,
        pnl: float = 0.0,
    ) -> float:
        """
        Update rule (safe):
          w <- clip( w + lr * (reward + pnl_scale * tanh(pnl)) )
        """
        base = self.get(expert_name, regime, default=1.0)

        reward = self.cfg.reward_win if bool(win) else self.cfg.reward_loss

        # optional pnl shaping
        shaped = 0.0
        if self.cfg.pnl_scale and pnl is not None:
            try:
                # smooth clamp with tanh-like behavior (no import math needed)
                x = float(pnl)
                # cheap stable squashing:
                shaped = x / (1.0 + abs(x))
            except Exception:
                shaped = 0.0

        delta = self.cfg.lr * (reward + self.cfg.pnl_scale * shaped)
        new_w = base + delta
        new_w = max(self.cfg.w_min, min(self.cfg.w_max, new_w))

        self._w[self._key(expert_name, regime)] = float(new_w)
        return float(new_w)

    # ---- persistence ----
    def load(self) -> None:
        try:
            if self.path.exists():
                obj = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    # keep only numeric weights
                    cleaned: Dict[str, float] = {}
                    for k, v in obj.items():
                        try:
                            cleaned[str(k)] = float(v)
                        except Exception:
                            continue
                    self._w = cleaned
        except Exception:
            # never crash the bot on weight file issues
            self._w = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._w, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def to_dict(self) -> Dict[str, float]:
        return dict(self._w)
