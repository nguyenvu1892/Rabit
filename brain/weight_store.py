# brain/weight_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import json
import os


@dataclass
class WeightCfg:
    # learning rate (EMA)
    alpha: float = 0.15
    # clamp to keep stable
    w_min: float = 0.20
    w_max: float = 5.00
    # default initial weight
    w0: float = 1.0


class WeightStore:
    """
    Store per-expert weight (optionally per-regime) and update with reward (EMA).

    Backward-compatible:
      - get(expert_name) still works.
      - get(expert_name, regime) also works.

    Persistence:
      - load_json(path), save_json(path)
    """

    def __init__(self, cfg: WeightCfg | None = None, path: str | None = None):
        self.cfg = cfg or WeightCfg()
        self._w: Dict[str, float] = {}
        self.path = path

    # ---------- keying ----------
    @staticmethod
    def _key(expert_name: str, regime: Optional[str] = None) -> str:
        # Prefer regime-specific weights if provided
        if regime:
            return f"{regime}::{expert_name}"
        return expert_name

    # ---------- basic ops ----------
    def get(self, expert_name: str, regime: Optional[str] = None) -> float:
        k = self._key(expert_name, regime)
        return float(self._w.get(k, self.cfg.w0))

    def set(self, expert_name: str, w: float, regime: Optional[str] = None) -> None:
        k = self._key(expert_name, regime)
        w = float(w)
        if w < self.cfg.w_min:
            w = self.cfg.w_min
        elif w > self.cfg.w_max:
            w = self.cfg.w_max
        self._w[k] = w

    def update(self, expert_name: str, reward: float, regime: Optional[str] = None) -> float:
        """
        EMA update: w <- clamp( w*(1-alpha) + f(reward)*alpha )
        where f(reward) = 1 + reward (small-signal).
        """
        w = self.get(expert_name, regime)
        target = 1.0 + float(reward)
        new_w = (1.0 - self.cfg.alpha) * w + self.cfg.alpha * target
        self.set(expert_name, new_w, regime=regime)
        return self.get(expert_name, regime)

    # ---------- reward helpers ----------
    @staticmethod
    def reward_from_outcome(win: Optional[bool], pnl: Optional[float]) -> float:
        """
        Safe small-signal reward:
          - win => +0.10, loss => -0.10
          - add tiny pnl contribution (clamped)
        """
        base = 0.0
        if win is True:
            base += 0.10
        elif win is False:
            base -= 0.10

        if pnl is not None:
            # pnl contribution is small & clamped for stability
            p = float(pnl)
            if p > 0:
                base += min(p, 10.0) / 100.0  # + up to +0.10
            elif p < 0:
                base -= min(abs(p), 10.0) / 100.0  # - up to -0.10
        return base

    def update_from_outcome(
        self,
        expert_name: str,
        win: Optional[bool],
        pnl: Optional[float],
        regime: Optional[str] = None,
    ) -> float:
        reward = self.reward_from_outcome(win, pnl)
        return self.update(expert_name, reward, regime=regime)

    # ---------- persistence ----------
    def to_dict(self) -> Dict[str, float]:
        return dict(self._w)

    def load_dict(self, d: Dict[str, Any]) -> None:
        self._w = {str(k): float(v) for k, v in (d or {}).items()}

    def load_json(self, path: str | None = None) -> None:
        p = path or self.path
        if not p:
            return
        if not os.path.exists(p):
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        # allow either {"weights": {...}} or plain dict
        if isinstance(data, dict) and "weights" in data and isinstance(data["weights"], dict):
            self.load_dict(data["weights"])
        elif isinstance(data, dict):
            self.load_dict(data)

    def save_json(self, path: str | None = None) -> None:
        p = path or self.path
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
