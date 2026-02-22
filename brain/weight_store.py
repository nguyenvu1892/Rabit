# brain/weight_store.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _now_ts() -> float:
    return time.time()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@dataclass
class WeightMeta:
    created_at: float
    updated_at: float
    version: str = "5.1.9"


class WeightStore:
    """
    Schema (KEEP):
      {
        "session": {...},
        "pattern": {...},
        "structure": {...},
        "trend": {...},
        "expert_regime": {
            "EXPERT|regime": weight,
            ...
        },
        "meta": { "created_at":..., "updated_at":..., "version":"..." }
      }

    Compat goals:
    - keep old buckets (session/pattern/structure/trend)
    - add stable expert_regime operations
    - accept legacy keys / signatures without breaking
    """

    def __init__(self, path: str = "data/weights.json", version: str = "5.1.9", **kwargs) -> None:
        # compat: accept weights_path alias
        path = str(kwargs.get("weights_path") or path)
        self.path = path

        now = _now_ts()
        self.data: Dict[str, Any] = {
            "session": {},
            "pattern": {},
            "structure": {},
            "trend": {},
            "expert_regime": {},
            "meta": {
                "created_at": now,
                "updated_at": now,
                "version": str(version),
            },
        }

        # auto-load if exists
        self.load(silent=True)

    # ---------- compat: dict-like ----------
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    # ---------- core ----------
    def _key(self, expert: Optional[str], regime: Optional[str]) -> str:
        e = str(expert or "UNKNOWN")
        r = str(regime or "unknown")
        return f"{e}|{r}"

    def get(self, expert: Optional[str] = None, regime: Optional[str] = None, default: float = 1.0) -> float:
        k = self._key(expert, regime)
        xs = self.data.get("expert_regime", {})
        if not isinstance(xs, dict):
            return float(default)
        return _safe_float(xs.get(k, default), float(default))

    def set(self, expert: Optional[str], regime: Optional[str], value: float) -> None:
        k = self._key(expert, regime)
        if not isinstance(self.data.get("expert_regime"), dict):
            self.data["expert_regime"] = {}
        self.data["expert_regime"][k] = float(value)
        self._touch()

    def update(self, expert: Optional[str], regime: Optional[str], delta: float, *, min_v: float = 0.05, max_v: float = 10.0) -> float:
        """
        Stability layer (light):
        - clamp weights to [min_v, max_v]
        - no schema changes
        """
        cur = self.get(expert, regime, default=1.0)
        nxt = float(cur) + float(delta)
        if nxt < float(min_v):
            nxt = float(min_v)
        if nxt > float(max_v):
            nxt = float(max_v)
        self.set(expert, regime, nxt)
        return nxt

    def _touch(self) -> None:
        meta = self.data.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            self.data["meta"] = meta
        meta["updated_at"] = _now_ts()

    # ---------- IO ----------
    def load(self, silent: bool = False) -> None:
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, dict):
                # merge but keep schema keys
                for k in ("session", "pattern", "structure", "trend", "expert_regime", "meta"):
                    if k in obj:
                        self.data[k] = obj[k]

                # ensure required keys exist
                self.data.setdefault("session", {})
                self.data.setdefault("pattern", {})
                self.data.setdefault("structure", {})
                self.data.setdefault("trend", {})
                self.data.setdefault("expert_regime", {})
                self.data.setdefault("meta", {"created_at": _now_ts(), "updated_at": _now_ts(), "version": "5.1.9"})
        except Exception:
            if not silent:
                raise

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._touch()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # compat aliases
    def dump(self) -> None:
        self.save()